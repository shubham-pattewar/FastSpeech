import json
import os

import numpy as np
import soundfile as sf
import torch

from .audio import AudioProcessor
from .hifigan import load_vocoder
from .model import FastSpeech2
from .plotting import plot_mel, plot_variance
from .text import get_frontend


class Synthesizer:
    def __init__(self, checkpoint, vocoder=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.cfg = ckpt["config"]
        self.stats = ckpt["stats"]
        self.speakers = ckpt["speakers"]
        self.default_speaker = ckpt.get("default_speaker")
        self.step = ckpt["step"]

        self.frontend = get_frontend(self.cfg["language"])
        if self.frontend.symbols != ckpt["symbols"]:
            raise RuntimeError("symbol table of the checkpoint does not match the current front end")

        self.model = FastSpeech2(self.cfg, self.stats, len(self.frontend.symbols), len(self.speakers))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)
        self.ap = AudioProcessor(self.cfg["audio"])

        self.vocoder = None
        if vocoder and os.path.exists(vocoder):
            self.vocoder = load_vocoder(vocoder, self.device)
        else:
            print(f"HiFi-GAN checkpoint not found ({vocoder}), falling back to Griffin-Lim")

    def speaker_id(self, name=None):
        name = name or self.cfg.get("inference", {}).get("speaker") or self.default_speaker
        if name not in self.speakers:
            raise ValueError(f"unknown speaker {name!r}, choose from {list(self.speakers)[:10]}...")
        return self.speakers[name]

    @torch.no_grad()
    def __call__(self, text, speaker=None, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
        phones = self.frontend.text_to_phones(text)
        ids = torch.tensor([self.frontend.phones_to_ids(phones)], device=self.device)
        out = self.model(
            ids, torch.tensor([ids.size(1)], device=self.device),
            torch.tensor([self.speaker_id(speaker)], device=self.device),
            pitch_control=pitch_control, energy_control=energy_control, duration_control=duration_control,
        )
        n = int(out["mel_lens"][0])
        mel = out["mel_post"][0, :n].T  # [n_mels, T]

        if self.vocoder is not None:
            wav = self.vocoder(mel.unsqueeze(0)).squeeze().cpu().numpy()
        else:
            wav = self.ap.griffin_lim(mel.cpu().numpy(), self.cfg["inference"]["griffin_lim_iters"])

        ps, es = self.stats["pitch"], self.stats["energy"]
        return {
            "text": text,
            "phones": phones,
            "durations": out["durations"][0].cpu().numpy(),
            "pitch": out["pitch_pred"][0].cpu().numpy() * ps["std"] + ps["mean"],
            "energy": out["energy_pred"][0].cpu().numpy() * es["std"] + es["mean"],
            "mel": mel.cpu().numpy(),
            "wav": wav,
        }


def run_test_set(synth, sentences, results_dir, speaker=None, **controls):
    """Synthesise every test sentence and write audio, variance plots and the raw values."""
    audio_dir = os.path.join(results_dir, "test_generated_audios")
    plot_dir = os.path.join(results_dir, "variance_adaptor_plots")
    os.makedirs(audio_dir, exist_ok=True)
    hop_ms = synth.cfg["audio"]["hop_length"] / synth.cfg["audio"]["sampling_rate"] * 1000

    summary = {}
    for sent_id, text in sentences.items():
        res = synth(text, speaker=speaker, **controls)
        sf.write(os.path.join(audio_dir, f"{sent_id}.wav"), res["wav"], synth.ap.sr)
        plot_variance(res, sent_id, plot_dir, hop_ms)
        plot_mel(res["mel"], os.path.join(audio_dir, f"{sent_id}_mel.png"), f"{sent_id}: predicted mel")
        summary[sent_id] = {
            "text": text,
            "phones": res["phones"],
            "duration_frames": res["durations"].tolist(),
            "pitch_hz": np.round(res["pitch"], 2).tolist(),
            "energy": np.round(res["energy"], 3).tolist(),
            "seconds": round(len(res["wav"]) / synth.ap.sr, 2),
        }
        print(f"{sent_id}: {summary[sent_id]['seconds']} s | {' '.join(res['phones'])}")

    with open(os.path.join(results_dir, "variance_values.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint_step": synth.step, "sentences": summary}, f, indent=1, ensure_ascii=False)
