import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.model import FastSpeech2, FastSpeech2Loss, length_regulate  # noqa: E402
from fastspeech2.preprocessing import SELECTION_FILE, run_feature_extraction, write_selection  # noqa: E402
from fastspeech2.text import get_frontend, get_symbols  # noqa: E402
from fastspeech2.text.telugu import number_to_words, ordinal_to_words  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(tmp_path, language="telugu"):
    cfg = load_config(ROOT / language.capitalize() / "config.yaml")
    cfg = copy.deepcopy(cfg)
    for key in ("corpus_dir", "textgrid_dir", "preprocessed_dir", "checkpoint_dir", "results_dir"):
        cfg["path"][key] = str(tmp_path / key)
    cfg["path"]["vocoder"] = str(tmp_path / "missing_vocoder.pth.tar")
    cfg["model"].update(encoder_layers=1, decoder_layers=1, encoder_hidden=32, decoder_hidden=32,
                        conv_filter_size=64, n_bins=16)
    cfg["model"]["variance_predictor"]["filter_size"] = 32
    cfg["model"]["postnet"].update(channels=32, layers=2)
    cfg["train"].update(batch_size=2, total_steps=4, log_step=1, val_step=2, save_step=2,
                        num_workers=0, amp=False, warmup_steps=10)
    cfg["inference"]["griffin_lim_iters"] = 2
    return cfg


def test_length_regulate():
    x = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
    out, lens = length_regulate(x, torch.tensor([[1, 0, 2]]))
    assert lens.tolist() == [3]
    assert out[0, :, 0].tolist() == [0.0, 4.0, 4.0]


def test_telugu_numbers():
    assert number_to_words(2026) == "రెండు వేల ఇరవై ఆరు"
    assert number_to_words(45) == "నలభై ఐదు"
    assert ordinal_to_words(12) == "పన్నెండవ"


def test_telugu_frontend():
    sys.path.insert(0, str(ROOT / "Telugu"))
    from inference import TEST_SENTENCES

    fe = get_frontend("telugu")
    for text in TEST_SENTENCES.values():
        phones = fe.text_to_phones(text)
        assert phones and all(p in fe.symbol_to_id for p in phones)
    assert "0" not in fe.normalize(TEST_SENTENCES["T4"])
    # inherent vowel after every consonant; anusvara before గ becomes the velar nasal
    assert fe.text_to_phones("అడవి") == ["a", "D", "a", "v", "i"]
    assert fe.text_to_phones("బలంగా") == ["b", "a", "l", "a", "ng", "g", "aa"]


def test_english_frontend():
    pytest.importorskip("g2p_en")
    fe = get_frontend("english")
    norm = fe.normalize("The meeting will start at 10:45 AM on 12 November 2026.")
    assert "ten forty five AM" in norm
    assert "twelfth November twenty twenty six" in norm
    noun = dict(fe.words("The orchestra's conduct was praised."))["conduct"]
    verb = dict(fe.words("They will conduct an experiment."))["conduct"]
    assert noun != verb


def test_extract_parquet(tmp_path):
    import io
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from fastspeech2.hf_parquet import extract_parquet, load_manifest

    sys.path.insert(0, str(ROOT / "Telugu"))
    from preprocess import gender_speaker

    rows = []
    for i in range(3):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(48000, dtype=np.float32), 48000, format="WAV")
        rows.append({"audio": {"bytes": buf.getvalue(), "path": f"x{i}.wav"}, "text": "పులి", "gender": i % 2})
    table = pa.Table.from_pylist(rows)
    features = {"gender": {"names": ["female", "male"], "_type": "ClassLabel"}}
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    (tmp_path / "pq").mkdir()
    pq.write_table(table, tmp_path / "pq" / "train-0.parquet")

    counts = extract_parquet(tmp_path / "pq", tmp_path / "raw", 22050, gender_speaker, "tel")
    assert counts == {"female": 2, "male": 1}
    utts = load_manifest(tmp_path / "raw", ["female"])
    assert len(utts) == 2 and abs(utts[0]["duration"] - 1.0) < 1e-3
    assert sf.info(utts[0]["path"]).samplerate == 22050


def test_model_forward_backward(tmp_path):
    cfg = tiny_config(tmp_path)
    stats = {"pitch": {"min": -2, "max": 2, "mean": 150, "std": 30},
             "energy": {"min": -2, "max": 2, "mean": 20, "std": 5}}
    symbols = get_symbols("telugu")
    model = FastSpeech2(cfg, stats, len(symbols), n_speakers=2)
    texts = torch.randint(1, len(symbols), (2, 7))
    src_lens = torch.tensor([7, 5])
    durations = torch.randint(1, 4, (2, 7))
    durations[1, 5:] = 0
    mel_lens = durations.sum(1)
    T = int(mel_lens.max())
    batch = {
        "mels": torch.randn(2, T, 80), "pitches": torch.randn(2, 7), "energies": torch.randn(2, 7),
        "durations": durations,
    }
    out = model(texts, src_lens, torch.tensor([0, 1]), mel_lens, T, batch["pitches"], batch["energies"], durations)
    loss, _ = FastSpeech2Loss()(out, batch)
    loss.backward()
    assert out["mel_post"].shape == (2, T, 80)

    model.eval()
    with torch.no_grad():
        out = model(texts[:1], src_lens[:1], torch.tensor([0]), pitch_control=1.2, duration_control=0.8)
    assert out["mel_post"].shape[1] == int(out["mel_lens"][0])


def _write_textgrid(path, phones, frame_sec, lead=0.1):
    import tgt

    tier = tgt.IntervalTier(name="phones")
    t = lead
    tier.add_interval(tgt.Interval(0.0, t, ""))
    for p in phones:
        tier.add_interval(tgt.Interval(t, t + 4 * frame_sec, p))
        t += 4 * frame_sec
    tier.add_interval(tgt.Interval(t, t + 0.1, ""))
    tg = tgt.TextGrid()
    tg.add_tier(tier)
    tgt.io.write_to_file(tg, str(path), format="long")
    return t + 0.1


def test_end_to_end(tmp_path):
    pytest.importorskip("tgt")
    from fastspeech2.synthesis import Synthesizer, run_test_set
    from fastspeech2.trainer import train

    cfg = tiny_config(tmp_path)
    fe = get_frontend("telugu")
    sr, hop = cfg["audio"]["sampling_rate"], cfg["audio"]["hop_length"]
    texts = ["పులి నడిచింది", "తాడు బలంగా ఉంది", "ఆమె వచ్చింది", "సమయానికి రండి", "నూతన ఔషధం", "చీర ధరించి"]

    utts = []
    rng = np.random.default_rng(0)
    for i, text in enumerate(texts):
        speaker = "female" if i % 2 == 0 else "male"
        utt_id = f"utt{i}"
        spk_dir = tmp_path / "corpus_dir" / speaker
        spk_dir.mkdir(parents=True, exist_ok=True)
        phones = [p for _, ps in fe.words(text) for p in ps]
        total = _write_textgrid(tmp_path / f"{utt_id}.TextGrid", phones, hop / sr)
        os.makedirs(tmp_path / "textgrid_dir", exist_ok=True)
        os.replace(tmp_path / f"{utt_id}.TextGrid", tmp_path / "textgrid_dir" / f"{utt_id}.TextGrid")
        t = np.arange(int(total * sr)) / sr
        wav = 0.3 * np.sin(2 * np.pi * (120 + 40 * i) * t) + 0.01 * rng.standard_normal(len(t))
        sf.write(spk_dir / f"{utt_id}.wav", wav.astype(np.float32), sr)
        utts.append({"id": utt_id, "speaker": speaker, "duration": total, "text": text})

    os.makedirs(cfg["path"]["preprocessed_dir"], exist_ok=True)
    write_selection(os.path.join(cfg["path"]["preprocessed_dir"], SELECTION_FILE), utts[:4], utts[4:])
    run_feature_extraction(cfg, fe, workers=1)

    pre = Path(cfg["path"]["preprocessed_dir"])
    assert len((pre / "train.txt").read_text(encoding="utf-8").splitlines()) == 4
    mel = np.load(pre / "mel" / "utt0.npy")
    dur = np.load(pre / "duration" / "utt0.npy")
    assert mel.shape[0] == dur.sum()

    train(cfg, device="cpu")
    ckpt_dir = Path(cfg["path"]["checkpoint_dir"])
    assert (ckpt_dir / "best.pt").exists() and (ckpt_dir / "last.pt").exists()
    train(cfg, resume=str(ckpt_dir / "last.pt"), device="cpu")  # already at total_steps, should just exit

    results = Path(cfg["path"]["results_dir"])
    assert (results / "training_loss.png").exists() and (results / "validation_loss.png").exists()

    synth = Synthesizer(str(ckpt_dir / "best.pt"), cfg["path"]["vocoder"], device="cpu")
    run_test_set(synth, {"T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.", "T4": "2026 నవంబర్ 12వ తేదీ 10:45"},
                 str(results), speaker="female")
    assert (results / "test_generated_audios" / "T4.wav").exists()
    for kind in ("duration", "pitch", "energy"):
        assert (results / "variance_adaptor_plots" / kind / "T2a.png").exists()
