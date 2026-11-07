import librosa
import numpy as np
import torch
import torch.nn.functional as F


class AudioProcessor:
    """Mel / energy / F0 extraction.

    The mel computation follows HiFi-GAN's `mel_spectrogram` (reflect padding,
    center=False, natural-log magnitude mel) so a pretrained HiFi-GAN can be
    used as the vocoder without fine-tuning. Each frame then covers exactly
    `hop_length` samples, i.e. n_frames == len(wav) // hop_length.
    """

    def __init__(self, cfg):
        self.sr = cfg["sampling_rate"]
        self.n_fft = cfg["n_fft"]
        self.hop = cfg["hop_length"]
        self.win = cfg["win_length"]
        self.n_mels = cfg["n_mels"]
        self.fmin = cfg["mel_fmin"]
        self.fmax = cfg["mel_fmax"]
        self.f0_min = cfg.get("f0_min", 60)
        self.f0_max = cfg.get("f0_max", 600)

        mel = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        self.mel_basis = torch.from_numpy(mel).float()
        self.window = torch.hann_window(self.win)

    def load(self, path):
        wav, _ = librosa.load(path, sr=self.sr, mono=True)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.95
        return wav.astype(np.float32)

    def mel_and_energy(self, wav):
        y = torch.from_numpy(wav).float().view(1, 1, -1)
        pad = (self.n_fft - self.hop) // 2
        y = F.pad(y, (pad, pad), mode="reflect").view(1, -1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
                          center=False, return_complex=True)
        mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)[0]
        mel = torch.log(torch.clamp(self.mel_basis @ mag, min=1e-5))
        energy = torch.linalg.norm(mag, dim=0)
        return mel.numpy(), energy.numpy()

    def f0(self, wav, n_frames):
        try:
            import pyworld
            x = wav.astype(np.float64)
            f0, t = pyworld.dio(x, self.sr, f0_floor=self.f0_min, f0_ceil=self.f0_max,
                                frame_period=self.hop / self.sr * 1000)
            f0 = pyworld.stonemask(x, f0, t, self.sr)
        except ImportError:
            # much slower, but avoids a hard dependency on pyworld (no wheels on some setups)
            f0, _, _ = librosa.pyin(wav, fmin=self.f0_min, fmax=self.f0_max, sr=self.sr,
                                    frame_length=2048, hop_length=self.hop)
            f0 = np.nan_to_num(f0)
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        return f0[:n_frames].astype(np.float32)

    def griffin_lim(self, log_mel, n_iter=60):
        return librosa.feature.inverse.mel_to_audio(
            np.exp(log_mel), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            power=1.0, n_iter=n_iter, fmin=self.fmin, fmax=self.fmax,
        )
