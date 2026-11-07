import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    # True on padded positions
    if max_len is None:
        max_len = int(lengths.max())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) >= lengths.unsqueeze(1)


def sinusoid_table(n_position, d_model):
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(n_position, d_model)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.register_buffer("table", sinusoid_table(max_len, d_model), persistent=False)

    def forward(self, x):
        T = x.size(1)
        if T <= self.table.size(0):
            table = self.table[:T]
        else:  # longer than anything seen in training, just build a bigger table
            table = sinusoid_table(T, x.size(2)).to(x)
        return x + table.unsqueeze(0)


class ConvFeedForward(nn.Module):
    def __init__(self, d_model, d_filter, kernel_sizes, dropout):
        super().__init__()
        k1, k2 = kernel_sizes
        self.w1 = nn.Conv1d(d_model, d_filter, k1, padding=(k1 - 1) // 2)
        self.w2 = nn.Conv1d(d_filter, d_model, k2, padding=(k2 - 1) // 2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.w2(F.relu(self.w1(x.transpose(1, 2)))).transpose(1, 2)
        return self.norm(x + self.dropout(y))


class FFTBlock(nn.Module):
    """Self-attention followed by the 1D-conv feed-forward (FastSpeech FFT block)."""

    def __init__(self, d_model, n_heads, d_filter, kernel_sizes, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = ConvFeedForward(d_model, d_filter, kernel_sizes, dropout)

    def forward(self, x, mask):
        y, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        x = self.norm(x + self.dropout(y)).masked_fill(mask.unsqueeze(-1), 0.0)
        return self.ffn(x).masked_fill(mask.unsqueeze(-1), 0.0)


class FFTStack(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, cfg):
        super().__init__()
        self.pos = PositionalEncoding(d_model, cfg["max_seq_len"])
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, cfg["conv_filter_size"], cfg["conv_kernel_size"], cfg["dropout"])
            for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class VariancePredictor(nn.Module):
    """conv-relu-ln-dropout x2 + linear. Used for duration, pitch and energy."""

    def __init__(self, d_in, d_filter, kernel_size, dropout):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_in, d_filter, kernel_size, padding=pad)
        self.norm1 = nn.LayerNorm(d_filter)
        self.conv2 = nn.Conv1d(d_filter, d_filter, kernel_size, padding=pad)
        self.norm2 = nn.LayerNorm(d_filter)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_filter, 1)

    def forward(self, x, mask):
        y = F.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(self.norm1(y))
        y = F.relu(self.conv2(y.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(self.norm2(y))
        return self.proj(y).squeeze(-1).masked_fill(mask, 0.0)


def length_regulate(x, durations, max_len=None):
    """Repeat every phoneme vector `duration` times. x: [B, N, D], durations: [B, N]."""
    expanded = [torch.repeat_interleave(xi, di, dim=0) for xi, di in zip(x, durations)]
    lens = torch.tensor([e.size(0) for e in expanded], device=x.device, dtype=torch.long)
    if max_len is None:
        max_len = max(1, int(lens.max()))
    out = x.new_zeros(x.size(0), max_len, x.size(2))
    for i, e in enumerate(expanded):
        n = min(e.size(0), max_len)
        out[i, :n] = e[:n]
    return out, lens.clamp(max=max_len)


class VarianceAdaptor(nn.Module):
    """Duration / pitch / energy predictors + length regulator.

    Pitch and energy are phoneme level and z-normalised in preprocessing. `stats`
    holds min/max of the normalised values (for the quantisation bins) and the
    raw mean/std so that pitch/energy control can be applied in Hz / magnitude.
    """

    def __init__(self, cfg, stats):
        super().__init__()
        d = cfg["encoder_hidden"]
        vp = cfg["variance_predictor"]
        self.duration_predictor = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        self.pitch_predictor = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        self.energy_predictor = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])

        n_bins = cfg["n_bins"]
        self.stats = stats
        self.register_buffer("pitch_bins", torch.linspace(stats["pitch"]["min"], stats["pitch"]["max"], n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(stats["energy"]["min"], stats["energy"]["max"], n_bins - 1))
        self.pitch_embedding = nn.Embedding(n_bins, d)
        self.energy_embedding = nn.Embedding(n_bins, d)

    def _apply_control(self, value, name, control):
        if control == 1.0:
            return value
        mean, std = self.stats[name]["mean"], self.stats[name]["std"]
        return ((value * std + mean) * control - mean) / std

    def forward(self, x, src_mask, max_mel_len=None, pitch_target=None, energy_target=None,
                duration_target=None, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
        log_duration_pred = self.duration_predictor(x, src_mask)

        pitch_pred = self.pitch_predictor(x, src_mask)
        if pitch_target is not None:
            pitch = pitch_target
        else:
            pitch = self._apply_control(pitch_pred, "pitch", pitch_control)
        x = x + self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))

        energy_pred = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            energy = energy_target
        else:
            energy = self._apply_control(energy_pred, "energy", energy_control)
        x = x + self.energy_embedding(torch.bucketize(energy, self.energy_bins))

        if duration_target is not None:
            durations = duration_target
        else:
            # predictor works on log(d + 1); keep at least one frame per phoneme
            durations = torch.round((torch.exp(log_duration_pred) - 1) * duration_control)
            durations = durations.clamp(min=1).long().masked_fill(src_mask, 0)

        x, mel_lens = length_regulate(x, durations, max_mel_len)
        return x, {
            "log_duration_pred": log_duration_pred,
            "pitch_pred": pitch,
            "energy_pred": energy,
            "pitch_raw_pred": pitch_pred,
            "energy_raw_pred": energy_pred,
            "durations": durations,
            "mel_lens": mel_lens,
        }


class PostNet(nn.Module):
    def __init__(self, n_mels, channels=512, kernel_size=5, n_layers=5, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(n_layers):
            c_in = n_mels if i == 0 else channels
            c_out = n_mels if i == n_layers - 1 else channels
            layers.append(nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(c_out),
            ))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = x.transpose(1, 2)
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1:
                y = torch.tanh(y)
            y = self.dropout(y)
        return y.transpose(1, 2)
