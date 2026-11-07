import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    """mel L1 (before and after postnet) + MSE for pitch, energy and log duration."""

    def forward(self, out, batch):
        src_valid = ~out["src_mask"]
        mel_valid = ~out["mel_mask"]
        mel_target = batch["mels"][:, : mel_valid.size(1)]

        mel_loss = F.l1_loss(out["mel"][mel_valid], mel_target[mel_valid])
        postnet_loss = F.l1_loss(out["mel_post"][mel_valid], mel_target[mel_valid])
        pitch_loss = F.mse_loss(out["pitch_raw_pred"][src_valid], batch["pitches"][src_valid])
        energy_loss = F.mse_loss(out["energy_raw_pred"][src_valid], batch["energies"][src_valid])
        log_d_target = torch.log(batch["durations"].float() + 1)
        duration_loss = F.mse_loss(out["log_duration_pred"][src_valid], log_d_target[src_valid])

        total = mel_loss + postnet_loss + pitch_loss + energy_loss + duration_loss
        parts = {
            "total": total.item(),
            "mel": mel_loss.item(),
            "postnet": postnet_loss.item(),
            "pitch": pitch_loss.item(),
            "energy": energy_loss.item(),
            "duration": duration_loss.item(),
        }
        return total, parts
