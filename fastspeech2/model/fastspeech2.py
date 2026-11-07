import torch.nn as nn

from .modules import FFTStack, PostNet, VarianceAdaptor, get_mask_from_lengths


class FastSpeech2(nn.Module):
    def __init__(self, cfg, stats, n_symbols, n_speakers=1):
        super().__init__()
        m = cfg["model"]
        assert m["encoder_hidden"] == m["decoder_hidden"], "encoder and decoder hidden sizes must match"
        d = m["encoder_hidden"]
        n_mels = cfg["audio"]["n_mels"]

        self.embed = nn.Embedding(n_symbols, d, padding_idx=0)
        self.encoder = FFTStack(m["encoder_layers"], d, m["encoder_heads"], m)
        self.variance_adaptor = VarianceAdaptor(m, stats)
        self.decoder = FFTStack(m["decoder_layers"], d, m["decoder_heads"], m)
        self.mel_linear = nn.Linear(d, n_mels)
        p = m["postnet"]
        self.postnet = PostNet(n_mels, p["channels"], p["kernel_size"], p["layers"], p["dropout"])

        # Svarah has ~100 speakers, IndicTTS Telugu has two, so the speaker table is optional
        self.speaker_embedding = nn.Embedding(n_speakers, d) if n_speakers > 1 else None

    def forward(self, texts, src_lens, speakers=None, mel_lens=None, max_mel_len=None,
                pitches=None, energies=None, durations=None,
                pitch_control=1.0, energy_control=1.0, duration_control=1.0):
        src_mask = get_mask_from_lengths(src_lens, texts.size(1))
        x = self.encoder(self.embed(texts), src_mask)

        if self.speaker_embedding is not None and speakers is not None:
            x = x + self.speaker_embedding(speakers).unsqueeze(1)

        x, var = self.variance_adaptor(
            x, src_mask, max_mel_len, pitches, energies, durations,
            pitch_control, energy_control, duration_control,
        )
        if mel_lens is None:
            mel_lens = var["mel_lens"]
        mel_mask = get_mask_from_lengths(mel_lens, x.size(1))

        x = self.decoder(x, mel_mask)
        mel = self.mel_linear(x).masked_fill(mel_mask.unsqueeze(-1), 0.0)
        mel_post = (mel + self.postnet(mel)).masked_fill(mel_mask.unsqueeze(-1), 0.0)

        var.update(mel=mel, mel_post=mel_post, src_mask=src_mask, mel_mask=mel_mask, mel_lens=mel_lens)
        return var
