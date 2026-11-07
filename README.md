# FastSpeech 2 for English and Telugu

FastSpeech 2 ([Ren et al., 2021](https://arxiv.org/abs/2006.04558)) trained separately for
English (Svarah) and Telugu (IndicTTS Telugu). The model follows the structure of
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2): FFT encoder/decoder, a
variance adaptor with phoneme-level pitch and energy, a postnet, and a pretrained HiFi-GAN
as the vocoder.

```
English/            preprocess.py, train.py, inference.py, config.yaml, checkpoints/
Telugu/             same for Telugu
fastspeech2/        code shared by both languages (model, front ends, features, training)
hifigan/            put generator_universal.pth.tar here
Results/<lang>/     loss curves, generated test audio, variance adaptor plots
Report/             technical report
```

## Setup

```bash
pip install -r requirements.txt
```

Montreal Forced Aligner (for phoneme durations) is easiest to install with conda, in a separate env:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
```

Vocoder: download `generator_universal.pth.tar.zip` from
[ming024/FastSpeech2/hifigan](https://github.com/ming024/FastSpeech2/tree/master/hifigan),
unzip it into `hifigan/`. Without it, inference falls back to Griffin-Lim.

All commands below are run from the repository root.

## English (Svarah)

Svarah: 9.6 h of Indian-accented English from 117 speakers
([AI4Bharat/Svarah](https://github.com/AI4Bharat/Svarah)). Extract it to `data/English/Svarah`
so that `svarah_manifest.json` and `audio/` are directly inside.

```bash
python English/preprocess.py select
conda run -n mfa mfa model download acoustic english_us_arpa
conda run -n mfa mfa align data/English/mfa_corpus data/English/lexicon.txt english_us_arpa data/English/TextGrid --clean -j 4
python English/preprocess.py features
python English/train.py
python English/inference.py
```

## Telugu (IndicTTS Telugu)

```bash
huggingface-cli download SPRINGLab/IndicTTS_Telugu --repo-type dataset --local-dir data/Telugu/IndicTTS_Telugu
python Telugu/preprocess.py extract
python Telugu/preprocess.py select
conda run -n mfa mfa train data/Telugu/mfa_corpus data/Telugu/lexicon.txt data/Telugu/telugu_mfa.zip --output_directory data/Telugu/TextGrid --clean -j 4
python Telugu/preprocess.py features
python Telugu/train.py
python Telugu/inference.py
```

`inference.py` without arguments synthesises the assignment test sentences into
`Results/<lang>/test_generated_audios/` and writes duration / pitch / energy plots to
`Results/<lang>/variance_adaptor_plots/`. The raw predicted values are also saved in
`Results/<lang>/variance_values.json`. A single sentence:

```bash
python English/inference.py --text "Hello, how are you?" --out hello.wav
python Telugu/inference.py --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" --speaker female --out namaskaram.wav
```

`--pitch_control`, `--energy_control` and `--duration_control` scale the predictions
(pitch and energy are scaled in Hz / magnitude, not in normalised units).

## Data selection

Both languages use the same procedure (`fastspeech2/preprocessing.py`):

1. Keep utterances between `min_duration` and `max_duration` seconds (1 to 12 s English,
   1 to 15 s Telugu) that have non-empty text after normalisation.
2. Group utterances by their normalised sentence, shuffle the groups with a fixed seed,
   and fill validation up to 1 h first, then training up to 5 h. Because whole groups are
   assigned, no utterance and no sentence appears in both splits.
3. The selected ids and splits are written to `data/<lang>/preprocessed/selection.tsv`.

For Telugu, the female speaker alone has only ~4.3 h, so both speakers are pooled and the model
gets a speaker embedding (inference uses the female voice by default). For Svarah, the manifest
has no speaker field, so the speaker is parsed from the file name (`speaker_regex` in the config).

## Changes compared to ming024/FastSpeech2

Shared:
- Durations come from our own lexicon: the front end used at inference also writes the MFA
  pronunciation dictionary, so training and inference use the same phoneme inventory.
- Mel extraction matches HiFi-GAN (reflect padding, `center=False`) and the audio is cut on
  frame boundaries, so `sum(durations) == n_frames` exactly with no trimming fix-ups.
- Pitch/energy statistics are computed on the training split only, and pitch/energy control
  is applied in the unnormalised domain.
- Speaker embedding is optional and switched on automatically when there is more than one speaker.
- Mixed precision training, a CSV loss log, and loss plots updated at every validation step.

English:
- Text normalisation for times (`10:45 AM` -> "ten forty five A M"), dates
  (`12 November 2026` -> "twelfth November twenty twenty six"), ordinals and plain numbers.
- Heteronyms are resolved with g2p_en's POS-based homograph table (E2: *conduct* as a noun
  vs. a verb). Out-of-vocabulary words (E1, the code-mixed *andaroo* in E4, the split word in E3)
  go through g2p_en's neural G2P.
- Fix for nltk >= 3.9, which renamed the POS tagger resource that g2p_en asks for.

Telugu:
- Rule-based grapheme-to-phoneme conversion of the Telugu script (inherent /a/, virama,
  vowel signs, anusvara assimilated to the following consonant), see `fastspeech2/text/telugu.py`.
- Telugu number normalisation, including ordinals (`12వ` -> పన్నెండవ), times and years.
- There is no pretrained Telugu MFA model, so MFA is trained on the selected corpus with the
  rule-based lexicon.
- Data loading from the HuggingFace parquet release (48 kHz audio, resampled to 22.05 kHz).

## Checkpoints

`English/checkpoints/best.pt` and `Telugu/checkpoints/best.pt` hold the model with the
lowest validation loss along with the config, symbol table and pitch/energy statistics, so they
are all inference needs. They are tracked with Git LFS (`.gitattributes`):

```bash
git lfs install
git lfs track "*.pt"
```

## Tests

```bash
pytest -q tests
```

The test builds a tiny synthetic dataset (fake audio + TextGrids), runs feature extraction,
a few training steps and inference, so the whole pipeline can be checked without the real data.
