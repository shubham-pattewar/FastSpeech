# FastSpeech 2 Text-to-Speech for English and Telugu - Technical Report

> Draft. Sections 6-9 depend on the trained models and are marked **[TODO after training]**.
> Convert to `Technical_Report.pdf` once they are filled in.

## 1. Objective

Two separate FastSpeech 2 models were trained: one for English on Svarah (Indian-accented
English) and one for Telugu on the IndicTTS Telugu corpus, each with about 5 h of training and
1 h of validation audio. Both models synthesise a fixed set of test sentences. The report looks
at the duration, pitch and energy that the variance adaptor predicts for those sentences.

References: Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (ICLR 2021);
reference implementation [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2).

## 2. Implementation approach and architecture

The model follows the paper and the ming024 implementation. The hyper-parameters are the same
for both languages:

| Component | Configuration | Parameters |
|---|---|---|
| Phoneme embedding | 256-d (72 English / 49 Telugu symbols) | 0.02 M |
| Encoder | 4 FFT blocks, 256 hidden, 2 heads, conv FFN 1024 with kernels [9, 1], dropout 0.2 | 11.55 M |
| Variance adaptor | duration, pitch and energy predictors (2 x conv1d k=3, 256 filters, LayerNorm, dropout 0.5) + 256-bin pitch/energy embeddings | 1.32 M |
| Length regulator | repeats each phoneme hidden state by its (target or predicted) duration | - |
| Decoder | 6 FFT blocks, same sizes as the encoder | 17.32 M |
| Mel projection + PostNet | linear 256->80, 5-layer conv postnet (512 ch, k=5) | 4.37 M |
| Speaker embedding | added to the encoder output when the corpus has more than one speaker | 0.03 M (EN) |
| **Total** | | **34.6 M** |

Taken from the reference design:
- Phoneme-level pitch and energy. Frame values are averaged over each phoneme's MFA duration.
- Pitch and energy are z-normalised and quantised into 256 linear bins for the embeddings.
- The duration predictor regresses log(d + 1).
- Loss: L1 on the mel before and after the postnet, plus MSE for pitch, energy and log duration.
- Adam (β = 0.9/0.98, ε = 1e-9) with the Noam schedule (4000 warm-up steps) and gradient clipping at 1.0.
- A pretrained universal HiFi-GAN (V1) vocoder, used without fine-tuning.

Code layout: `fastspeech2/` holds everything both languages share. `English/` and `Telugu/`
hold the language-specific preprocessing, configuration and entry points.

## 3. Modifications for each language

### Shared changes
- **Consistent lexicon.** The front end used at inference also writes the MFA pronunciation
  dictionary. The phoneme sequences the model is trained on (from the MFA TextGrids) therefore
  come from the same G2P as the sequences it sees at inference.
- **HiFi-GAN-compatible features.** The mel spectrogram is computed exactly as in HiFi-GAN
  (reflect padding of (n_fft - hop)/2, `center=False`, natural-log magnitude mel, 80 bins,
  0-8 kHz at 22.05 kHz). Audio is cut on frame boundaries at the first/last non-silent phone,
  so the number of frames equals the sum of the durations exactly.
- **Statistics from training data only.** Pitch/energy mean, std, min and max are computed on
  the training split. Pitch/energy control at inference scales values in Hz/magnitude rather
  than in normalised units.
- **Multi-speaker support.** An optional speaker embedding, switched on automatically
  (Svarah has many speakers, IndicTTS Telugu has two).
- **Training robustness.** Mixed precision, CSV loss logs, loss plots refreshed at every
  validation step, and a wall-clock budget (`--max_hours`) so that a time-limited GPU session
  always ends with a validated `best.pt`.

### English (Svarah)
- **Text normalisation.** Times (`10:45 AM` → "ten forty five A M"), day-month dates
  (`12 November` → "twelfth November"), years (`2026` → "twenty twenty six"), ordinals,
  decimals and common abbreviations.
- **G2P.** g2p_en's lookup order (homograph table by POS tag → CMUdict → neural G2P), but run
  per token so that each word's pronunciation can also be written to the MFA lexicon. All-caps
  tokens of up to three letters (AM, PM) are spelled out letter by letter.
- **Compatibility fix.** nltk ≥ 3.9 renamed the POS tagger resource; the front end downloads
  both names before g2p_en is imported.
- **Alignment.** MFA `english_us_arpa` pretrained acoustic model with the custom lexicon.
- **Speakers.** Svarah has no speaker-id column, so a speaker is identified by the
  combination of its metadata fields (gender, age group, first language, native state and
  district, education, job). These fields are constant per speaker.

### Telugu (IndicTTS)
- **Rule-based G2P** for the Telugu script (`fastspeech2/text/telugu.py`):
  - every consonant carries an inherent /a/ unless a vowel sign (matra) or virama follows;
  - independent vowels and matras map to 12 vowel phones;
  - anusvara (ం) becomes the nasal homorganic with the next consonant (ng, nj, N, n, m), and /m/ at the end of a word;
  - visarga becomes /h/.
  This gives a 46-phone inventory. Retroflexes are upper case (T, D, N, S, L).
- **Number normalisation into Telugu words**: cardinals up to crores, ordinals with the -వ suffix
  (`12వ` → పన్నెండవ), times (`10:45` → పది నలభై ఐదు) and years (`2026` → రెండు వేల ఇరవై ఆరు).
- **Alignment.** No pretrained Telugu MFA model exists, so an MFA acoustic model was trained
  from scratch (`mfa train`) on the selected corpus with the rule-based lexicon.
- **Data loading** from the HuggingFace parquet release (48 kHz → 22.05 kHz).

## 4. Datasets and preprocessing

| | English | Telugu |
|---|---|---|
| Corpus | Svarah (AI4Bharat), CC BY 4.0 | IndicTTS Telugu (IIT Madras), via `SPRINGLab/IndicTTS_Telugu` |
| Content | 9.6 h, 117 speakers, read + extempore Indian English | 8.74 h, 2 speakers (female 4.27 h, male 4.47 h), studio read speech |
| Original sample rate | 16 kHz | 48 kHz |
| Duration filter | 1-20 s | 1-20 s |
| Train / val | **[TODO: hours and utterances from `data_info/selection.tsv`]** | **[TODO]** |

Selection and subsampling, the same for both languages (`fastspeech2/preprocessing.py`):
1. Drop utterances outside the duration range or with empty text after normalisation.
2. Group the remaining utterances by their normalised sentence.
3. Shuffle the groups with a fixed seed (1234). Assign whole groups to validation until it
   reaches 1 h, then to training until it reaches 5 h.

Because whole sentence groups are assigned, no utterance and no sentence text occurs in both
splits. The chosen ids are listed in `<Language>/data_info/selection.tsv`.

For Telugu, the female speaker alone (4.27 h) cannot provide 6 h, so both speakers are pooled.
A speaker embedding separates them, and inference uses the female voice.

Preprocessing steps:
1. Decode the parquet audio, convert to mono, resample to 22.05 kHz and peak-normalise.
2. Normalise the text and run G2P. Write the MFA corpus (wav + .lab per utterance, one folder
   per speaker) and the lexicon.
3. MFA alignment (English: `mfa align` with `english_us_arpa`; Telugu: `mfa train`). Empty and
   `sil` intervals are mapped to the pause symbol `sp`. Leading and trailing silences are removed.
4. Extract per utterance:
   - 80-bin log-mel spectrogram (hop 256 = 11.6 ms);
   - frame energy (L2 norm of the STFT magnitude);
   - F0 with WORLD (DIO + StoneMask), with unvoiced frames linearly interpolated;
   - phoneme-level averages of energy and F0.
5. Z-normalise pitch and energy with the training-split statistics.

## 5. Training configuration

| | English | Telugu |
|---|---|---|
| Batch size | 16 | 16 |
| Steps | **[TODO]** | **[TODO]** |
| Learning rate | Noam, d_model = 256, 4000 warm-up | same |
| Precision | fp16 autocast | same |
| GPU / time | **[TODO]** | **[TODO]** |
| Validation | every 1000 steps on the full validation split; `best.pt` = lowest total validation loss | same |

## 6. Loss curves **[TODO after training]**

Figures: `Results/<Language>/training_loss.png`, `validation_loss.png`.
Things to discuss:
- convergence speed of the mel and postnet terms;
- the gap between training and validation total loss, and whether the validation loss rises (overfitting);
- noise or instability in the pitch and energy terms;
- English vs Telugu: noisy multi-speaker ASR data vs clean studio data.

## 7. Generated speech **[TODO after listening]**

Audio: `Results/<Language>/test_generated_audios/`. Phoneme sequences produced by the front ends
(from `variance_values.json`):

| ID | Front-end output (normalised text / notes) |
|---|---|
| E1 | *antidisestablishmentarianism* is in CMUdict: `AE2 N T AY0 D IH0 S AH0 S T AE2 B L IH0 SH M AH0 N T EH1 R IY0 AH0 N IH0 Z AH0 M` |
| E2a | noun *conduct* → `K AA1 N D AH0 K T` (stress on the first syllable) |
| E2b | verb *conduct* → `K AA0 N D AH1 K T` (stress on the second syllable) |
| E3 | the split word is read as two words: *rhyth* → `R AY1 DH` (neural G2P guess) + *m* → `EH1 M` |
| E4 | "ten forty five A M on twelfth November twenty twenty six"; the code-mixed *andaroo* → `AE0 N D AA1 R OW0` |
| T1 | న్యూరోఫార్మకోలాజికల్ → `n y uu r oo ph aa r m a k oo l aa j i k a l` |
| T2a/T2b | పట్టు is the same written word in both sentences (grip vs silk) and gets the same phonemes `p a T T u`; any difference can come only from context in the encoder |
| T3 | అడ / విలో are two separate words → `a D a` + `v i l oo` |
| T4 | 2026 → రెండు వేల ఇరవై ఆరు, 12వ → పన్నెండవ, 10:45 → పది నలభై ఐదు |

For each sentence, comment on intelligibility, pronunciation errors, prosody at the comma and
the sentence end, and artefacts (buzz, metallic sound, skipped or repeated phonemes).

## 8. Variance adaptor analysis **[TODO after training]**

Figures: `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`.
Raw values: `variance_values.json`.

Points to check:
- durations of the pause (`sp`) after commas, longer vowels in stressed syllables, and
  phrase-final lengthening;
- the F0 declination across the sentence, F0 peaks on stressed syllables, and the pitch
  reset after the pause in E4/T4;
- E2a vs E2b: whether the stressed vowel of *conduct* gets the higher F0 and energy and the
  longer duration (AA1 in the noun, AH1 in the verb);
- E3 and T3: whether a pause or a boundary-like F0 or energy dip appears inside the split word;
- energy dips on stops and fricatives, and peaks on open vowels.

## 9. Comparison, limitations and improvements **[TODO, partly after training]**

Expected limitations, to be confirmed against the results:
- **English data quality.** Svarah is an ASR benchmark: many speakers with about 3-5 min
  each, varied microphones, and some spontaneous speech. Expect a less stable voice and more
  alignment noise than in the Telugu studio data.
- **English G2P.** The G2P is US English (CMUdict + g2p_en), while the voices are Indian
  English, so the lexicon and the actual pronunciation do not always match.
  Out-of-vocabulary words (*rhyth*, *andaroo*) depend on the neural G2P guess.
- **Telugu G2P.** Rule-based: English loanwords written in Telugu script (T1) are
  pronounced as spelled. Same-spelling homographs such as పట్టు cannot be separated at the
  phoneme level.
- **Telugu alignment** is learned from only about 6 h, so it is less precise than English
  alignment with a pretrained acoustic model.
- **Vocoder mismatch.** The universal HiFi-GAN was not fine-tuned on the predicted mels.

Possible improvements:
- train for longer;
- fine-tune HiFi-GAN on ground-truth-aligned mels;
- use an Indian-English lexicon or G2P;
- for Telugu, detect English loanwords and handle schwa/gemination more finely;
- use length-sorted batching;
- use more data per speaker, or a single-speaker English subset.

## 10. Reproducibility

- Code: this repository. Dependencies: `requirements.txt` (PyTorch, librosa, g2p_en, tgt,
  pyworld, pyarrow, ...). Montreal Forced Aligner 3.x is installed through conda-forge.
- Configs: `English/config.yaml`, `Telugu/config.yaml`.
- One-click run on Kaggle/Colab: `notebooks/english_gpu.ipynb`, `notebooks/telugu_gpu.ipynb`.
- Manual commands: see `README.md` (extract → select → MFA → features → train → inference).
- Seeds: data selection 1234, training 1234.
- Checkpoints: `English/checkpoints/best.pt`, `Telugu/checkpoints/best.pt` (Git LFS). Each
  contains the config, symbol table and statistics needed for inference.
