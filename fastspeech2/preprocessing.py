"""Shared preprocessing: data selection, MFA corpus/lexicon, feature extraction.

Pipeline for each language:
    1. select   - filter the raw dataset, pick ~5h train / ~1h val with no shared
                  utterances or sentences, write an MFA corpus (22.05 kHz wav + .lab)
                  and a pronunciation lexicon produced by our own front end
    2. (MFA)    - align (English) or train + align (Telugu), outside of Python
    3. features - read TextGrids, extract mel / pitch / energy / durations
"""

import csv
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .audio import AudioProcessor

SELECTION_FILE = "selection.tsv"


# ----------------------------------------------------------------------------
# 1. selection + MFA corpus
# ----------------------------------------------------------------------------

def select_utterances(utts, frontend, ds_cfg):
    """utts: dicts with id, speaker, path, duration, text. Returns (train, val)."""
    lo, hi = ds_cfg["min_duration"], ds_cfg["max_duration"]
    kept, dropped = [], defaultdict(int)
    for u in tqdm(utts, desc="g2p"):
        if not lo <= u["duration"] <= hi:
            dropped["duration"] += 1
            continue
        words = frontend.words(u["text"])
        if not words:
            dropped["empty text"] += 1
            continue
        u["words"] = words
        kept.append(u)
    print(f"kept {len(kept)} / {len(utts)} utterances, dropped: {dict(dropped)}")

    # Group by normalised sentence so the same sentence (read by different
    # speakers) never ends up in both train and val.
    groups = defaultdict(list)
    for u in kept:
        groups[" ".join(w for w, _ in u["words"])].append(u)
    keys = sorted(groups)
    random.Random(ds_cfg["seed"]).shuffle(keys)

    train, val = [], []
    val_sec, train_sec = 0.0, 0.0
    val_target, train_target = ds_cfg["val_hours"] * 3600, ds_cfg["train_hours"] * 3600
    for k in keys:
        dur = sum(u["duration"] for u in groups[k])
        if val_sec < val_target:
            val.extend(groups[k])
            val_sec += dur
        elif train_sec < train_target:
            train.extend(groups[k])
            train_sec += dur
    print(f"train: {len(train)} utts, {train_sec / 3600:.2f} h | val: {len(val)} utts, {val_sec / 3600:.2f} h")
    if train_sec < train_target * 0.95:
        print("warning: not enough data left for the requested amount of training audio")
    return train, val


def write_selection(path, train, val):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "speaker", "split", "duration", "text"])
        for split, items in (("train", train), ("val", val)):
            for u in items:
                w.writerow([u["id"], u["speaker"], split, f"{u['duration']:.3f}", u["text"]])


def read_selection(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_mfa_corpus(utts, audio_cfg, corpus_dir, lexicon_path):
    """Resample to the model rate and write <corpus>/<speaker>/<id>.wav + .lab, plus the lexicon."""
    ap = AudioProcessor(audio_cfg)
    lexicon = set()
    for u in tqdm(utts, desc="mfa corpus"):
        spk_dir = Path(corpus_dir) / u["speaker"]
        spk_dir.mkdir(parents=True, exist_ok=True)
        wav = ap.load(u["path"])
        sf.write(spk_dir / f"{u['id']}.wav", wav, ap.sr, subtype="PCM_16")
        (spk_dir / f"{u['id']}.lab").write_text(" ".join(w for w, _ in u["words"]), encoding="utf-8")
        for word, phones in u["words"]:
            lexicon.add((word, " ".join(phones)))

    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word, phones in sorted(lexicon):
            f.write(f"{word}\t{phones}\n")
    print(f"lexicon: {len(lexicon)} entries -> {lexicon_path}")


def run_selection(cfg, frontend, utts):
    p = cfg["path"]
    os.makedirs(p["preprocessed_dir"], exist_ok=True)
    print(f"{len(utts)} utterances, {len({u['speaker'] for u in utts})} speakers in the raw data")
    train, val = select_utterances(utts, frontend, cfg["dataset"])
    write_selection(os.path.join(p["preprocessed_dir"], SELECTION_FILE), train, val)
    write_mfa_corpus(train + val, cfg["audio"], p["corpus_dir"], p["lexicon"])


# ----------------------------------------------------------------------------
# 3. features from alignments
# ----------------------------------------------------------------------------

def parse_textgrid(path, sr, hop):
    """Phones + durations (frames) from an MFA TextGrid, leading/trailing silence removed."""
    import tgt

    tier = tgt.io.read_textgrid(str(path), include_empty_intervals=True).get_tier_by_name("phones")
    phones, durations = [], []
    start = end = None
    last = 0
    for iv in tier.intervals:
        p = iv.text.strip()
        if p in ("", "sil", "sp"):
            p = "sp"
        if not phones and p == "sp":
            continue
        if start is None:
            start = iv.start_time
        phones.append(p)
        durations.append(int(round(iv.end_time * sr / hop) - round(iv.start_time * sr / hop)))
        if p != "sp":
            end = iv.end_time
            last = len(phones)
    return phones[:last], durations[:last], start, end


def phoneme_average(values, durations):
    out = np.zeros(len(durations), dtype=np.float32)
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            out[i] = values[pos: pos + d].mean()
        else:
            out[i] = values[min(pos, len(values) - 1)]
        pos += d
    return out


_ap = None


def _init_worker(audio_cfg):
    global _ap
    _ap = AudioProcessor(audio_cfg)


def _process(job):
    utt_id, wav_path, tg_path, out_dir = job
    phones, durations, start, end = parse_textgrid(tg_path, _ap.sr, _ap.hop)
    if not phones:
        return utt_id, None, "empty alignment"

    wav = _ap.load(wav_path)
    # cut on frame boundaries so the mel has exactly sum(durations) frames
    s = int(round(start * _ap.sr / _ap.hop)) * _ap.hop
    e = int(round(end * _ap.sr / _ap.hop)) * _ap.hop
    wav = wav[s:e]
    if len(wav) < e - s:
        wav = np.pad(wav, (0, e - s - len(wav)))

    mel, energy = _ap.mel_and_energy(wav)
    n = sum(durations)
    mel, energy = mel[:, :n], energy[:n]
    if mel.shape[1] != n:
        return utt_id, None, f"frame mismatch {mel.shape[1]} vs {n}"

    f0 = _ap.f0(wav, n)
    voiced = np.nonzero(f0)[0]
    if len(voiced) < 2:
        return utt_id, None, "no voiced frames"
    f0 = np.interp(np.arange(n), voiced, f0[voiced])  # fill unvoiced gaps before averaging

    np.save(os.path.join(out_dir, "mel", f"{utt_id}.npy"), mel.T.astype(np.float32))
    np.save(os.path.join(out_dir, "duration", f"{utt_id}.npy"), np.array(durations, dtype=np.int64))
    return utt_id, (phones, phoneme_average(f0, durations), phoneme_average(energy, durations)), None


def run_feature_extraction(cfg, frontend, workers=4):
    p = cfg["path"]
    out_dir = p["preprocessed_dir"]
    for sub in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    selection = read_selection(os.path.join(out_dir, SELECTION_FILE))
    textgrids = {tg.stem: tg for tg in Path(p["textgrid_dir"]).rglob("*.TextGrid")}

    jobs, meta = [], {}
    for row in selection:
        tg = textgrids.get(row["id"])
        if tg is None:
            continue
        wav_path = Path(p["corpus_dir"]) / row["speaker"] / f"{row['id']}.wav"
        jobs.append((row["id"], str(wav_path), str(tg), out_dir))
        meta[row["id"]] = row
    print(f"{len(jobs)} / {len(selection)} selected utterances have alignments")

    results, errors = {}, defaultdict(int)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg["audio"],)) as ex:
        for utt_id, res, err in tqdm(ex.map(_process, jobs, chunksize=8), total=len(jobs), desc="features"):
            if err:
                errors[err.split()[0]] += 1
                continue
            phones = res[0]
            bad = [ph for ph in phones if ph not in frontend.symbol_to_id]
            if bad:
                errors["unknown phones"] += 1
                continue
            results[utt_id] = res
    if errors:
        print("skipped:", dict(errors))

    # pitch / energy statistics over the training split only
    train_ids = [i for i in results if meta[i]["split"] == "train"]
    stats = {}
    for k, name in ((1, "pitch"), (2, "energy")):
        values = np.concatenate([results[i][k] for i in train_ids])
        mean, std = float(values.mean()), float(values.std())
        lo, hi = np.inf, -np.inf
        for utt_id, res in results.items():
            norm = (res[k] - mean) / std
            np.save(os.path.join(out_dir, name, f"{utt_id}.npy"), norm.astype(np.float32))
            lo, hi = min(lo, float(norm.min())), max(hi, float(norm.max()))
        stats[name] = {"min": lo, "max": hi, "mean": mean, "std": std}

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    speakers = sorted({meta[i]["speaker"] for i in results})
    with open(os.path.join(out_dir, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({s: i for i, s in enumerate(speakers)}, f, indent=2, ensure_ascii=False)

    hop_sec = cfg["audio"]["hop_length"] / cfg["audio"]["sampling_rate"]
    for split in ("train", "val"):
        ids = [i for i in results if meta[i]["split"] == split]
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            for i in ids:
                f.write(f"{i}|{meta[i]['speaker']}|{' '.join(results[i][0])}|{meta[i]['text']}\n")
        hours = sum(np.load(os.path.join(out_dir, "duration", f"{i}.npy")).sum() for i in ids) * hop_sec / 3600
        print(f"{split}: {len(ids)} utterances, {hours:.2f} h after trimming silences")
