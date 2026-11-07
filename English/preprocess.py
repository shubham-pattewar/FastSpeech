"""Preprocessing for the English model (Svarah, HuggingFace parquet release).

    python English/preprocess.py extract     # parquet -> 22.05 kHz wavs + manifest
    python English/preprocess.py select      # pick train/val, write MFA corpus + lexicon
    (align with MFA, see README)
    python English/preprocess.py features    # mel / pitch / energy / durations
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.hf_parquet import extract_parquet, load_manifest  # noqa: E402
from fastspeech2.preprocessing import run_feature_extraction, run_selection  # noqa: E402
from fastspeech2.text import get_frontend  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent

SPEAKER_COLUMNS = ("speaker_id", "speaker", "client_id", "spk_id")
# Svarah's per-utterance metadata; used as a speaker stand-in when no id column exists
PROFILE_COLUMNS = ("gender", "age-group", "primary_language", "native_place_state", "native_place_district",
                   "highest_qualification", "job_category", "occupation_domain")


def make_speaker_fn(regex):
    pattern = re.compile(regex) if regex else None

    def speaker(row, feats):
        for col in SPEAKER_COLUMNS:
            if row.get(col) not in (None, ""):
                return row[col]
        if pattern is not None:
            m = pattern.match(Path(row["_audio_path"]).stem)
            if m:
                return m.group(1)
        profile = [str(row[c]) for c in PROFILE_COLUMNS if row.get(c) is not None]
        if profile:
            return "spk_" + hashlib.md5("|".join(profile).encode()).hexdigest()[:8]
        return "svarah"

    return speaker


def clean_transcript(text):
    # Svarah transcripts are ASR style; drop bracketed tags such as [noise] or <unk>
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "select", "features"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    p = cfg["path"]

    if args.stage == "extract":
        extract_parquet(p["parquet_dir"], p["raw_dir"], cfg["audio"]["sampling_rate"],
                        make_speaker_fn(cfg["dataset"].get("speaker_regex")), "svarah")
        return

    frontend = get_frontend(cfg["language"])
    if args.stage == "select":
        utts = load_manifest(p["raw_dir"])
        for u in utts:
            u["text"] = clean_transcript(u["text"])
        run_selection(cfg, frontend, utts)
        print("\nnext, align with MFA:")
        print("  mfa model download acoustic english_us_arpa")
        print(f"  mfa align {p['corpus_dir']} {p['lexicon']} english_us_arpa {p['textgrid_dir']} --clean -j {args.workers}")
    else:
        run_feature_extraction(cfg, frontend, args.workers)


if __name__ == "__main__":
    main()
