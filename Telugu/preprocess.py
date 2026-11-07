"""Preprocessing for the Telugu model (IndicTTS Telugu, HuggingFace parquet release).

    python Telugu/preprocess.py extract     # parquet -> 22.05 kHz wavs + manifest
    python Telugu/preprocess.py select      # pick train/val, write MFA corpus + lexicon
    (train MFA on the corpus, see README)
    python Telugu/preprocess.py features
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.hf_parquet import extract_parquet, load_manifest  # noqa: E402
from fastspeech2.preprocessing import run_feature_extraction, run_selection  # noqa: E402
from fastspeech2.text import get_frontend  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent


def gender_speaker(row, feats):
    # two speakers in the corpus, identified only by the gender label
    names = feats.get("gender", {}).get("names") or ["female", "male"]
    return names[row["gender"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "select", "features"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    p = cfg["path"]

    if args.stage == "extract":
        extract_parquet(p["parquet_dir"], p["raw_dir"], cfg["audio"]["sampling_rate"], gender_speaker, "tel")
        return

    frontend = get_frontend(cfg["language"])
    if args.stage == "select":
        run_selection(cfg, frontend, load_manifest(p["raw_dir"], cfg["dataset"].get("speakers")))
        print("\nnext, train an MFA acoustic model on the corpus (there is no pretrained Telugu one):")
        print(f"  mfa train {p['corpus_dir']} {p['lexicon']} {p['mfa_model']} "
              f"--output_directory {p['textgrid_dir']} --clean -j {args.workers}")
    else:
        run_feature_extraction(cfg, frontend, args.workers)


if __name__ == "__main__":
    main()
