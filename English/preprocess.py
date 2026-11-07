"""Preprocessing for the English model (Svarah).

    python English/preprocess.py select      # pick train/val, write MFA corpus + lexicon
    (run MFA, see README)
    python English/preprocess.py features    # mel / pitch / energy / durations
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.preprocessing import run_feature_extraction, run_selection  # noqa: E402
from fastspeech2.text import get_frontend  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent


def clean_transcript(text):
    # Svarah transcripts are ASR style; drop any bracketed tags such as [noise] or <unk>
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_svarah(cfg):
    ds = cfg["dataset"]
    raw = Path(cfg["path"]["raw_dir"])
    speaker_re = re.compile(ds["speaker_regex"]) if ds.get("speaker_regex") else None

    utts = []
    with open(raw / ds["manifest"], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            audio = Path(item["audio_filepath"])
            if not audio.is_absolute():
                audio = raw / audio
            if not audio.exists():  # manifest paths may point to the authors' machine
                audio = raw / "audio" / audio.name

            speaker = item.get("speaker_id")
            if speaker is None and speaker_re is not None:
                m = speaker_re.match(audio.stem)
                speaker = m.group(1) if m else None
            utts.append({
                "id": re.sub(r"[^\w\-]", "_", audio.stem),
                "speaker": str(speaker or "svarah"),
                "path": str(audio),
                "duration": float(item["duration"]),
                "text": clean_transcript(item["text"]),
            })
    return utts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "features"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    frontend = get_frontend(cfg["language"])

    if args.stage == "select":
        run_selection(cfg, frontend, load_svarah(cfg))
        p = cfg["path"]
        print("\nnext, align with MFA:")
        print("  mfa model download acoustic english_us_arpa")
        print(f"  mfa align {p['corpus_dir']} {p['lexicon']} english_us_arpa {p['textgrid_dir']} --clean -j {args.workers}")
    else:
        run_feature_extraction(cfg, frontend, args.workers)


if __name__ == "__main__":
    main()
