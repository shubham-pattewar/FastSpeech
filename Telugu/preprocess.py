"""Preprocessing for the Telugu model (IndicTTS Telugu, HuggingFace parquet release).

    python Telugu/preprocess.py extract     # parquet -> 22.05 kHz wavs + manifest
    python Telugu/preprocess.py select      # pick train/val, write MFA corpus + lexicon
    (train MFA on the corpus, see README)
    python Telugu/preprocess.py features
"""

import argparse
import io
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.preprocessing import run_feature_extraction, run_selection  # noqa: E402
from fastspeech2.text import get_frontend  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent


def gender_names(parquet_file):
    # HF stores the ClassLabel names in the parquet schema metadata
    try:
        meta = json.loads(parquet_file.schema_arrow.metadata[b"huggingface"])
        return meta["info"]["features"]["gender"]["names"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return ["female", "male"]


def extract(cfg):
    import pyarrow.parquet as pq

    sr = cfg["audio"]["sampling_rate"]
    raw = Path(cfg["path"]["raw_dir"])
    wav_dir = raw / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(cfg["path"]["parquet_dir"]).rglob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files under {cfg['path']['parquet_dir']}")

    counts = {}
    with open(raw / "manifest.jsonl", "w", encoding="utf-8") as out:
        for path in files:
            pf = pq.ParquetFile(path)
            names = gender_names(pf)
            for batch in tqdm(pf.iter_batches(batch_size=32, columns=["audio", "text", "gender"]),
                              desc=path.name, total=pf.metadata.num_rows // 32 + 1):
                for row in batch.to_pylist():
                    audio, orig_sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if orig_sr != sr:
                        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

                    speaker = names[row["gender"]]
                    counts[speaker] = counts.get(speaker, 0) + 1
                    utt_id = f"tel_{speaker}_{counts[speaker]:05d}"
                    wav_path = wav_dir / f"{utt_id}.wav"
                    sf.write(wav_path, audio, sr, subtype="PCM_16")
                    out.write(json.dumps({
                        "id": utt_id,
                        "speaker": speaker,
                        "audio_filepath": str(wav_path),
                        "duration": len(audio) / sr,
                        "text": row["text"].strip(),
                    }, ensure_ascii=False) + "\n")
    print("utterances per speaker:", counts)


def load_manifest(cfg):
    allowed = set(cfg["dataset"].get("speakers") or [])
    utts = []
    with open(Path(cfg["path"]["raw_dir"]) / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if allowed and item["speaker"] not in allowed:
                continue
            utts.append({
                "id": item["id"],
                "speaker": item["speaker"],
                "path": item["audio_filepath"],
                "duration": item["duration"],
                "text": item["text"],
            })
    return utts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "select", "features"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage == "extract":
        extract(cfg)
        return

    frontend = get_frontend(cfg["language"])
    if args.stage == "select":
        run_selection(cfg, frontend, load_manifest(cfg))
        p = cfg["path"]
        print("\nnext, train an MFA acoustic model on the corpus (there is no pretrained Telugu one):")
        print(f"  mfa train {p['corpus_dir']} {p['lexicon']} {p['mfa_model']} "
              f"--output_directory {p['textgrid_dir']} --clean -j {args.workers}")
    else:
        run_feature_extraction(cfg, frontend, args.workers)


if __name__ == "__main__":
    main()
