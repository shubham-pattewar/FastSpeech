"""Turn a HuggingFace audio dataset (parquet files) into wav files + a JSON-lines manifest.

Both Svarah and IndicTTS Telugu are distributed this way. The manifest has one
object per line: {id, speaker, audio_filepath, duration, text}.
"""

import io
import json
import re
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def hf_features(parquet_file):
    try:
        return json.loads(parquet_file.schema_arrow.metadata[b"huggingface"])["info"]["features"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}


def audio_column(parquet_file):
    """Name of the column holding {bytes, path} audio structs."""
    for field in parquet_file.schema_arrow:
        t = field.type
        if hasattr(t, "num_fields") and {t.field(i).name for i in range(t.num_fields)} >= {"bytes"}:
            return field.name
    raise ValueError("no audio column found")


def extract_parquet(parquet_dir, raw_dir, sr, speaker_fn, id_prefix):
    """speaker_fn(row, features) -> speaker name. Writes raw_dir/wavs/*.wav and raw_dir/manifest.jsonl."""
    import pyarrow.parquet as pq

    files = sorted(Path(parquet_dir).rglob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files under {parquet_dir}")
    raw_dir = Path(raw_dir)
    wav_dir = raw_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    n = 0
    with open(raw_dir / "manifest.jsonl", "w", encoding="utf-8") as out:
        for path in files:
            pf = pq.ParquetFile(path)
            feats = hf_features(pf)
            audio_col = audio_column(pf)
            if n == 0:
                print("columns:", pf.schema_arrow.names)
            for batch in tqdm(pf.iter_batches(batch_size=32), desc=path.name,
                              total=-(-pf.metadata.num_rows // 32)):
                for row in batch.to_pylist():
                    audio = row.pop(audio_col)
                    wav, orig_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    if orig_sr != sr:
                        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)

                    row["_audio_path"] = audio.get("path") or ""
                    speaker = re.sub(r"[^\w\-]", "_", str(speaker_fn(row, feats)))
                    counts[speaker] = counts.get(speaker, 0) + 1
                    n += 1
                    utt_id = f"{id_prefix}_{n:05d}"
                    wav_path = wav_dir / f"{utt_id}.wav"
                    sf.write(wav_path, wav, sr, subtype="PCM_16")
                    out.write(json.dumps({
                        "id": utt_id,
                        "speaker": speaker,
                        "audio_filepath": str(wav_path),
                        "duration": len(wav) / sr,
                        "text": str(row.get("text", "")).strip(),
                        "source_path": row["_audio_path"],
                    }, ensure_ascii=False) + "\n")

    print(f"{n} utterances, {len(counts)} speakers")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    print("largest speakers:", top)
    return counts


def load_manifest(raw_dir, speakers=None):
    allowed = set(speakers or [])
    utts = []
    with open(Path(raw_dir) / "manifest.jsonl", encoding="utf-8") as f:
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
