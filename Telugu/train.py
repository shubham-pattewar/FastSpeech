"""Train the Telugu FastSpeech 2 model.

    python Telugu/train.py
    python Telugu/train.py --resume Telugu/checkpoints/last.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.trainer import train  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--resume", default=None, help="path to last.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help="override train.total_steps")
    parser.add_argument("--max_hours", type=float, default=None, help="stop after this much wall time")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg["train"]["total_steps"] = args.steps
    train(cfg, args.resume, args.device, args.max_hours)
