"""Train the English FastSpeech 2 model.

    python English/train.py
    python English/train.py --resume English/checkpoints/last.pt
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
    args = parser.parse_args()
    train(load_config(args.config), args.resume, args.device)
