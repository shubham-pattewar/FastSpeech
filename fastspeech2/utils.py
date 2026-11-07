import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    """Load a language config. Relative entries under `path:` are resolved from the repo root,
    so the scripts work no matter which directory they are started from."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["path"].items():
        if value and not os.path.isabs(value):
            cfg["path"][key] = str(ROOT / value)
    return cfg
