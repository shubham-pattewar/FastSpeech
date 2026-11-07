import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def read_filelist(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, speaker, phones, text = line.split("|", 3)
            items.append((utt_id, speaker, phones.split(), text))
    return items


class TTSDataset(Dataset):
    def __init__(self, filelist, preprocessed_dir, symbol_to_id):
        self.dir = preprocessed_dir
        self.items = read_filelist(filelist)
        self.symbol_to_id = symbol_to_id
        with open(os.path.join(preprocessed_dir, "speakers.json"), encoding="utf-8") as f:
            self.speakers = json.load(f)

    def __len__(self):
        return len(self.items)

    def _load(self, kind, utt_id):
        return np.load(os.path.join(self.dir, kind, f"{utt_id}.npy"))

    def __getitem__(self, idx):
        utt_id, speaker, phones, _ = self.items[idx]
        return {
            "id": utt_id,
            "speaker": self.speakers[speaker],
            "text": np.array([self.symbol_to_id[p] for p in phones], dtype=np.int64),
            "mel": self._load("mel", utt_id),
            "pitch": self._load("pitch", utt_id),
            "energy": self._load("energy", utt_id),
            "duration": self._load("duration", utt_id),
        }


def _pad_1d(arrays, dtype):
    out = np.zeros((len(arrays), max(len(a) for a in arrays)), dtype=dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def _pad_2d(arrays):
    out = np.zeros((len(arrays), max(a.shape[0] for a in arrays), arrays[0].shape[1]), dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return torch.from_numpy(out)


def collate(batch):
    return {
        "ids": [b["id"] for b in batch],
        "speakers": torch.tensor([b["speaker"] for b in batch], dtype=torch.long),
        "texts": _pad_1d([b["text"] for b in batch], np.int64),
        "src_lens": torch.tensor([len(b["text"]) for b in batch], dtype=torch.long),
        "mels": _pad_2d([b["mel"] for b in batch]),
        "mel_lens": torch.tensor([b["mel"].shape[0] for b in batch], dtype=torch.long),
        "pitches": _pad_1d([b["pitch"] for b in batch], np.float32),
        "energies": _pad_1d([b["energy"] for b in batch], np.float32),
        "durations": _pad_1d([b["duration"] for b in batch], np.int64),
    }


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
