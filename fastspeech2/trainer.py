import csv
import json
import math
import os
import time
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from .dataset import TTSDataset, collate, to_device
from .model import FastSpeech2, FastSpeech2Loss
from .plotting import plot_losses
from .text import get_symbols

LOG_FIELDS = ["step", "total", "mel", "postnet", "pitch", "energy", "duration"]


class CsvLog:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in self.fields])


def noam(d_model, warmup):
    return lambda step: d_model ** -0.5 * min((step + 1) ** -0.5, (step + 1) * warmup ** -1.5)


def forward_batch(model, batch):
    return model(
        batch["texts"], batch["src_lens"], batch["speakers"],
        mel_lens=batch["mel_lens"], max_mel_len=batch["mels"].size(1),
        pitches=batch["pitches"], energies=batch["energies"], durations=batch["durations"],
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    sums, n = defaultdict(float), 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, enabled=use_amp):
            out = forward_batch(model, batch)
            _, parts = criterion(out, batch)
        bs = batch["texts"].size(0)
        for k, v in parts.items():
            sums[k] += v * bs
        n += bs
    model.train()
    return {k: v / n for k, v in sums.items()}


def train(cfg, resume=None, device=None):
    p, t = cfg["path"], cfg["train"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(t.get("amp")) and device.type == "cuda"
    torch.manual_seed(t["seed"])
    os.makedirs(p["checkpoint_dir"], exist_ok=True)

    pre = p["preprocessed_dir"]
    with open(os.path.join(pre, "stats.json")) as f:
        stats = json.load(f)
    with open(os.path.join(pre, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)
    symbols = get_symbols(cfg["language"])
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    train_set = TTSDataset(os.path.join(pre, "train.txt"), pre, symbol_to_id)
    val_set = TTSDataset(os.path.join(pre, "val.txt"), pre, symbol_to_id)
    print(f"device {device} | train {len(train_set)} | val {len(val_set)} | speakers {len(speakers)}")

    workers = t["num_workers"]
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True, collate_fn=collate,
                              num_workers=workers, drop_last=True, pin_memory=device.type == "cuda",
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=t["batch_size"], shuffle=False, collate_fn=collate,
                            num_workers=workers)

    model = FastSpeech2(cfg, stats, len(symbols), len(speakers)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    criterion = FastSpeech2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr_scale"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam(cfg["model"]["encoder_hidden"], t["warmup_steps"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_val = 0, math.inf
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resumed from {resume} at step {step}")

    # the speaker with the most training utterances is the default voice at inference
    spk_counts = Counter(item[1] for item in train_set.items)
    meta = {
        "config": cfg, "stats": stats, "speakers": speakers, "symbols": symbols,
        "default_speaker": spk_counts.most_common(1)[0][0],
    }

    def save(path, val_loss=None, full=False):
        state = {"model": model.state_dict(), "step": step, "val_loss": val_loss, **meta}
        if full:
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), best_val=best_val)
        torch.save(state, path)

    train_log = CsvLog(os.path.join(p["checkpoint_dir"], "train_log.csv"), LOG_FIELDS + ["lr"])
    val_log = CsvLog(os.path.join(p["checkpoint_dir"], "val_log.csv"), LOG_FIELDS)

    acc = t["grad_acc_steps"]
    running, n_running, micro = defaultdict(float), 0, 0
    tic = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < t["total_steps"]:
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                out = forward_batch(model, batch)
                loss, parts = criterion(out, batch)
            scaler.scale(loss / acc).backward()
            for k, v in parts.items():
                running[k] += v
            n_running += 1
            micro += 1
            if micro % acc:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % t["log_step"] == 0:
                avg = {k: v / n_running for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]
                train_log.write({"step": step, **avg, "lr": lr})
                print(f"step {step:6d} | loss {avg['total']:.4f} (mel {avg['mel']:.3f} post {avg['postnet']:.3f} "
                      f"p {avg['pitch']:.3f} e {avg['energy']:.3f} d {avg['duration']:.3f}) | lr {lr:.2e} "
                      f"| {(time.time() - tic) / t['log_step']:.2f}s/step")
                running, n_running, tic = defaultdict(float), 0, time.time()

            if step % t["val_step"] == 0:
                val = evaluate(model, val_loader, criterion, device, use_amp)
                val_log.write({"step": step, **val})
                print(f"  val loss {val['total']:.4f} (mel {val['mel']:.3f} post {val['postnet']:.3f})")
                if val["total"] < best_val:
                    best_val = val["total"]
                    save(os.path.join(p["checkpoint_dir"], "best.pt"), val["total"])
                plot_losses(p["checkpoint_dir"], p["results_dir"])

            if step % t["save_step"] == 0:
                save(os.path.join(p["checkpoint_dir"], "last.pt"), full=True)

            if step >= t["total_steps"]:
                break

    save(os.path.join(p["checkpoint_dir"], "last.pt"), full=True)
    plot_losses(p["checkpoint_dir"], p["results_dir"])
    print(f"done, best val loss {best_val:.4f}")
