import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMPONENTS = ["mel", "postnet", "pitch", "energy", "duration"]


def read_log(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _loss_figure(log, title, out_path, compare=None, compare_label=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(log["step"], log["total"], label="total")
    if compare:
        ax1.plot(compare["step"], compare["total"], "--", alpha=0.7, label=compare_label)
    ax1.set_title(f"{title} - total loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for name in COMPONENTS:
        ax2.plot(log["step"], log[name], label=name)
    ax2.set_title(f"{title} - components")
    ax2.set_xlabel("step")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_losses(log_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    train = read_log(os.path.join(log_dir, "train_log.csv"))
    val = read_log(os.path.join(log_dir, "val_log.csv"))
    if train:
        _loss_figure(train, "Training", os.path.join(results_dir, "training_loss.png"))
    if val:
        # training curve drawn behind the validation one makes overfitting easy to spot
        _loss_figure(val, "Validation", os.path.join(results_dir, "validation_loss.png"),
                     compare=train or None, compare_label="train total")


def _phone_axis(ax, phones):
    ax.set_xticks(range(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(phones) - 0.4)
    ax.grid(alpha=0.3, axis="y")


def plot_variance(result, sent_id, out_dir, hop_ms):
    """One figure each for predicted duration, F0 and energy (phoneme level)."""
    phones = result["phones"]
    width = max(8, 0.22 * len(phones))
    x = np.arange(len(phones))
    sil = np.array([p in ("sp", "spn") for p in phones])

    figs = {
        "duration": (result["durations"], "duration (frames)", f"{sent_id}: predicted duration"
                     f"  (1 frame = {hop_ms:.1f} ms, total {result['durations'].sum() * hop_ms / 1000:.2f} s)"),
        "pitch": (result["pitch"], "F0 (Hz)", f"{sent_id}: predicted pitch (F0)"),
        "energy": (result["energy"], "energy (L2 norm of STFT frame)", f"{sent_id}: predicted energy"),
    }
    for kind, (values, ylabel, title) in figs.items():
        os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
        fig, ax = plt.subplots(figsize=(width, 3.8))
        if kind == "duration":
            ax.bar(x, values, color=np.where(sil, "lightgray", "tab:blue"))
        else:
            ax.plot(x, values, marker="o", markersize=3)
            ax.scatter(x[sil], values[sil], color="lightgray", zorder=3, label="pause")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _phone_axis(ax, phones)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, kind, f"{sent_id}.png"), dpi=150)
        plt.close(fig)


def plot_mel(mel, path, title):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.imshow(mel, aspect="auto", origin="lower", interpolation="none")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
