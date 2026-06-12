"""
Poster-ready ROC figure overlaying all three datasets (LFW, AgeDB-30, CFP-FP).
Transparent background, large fonts, AUC in legend. Reads the pair_scores.csv
files already produced by tests 02 / 07 / 08.

Output: poster_roc.png (transparent) + poster_roc_white.png (white bg fallback)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
DATASETS = [
    ("LFW",      BASE / "02_far_frr" / "pair_scores.csv",  "#1f77b4"),
    ("CFP-FP",   BASE / "08_cfp_fp"  / "pair_scores.csv",  "#2ca02c"),
    ("AgeDB-30", BASE / "07_agedb30" / "pair_scores.csv",  "#ff7f0e"),
]


def load(path):
    labels, dists = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lab = int(r["label"])
                d = float(r["distance"])
            except (ValueError, KeyError, TypeError):
                continue
            if "status" in r and r["status"] not in ("", "ok", "OK", "True"):
                # 07/08 mark invalid rows; skip anything not a clean score
                if r.get("error"):
                    continue
            labels.append(lab)
            dists.append(d)
    return np.array(labels), np.array(dists)


def roc(labels, dists):
    # score = similarity = -distance (higher => same person)
    scores = -dists
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0:
        return None, None, 0.0
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / P
    fpr = fp / N
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc


def make(transparent: bool, out: Path):
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    for name, path, color in DATASETS:
        if not path.exists():
            continue
        labels, dists = load(path)
        fpr, tpr, auc = roc(labels, dists)
        if fpr is None:
            continue
        ax.plot(fpr, tpr, color=color, lw=3.2, label=f"{name}  (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.5)

    ax.set_xlabel("False Accept Rate", fontsize=18, fontweight="bold")
    ax.set_ylabel("True Accept Rate", fontsize=18, fontweight="bold")
    ax.set_title("Face Verification ROC — Cross-Dataset", fontsize=20, fontweight="bold", pad=14)
    ax.tick_params(labelsize=14)
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.3, lw=0.8)
    ax.legend(fontsize=15, loc="lower right", frameon=True, framealpha=0.9)
    for s in ax.spines.values():
        s.set_linewidth(1.4)
    plt.tight_layout()
    plt.savefig(out, dpi=200, transparent=transparent,
                facecolor=("none" if transparent else "white"))
    plt.close()
    print(f"wrote {out}  (transparent={transparent})")


if __name__ == "__main__":
    make(True,  BASE / "poster_roc.png")
    make(False, BASE / "poster_roc_white.png")
