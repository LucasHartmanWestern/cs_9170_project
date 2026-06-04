"""
Training curves: episode return for two panels (Census | Capture-24).

Uses final 5-seed/5-fold runs reported in the comparative table (EXP-050):
  Census:    k=10, pca=10, ep=30, traj=2000 (experiment3/census_forge, 5 seeds)
  Capture-24: k=5, pca=15, ep=10, traj=1000, rand_0010 (experiment3/capture24_forge, 5 folds)

Output:
  paper/figures/fig_training_curves_nophase2.png
"""

import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

STORAGE = Path("/storage_1/epigou_storage/FORGE")

CENSUS_RUN = STORAGE / "experiment3/census_forge"
CENSUS_SEEDS = ["0", "1", "2", "3", "42"]

# Capture-24: 5 fold directories, each with seed_42
CAPTURE24_FOLD_DIRS = sorted(
    Path(d) for d in glob.glob(str(STORAGE / "experiment3/capture24_forge/SPECrand_0010_*"))
)
CAPTURE24_SEED = "42"

FIG_DIR = Path("/home/epigou/cs_9170_project/paper/figures")

METRIC_COL = "meta.episode_return"
SMOOTH_WINDOW = 40


def smooth(s, w=SMOOTH_WINDOW):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def load_seed(run_dir, seed):
    p = run_dir / f"seed_{seed}" / "metrics.csv"
    df = pd.read_csv(p, usecols=["episode", "meta.phase", METRIC_COL])
    ph1 = df[df["meta.phase"] == "phase1_class1"].copy()
    ph1 = ph1[~ph1["episode"].duplicated(keep="last")]
    return ph1.set_index("episode")[METRIC_COL]


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.75), sharey=False)

# ── Census panel ──────────────────────────────────────────────────────────────
ax = axes[0]
label, color_main, color_seed = "Census Income", "#1a4f8a", "#6699cc"
ax.text(-0.20, 0.97, "a)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)

seed_curves = []
for s in CENSUS_SEEDS:
    try:
        series = load_seed(CENSUS_RUN, s)
        smoothed = smooth(series)
        seed_curves.append(smoothed)
        ax.plot(smoothed.index, smoothed.values,
                color=color_seed, linewidth=0.5, alpha=0.35, zorder=2)
    except Exception as e:
        print(f"  [warn] Census seed {s}: {e}")

def finish_panel(ax, seed_curves, color_main, label):
    if not seed_curves:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return
    min_len = min(len(c) for c in seed_curves)
    arr = np.array([c.values[:min_len] for c in seed_curves])
    eps = seed_curves[0].index[:min_len]
    ax.plot(eps, arr.mean(0), color=color_main, linewidth=2.2, zorder=5,
            label=f"Mean ({len(seed_curves)} {'seeds' if len(seed_curves) > 1 else 'seed'})")
    ax.fill_between(eps, arr.min(0), arr.max(0),
                    color=color_main, alpha=0.20, zorder=4, label="Min-max range")
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Episode return", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=12, loc="lower right")


finish_panel(axes[0], seed_curves, color_main, label)

# ── Capture-24 panel (5 fold dirs, each contributing seed_42) ─────────────────
ax = axes[1]
color_main, color_seed = "#27ae60", "#7ecfa0"
ax.text(-0.20, 0.97, "b)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)

fold_curves = []
for fold_dir in CAPTURE24_FOLD_DIRS:
    try:
        series = load_seed(fold_dir, CAPTURE24_SEED)
        smoothed = smooth(series)
        fold_curves.append(smoothed)
        ax.plot(smoothed.index, smoothed.values,
                color=color_seed, linewidth=0.5, alpha=0.35, zorder=2)
    except Exception as e:
        print(f"  [warn] Capture-24 {fold_dir.name}: {e}")

finish_panel(axes[1], fold_curves, color_main, "Capture-24")

plt.tight_layout()

out = FIG_DIR / "fig_training_curves_nophase2.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
print("Done.")
