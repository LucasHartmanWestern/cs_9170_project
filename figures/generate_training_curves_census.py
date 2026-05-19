"""
Training curves: episode return for two panels (Census | Capture-24).

Uses confirmed best configurations:
  Census:    k=5, pca=10, ep=30, traj=2000 (EXP-021 best, 9af13c63)
  Capture-24: k=5, pca=15, ep=10, traj=1000 (EXP-025 best, 779cf9c5)

Output:
  paper/figures/fig_training_curves_nophase2.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CENSUS_RUN = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/census/k5/"
    "SPECcensus_k5_gpu1_EP5000_PCA10_REWwgl_minID0_majID1_"
    "TRJ2000_REAL3000_GG202604251803_9af13c63"
)
CAPTURE24_RUN = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/capture_24/k5/"
    "SPECcapture24_k5_gpu0_EP5000_PCA15_REWwgl_minID1_majID0_"
    "TRJ1000_REAL4000_GG202605041119_779cf9c5"
)
FIG_DIR = Path("/home/epigou/cs_9170_project/paper/figures")

SEEDS = ["0", "1", "42"]
METRIC_COL = "meta.episode_return"
SMOOTH_WINDOW = 40

PANELS = [
    (CENSUS_RUN,    "Census Income",  "#1a4f8a", "#6699cc"),
    (CAPTURE24_RUN, "Capture-24",     "#27ae60", "#7ecfa0"),
]


def smooth(s, w=SMOOTH_WINDOW):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def load_seed(run_dir, seed):
    p = run_dir / f"seed_{seed}" / "metrics.csv"
    df = pd.read_csv(p, usecols=["episode", "meta.phase", METRIC_COL])
    ph1 = df[df["meta.phase"] == "phase1_class1"].copy()
    ph1 = ph1[~ph1["episode"].duplicated(keep="last")]
    return ph1.set_index("episode")[METRIC_COL]


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.75), sharey=False)

for ax, (run_dir, label, color_main, color_seed) in zip(axes, PANELS):
    panel_label = "a)" if label == "Census Income" else "b)"
    ax.text(-0.20, 0.97, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)

    seed_curves = []
    for s in SEEDS:
        try:
            series = load_seed(run_dir, s)
            smoothed = smooth(series)
            seed_curves.append(smoothed)
            ax.plot(smoothed.index, smoothed.values,
                    color=color_seed, linewidth=0.5, alpha=0.35, zorder=2)
        except Exception as e:
            print(f"  [warn] {label} seed {s}: {e}")

    if not seed_curves:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        continue

    min_len = min(len(c) for c in seed_curves)
    arr = np.array([c.values[:min_len] for c in seed_curves])
    eps = seed_curves[0].index[:min_len]

    ax.plot(eps, arr.mean(0), color=color_main, linewidth=2.2, zorder=5,
            label=f"Mean {label}")
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

plt.tight_layout()

out = FIG_DIR / "fig_training_curves_nophase2.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
print("Done.")
