"""
Training curves: episode return for two panels (Census | Capture-24).

Uses confirmed final configurations:
  Census:    k=10, pca=10, ep=30, traj=2000 — experiment3/census_forge/ (5 seeds: 0,1,2,3,42)
  Capture-24: k=5, pca=15, ep=10, rand_0010 secondary params — 5 fold dirs treated as seeds

Output:
  paper/figures/fig_training_curves_nophase2.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Census: single run dir with seed_* subdirs
CENSUS_RUN = Path("/storage_1/epigou_storage/FORGE/experiment3/census_forge")

# Capture-24: one dir per fold (all 5 folds via experiment1 symlinks)
_EXP1_C24 = Path("/storage_1/epigou_storage/FORGE/experiment1/capture24_random")
CAPTURE24_FOLD_DIRS = sorted(
    _EXP1_C24.glob(
        "SPECrand_0010_learning_rate0p00024106501268706255_"
        "lr0p00013646506337961615_delta_scale0p10965398215380505_"
        "optimizeradam_optimizeradamw_fold*_EP5000_PCA15_*"
    )
)

FIG_DIR = Path("/home/epigou/cs_9170_project/paper/figures")
METRIC_COL   = "meta.episode_return"
SMOOTH_WINDOW = 40


def smooth(s, w=SMOOTH_WINDOW):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def load_metrics_csv(seed_dir: Path):
    p = seed_dir / "metrics.csv"
    df = pd.read_csv(p, usecols=["episode", "meta.phase", METRIC_COL])
    ph1 = df[df["meta.phase"] == "phase1_class1"].copy()
    ph1 = ph1[~ph1["episode"].duplicated(keep="last")]
    return ph1.set_index("episode")[METRIC_COL]


def collect_curves(seed_dirs: list[Path], label: str, color_seed: str, ax):
    """Load and plot per-seed curves; return list of smoothed series."""
    curves = []
    for sd in seed_dirs:
        try:
            series  = load_metrics_csv(sd)
            smoothed = smooth(series)
            curves.append(smoothed)
            ax.plot(smoothed.index, smoothed.values,
                    color=color_seed, linewidth=0.5, alpha=0.35, zorder=2)
        except Exception as e:
            print(f"  [warn] {label} {sd.name}: {e}")
    return curves


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.75), sharey=False)

PANELS = [
    ("Census Income",  "#1a4f8a", "#6699cc"),
    ("Capture-24",     "#27ae60", "#7ecfa0"),
]

for ax, (label, color_main, color_seed) in zip(axes, PANELS):
    panel_label = "a)" if label == "Census Income" else "b)"
    ax.text(-0.20, 0.97, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)

    if label == "Census Income":
        seed_dirs = sorted(
            d for d in CENSUS_RUN.iterdir()
            if d.is_dir() and d.name.startswith("seed_")
        )
    else:
        # Treat seed_42/ from each fold dir as one seed
        seed_dirs = [d / "seed_42" for d in CAPTURE24_FOLD_DIRS
                     if (d / "seed_42").exists()]

    seed_curves = collect_curves(seed_dirs, label, color_seed, ax)

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
