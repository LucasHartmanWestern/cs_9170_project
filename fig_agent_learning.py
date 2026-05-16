"""
Figure: FORGE agent placement of synthetic minority-positive examples.
Compares early vs late training snapshots for:
  - Census Income k=5, ep=30 run (seed 42) — paper best config
  - Capture-24 k=5 run (seed 42) — illustrative; best config (k=3) not yet synced

Run from project root:  python fig_agent_learning.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
# Census: best paper config (k=5, ep=30, pca=10, da_pct=0.01433); snapshots every 150 ep
CENSUS_RUN = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/"
    "SPECcensus_k5_gpu1_EP5000_PCA10_REWwgl_minID0_majID1_"
    "TRJ2000_REAL3000_GG202604251803_9af13c63"
)
CENSUS_EARLY_EP = 150
CENSUS_LATE_EP  = 4950

# Capture-24: k=5 wgl run (illustrative; best k=3 config not yet synced from Lambda)
# Snapshots every 50 ep.
CAPTURE24_RUN = Path(
    "/storage_1/epigou_storage/FORGE/training_runs/"
    "SPECcapture24_wgl_k5_5000ep_EP5000_PCA10_REWwgl_minID1_majID0_"
    "TRJ2000_REAL3000_GG202604132347_04d4b2e0"
)
CAPTURE24_EARLY_EP = 50
CAPTURE24_LATE_EP  = 5000
PCA_COMPONENTS = 10
SYNTH_MAX_PLOT = 600  # subsample trajectory for readability
RNG = np.random.default_rng(0)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_snapshot(run_dir: Path, seed: int, episode: int) -> np.ndarray:
    p = run_dir / f"seed_{seed}" / "synthetic_snapshots" / f"synthetic_ep{episode:04d}_phase1_class1.npz"
    return np.load(p)["x"]  # (TRAJ_LENGTH, pca_components)


def load_train_data(dataset_name: str, seed: int, minority_id: int, da_pct: float = 0.014):
    majority_id = 1 - minority_id
    ds = Dataset(
        dataset_name,
        multiclass=False,
        minority_id=minority_id,
        majority_id=majority_id,
        third_id=None,
        pca_components=PCA_COMPONENTS,
        seed=seed,
        device="cpu",
        use_pca=True,
    )
    outs = ds.get_data_splits(da_pct=da_pct, train_size=3000, pca_components=PCA_COMPONENTS)
    X = outs[0].numpy()   # (N, 10) PCA features
    y = outs[3].numpy()   # (N,)
    a = ds.a_train.numpy()
    return X, y, a


# ── Plot helper ────────────────────────────────────────────────────────────────

ADV_COLOR = "#4878d0"    # blue
DIS_COLOR = "#d65f5f"    # red
SYN_COLOR = "#f4a522"    # amber

def _robust_limits(arr, lo=2, hi=98, pad=0.15):
    """Return (min, max) based on percentile range with padding."""
    lo_v, hi_v = np.percentile(arr, lo), np.percentile(arr, hi)
    span = hi_v - lo_v or 1.0
    return lo_v - pad * span, hi_v + pad * span


def compute_row_limits(X, y, a, minority_id, *synth_list):
    """Compute shared axis limits for a dataset row from real positives + all snapshots."""
    majority_id = 1 - minority_id
    adv_pos = (y == 1) & (a == majority_id)
    dis_pos = (y == 1) & (a == minority_id)
    pts = [X[adv_pos | dis_pos]] + [s for s in synth_list if s is not None]
    focus = np.vstack(pts)
    return _robust_limits(focus[:, 0]), _robust_limits(focus[:, 1])


def draw_panel(ax, X, y, a, synth_raw, minority_id: int, episode: int,
               xlim=None, ylim=None):
    """Scatter PC1 vs PC2 of real data + synthetic trajectory."""
    majority_id = 1 - minority_id

    # Subsample synthetic trajectory for readability
    if synth_raw.shape[0] > SYNTH_MAX_PLOT:
        idx = RNG.choice(synth_raw.shape[0], SYNTH_MAX_PLOT, replace=False)
        synth = synth_raw[idx]
    else:
        synth = synth_raw

    adv_pos = (y == 1) & (a == majority_id)
    dis_pos = (y == 1) & (a == minority_id)

    # Background: negatives visible inside the shared view
    neg = y == 0
    if xlim and ylim:
        in_view = (
            (X[:, 0] >= xlim[0]) & (X[:, 0] <= xlim[1]) &
            (X[:, 1] >= ylim[0]) & (X[:, 1] <= ylim[1])
        )
    else:
        in_view = np.ones(len(X), dtype=bool)
    ax.scatter(X[neg & in_view, 0], X[neg & in_view, 1],
               c="lightgray", s=4, alpha=0.30, zorder=1, linewidths=0, rasterized=True)

    # Advantaged positives
    ax.scatter(X[adv_pos, 0], X[adv_pos, 1], c=ADV_COLOR, s=22, alpha=0.75,
               zorder=2, linewidths=0, rasterized=True, label="Adv. pos. (real)")

    # Disadvantaged positives (the scarce class)
    ax.scatter(X[dis_pos, 0], X[dis_pos, 1], c=DIS_COLOR, s=35, alpha=0.90,
               marker="s", zorder=3, linewidths=0, rasterized=True,
               label="Disadv. pos. (real)")

    # Synthetic trajectory
    ax.scatter(synth[:, 0], synth[:, 1], c=SYN_COLOR, s=14, alpha=0.55,
               marker="*", zorder=4, linewidths=0, rasterized=True,
               label="Synthetic (FORGE)")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"Episode {episode:,}", fontsize=9, pad=3)
    ax.spines[["top", "right"]].set_visible(False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("figs", exist_ok=True)

    print("Loading Capture-24 training data (seed 42)…")
    X_cap, y_cap, a_cap = load_train_data("capture24", seed=42, minority_id=1, da_pct=0.015)

    print("Loading Census Income training data (seed 42)…")
    X_cen, y_cen, a_cen = load_train_data("census_income", seed=42, minority_id=0, da_pct=0.01433)

    print("Loading snapshots…")
    cap_early = load_snapshot(CAPTURE24_RUN, seed=42, episode=CAPTURE24_EARLY_EP)
    cap_late  = load_snapshot(CAPTURE24_RUN, seed=42, episode=CAPTURE24_LATE_EP)
    cen_early = load_snapshot(CENSUS_RUN,    seed=42, episode=CENSUS_EARLY_EP)
    cen_late  = load_snapshot(CENSUS_RUN,    seed=42, episode=CENSUS_LATE_EP)

    # Compute shared axis limits per row (union of real positives + both snapshots)
    cap_xlim, cap_ylim = compute_row_limits(X_cap, y_cap, a_cap, 1, cap_early, cap_late)
    cen_xlim, cen_ylim = compute_row_limits(X_cen, y_cen, a_cen, 0, cen_early, cen_late)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.35, bottom=0.13)

    # Row 0: Capture-24
    draw_panel(axes[0, 0], X_cap, y_cap, a_cap, cap_early, minority_id=1,
               episode=CAPTURE24_EARLY_EP, xlim=cap_xlim, ylim=cap_ylim)
    draw_panel(axes[0, 1], X_cap, y_cap, a_cap, cap_late,  minority_id=1,
               episode=CAPTURE24_LATE_EP,  xlim=cap_xlim, ylim=cap_ylim)

    # Row 1: Census Income
    draw_panel(axes[1, 0], X_cen, y_cen, a_cen, cen_early, minority_id=0,
               episode=CENSUS_EARLY_EP, xlim=cen_xlim, ylim=cen_ylim)
    draw_panel(axes[1, 1], X_cen, y_cen, a_cen, cen_late,  minority_id=0,
               episode=CENSUS_LATE_EP,  xlim=cen_xlim, ylim=cen_ylim)

    # Row labels
    for ax, label in zip(axes[:, 0], ["Capture-24 (k=5)", "Census Income (k=5, ep=30)"]):
        ax.annotate(
            label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 28, 0),
            xycoords=ax.yaxis.label, textcoords="offset points",
            ha="right", va="center", fontsize=9, fontweight="bold", rotation=90,
        )

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.01), frameon=False)

    for ext in ("pdf", "png"):
        out = f"figs/fig_agent_learning.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
