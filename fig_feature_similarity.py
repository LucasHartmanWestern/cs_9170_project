"""
Figure: Feature-space separation of protected-group positives across datasets.
Shows PC1 vs PC2 of positive-class examples, coloured by protected group,
to illustrate why census_income and capture24 are viable for FORGE while
COMPAS and PTB-XL are not.

Run from project root:  python fig_feature_similarity.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pathlib import Path

from dataset import Dataset

# ── Dataset configs ────────────────────────────────────────────────────────────
#  Each entry: (dataset_name, minority_id, dp_protected_col, adv_label, dis_label, title)
DATASETS = [
    ("census_income", 0, "sex",  "Male (adv.)",   "Female (dis.)", "Census Income"),
    ("capture24",     1, "sex",  "Male (adv.)",   "Female (dis.)", "Capture-24"),
    ("compas",        None, "sex", "Male",         "Female",        "COMPAS"),
    ("ptb_xl",        None, "age", "Old (adv.)",   "Young (dis.)",  "PTB-XL"),
]

PCA_COMPONENTS = 2   # fit 2-component PCA directly for visualisation
SEED = 42
KDE_LEVELS = 4        # number of KDE contour levels
MAX_BG_PTS = 3000     # background scatter cap per panel (for large datasets)
MAX_POS_PTS = 2000    # positive-class scatter cap per group


# ── Helpers ────────────────────────────────────────────────────────────────────

ADV_COLOR = "#4878d0"
DIS_COLOR = "#d65f5f"

RNG = np.random.default_rng(0)


def subsample(arr, n, rng=RNG):
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), n, replace=False)]


def _robust_limits(arr, lo=1, hi=99, pad=0.12):
    lo_v, hi_v = np.percentile(arr, lo), np.percentile(arr, hi)
    span = hi_v - lo_v or 1.0
    return lo_v - pad * span, hi_v + pad * span


def load_dataset(dataset_name: str, dp_protected_col: str, minority_id):
    """Return X_train (2D PCA), y_train, a_train as numpy arrays."""
    minority_id_arg = minority_id if minority_id is not None else 1
    majority_id_arg = 1 - minority_id_arg

    ds = Dataset(
        dataset_name,
        multiclass=False,
        minority_id=minority_id_arg,
        majority_id=majority_id_arg,
        third_id=None,
        pca_components=PCA_COMPONENTS,
        seed=SEED,
        device="cpu",
        use_pca=True,
    )
    kw = dict(pca_components=PCA_COMPONENTS)
    if dataset_name in ("census_income", "compas"):
        kw["dp_protected_col"] = dp_protected_col
    elif dataset_name == "ptb_xl":
        kw["dp_protected_col"] = dp_protected_col
    # capture24 uses sex by default; no extra arg needed

    outs = ds.get_data_splits(**kw)
    X = outs[0].numpy()
    y = outs[3].numpy()
    a = ds.a_train.numpy()
    return X, y, a


def kde_contour(ax, pts, color, levels=KDE_LEVELS, alpha=0.75):
    """Overlay a 2-D KDE contour for a set of 2-D points."""
    if len(pts) < 10:
        return
    try:
        kde = gaussian_kde(pts.T, bw_method="scott")
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        margin_x = (xmax - xmin) * 0.15
        margin_y = (ymax - ymin) * 0.15
        gx = np.linspace(xmin - margin_x, xmax + margin_x, 120)
        gy = np.linspace(ymin - margin_y, ymax + margin_y, 120)
        GX, GY = np.meshgrid(gx, gy)
        Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        ax.contour(GX, GY, Z, levels=levels, colors=color, alpha=alpha, linewidths=0.9)
    except Exception:
        pass


def draw_panel(ax, X, y, a, adv_label, dis_label, title, minority_id):
    """Draw one dataset panel."""
    minority_id_val = minority_id if minority_id is not None else 1
    majority_id_val = 1 - minority_id_val

    adv_pos = (y == 1) & (a == majority_id_val)
    dis_pos = (y == 1) & (a == minority_id_val)
    n_adv, n_dis = adv_pos.sum(), dis_pos.sum()

    # Robust axis limits from positive examples (handles PTB-XL outliers)
    pos_pts = X[adv_pos | dis_pos]
    xlim = _robust_limits(pos_pts[:, 0]) if len(pos_pts) > 4 else None
    ylim = _robust_limits(pos_pts[:, 1]) if len(pos_pts) > 4 else None

    # All data as grey background (subsampled for large datasets)
    bg_pts = subsample(X, MAX_BG_PTS)
    ax.scatter(bg_pts[:, 0], bg_pts[:, 1], c="lightgray", s=3, alpha=0.20,
               zorder=1, linewidths=0, rasterized=True)

    # Scatter positive examples by group (subsampled)
    adv_pts = subsample(X[adv_pos], MAX_POS_PTS)
    dis_pts = subsample(X[dis_pos], MAX_POS_PTS)
    ax.scatter(adv_pts[:, 0], adv_pts[:, 1], c=ADV_COLOR, s=10, alpha=0.60,
               zorder=3, linewidths=0, rasterized=True)
    ax.scatter(dis_pts[:, 0], dis_pts[:, 1], c=DIS_COLOR, s=10, alpha=0.60,
               marker="s", zorder=3, linewidths=0, rasterized=True)

    # KDE contours
    if n_adv >= 10:
        kde_contour(ax, X[adv_pos], color=ADV_COLOR)
    if n_dis >= 10:
        kde_contour(ax, X[dis_pos], color=DIS_COLOR)

    # Counts in corner
    ax.text(0.97, 0.97, f"n={n_adv:,}", color=ADV_COLOR, fontsize=7,
            ha="right", va="top", transform=ax.transAxes)
    ax.text(0.97, 0.89, f"n={n_dis:,}", color=DIS_COLOR, fontsize=7,
            ha="right", va="top", transform=ax.transAxes)

    if xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10, pad=4, fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("figs", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    fig.subplots_adjust(hspace=0.40, wspace=0.35, bottom=0.12)

    for ax, (ds_name, min_id, dp_col, adv_lbl, dis_lbl, title) in zip(axes.flat, DATASETS):
        print(f"Loading {ds_name}…")
        try:
            X, y, a = load_dataset(ds_name, dp_col, min_id)
            draw_panel(ax, X, y, a, adv_lbl, dis_lbl, title, min_id)
        except Exception as exc:
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, f"Load error:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red", wrap=True)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ADV_COLOR,
               markersize=8, label="Advantaged group (pos.)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=DIS_COLOR,
               markersize=8, label="Disadvantaged group (pos.)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
               markersize=8, label="All classes (background)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.01), frameon=False)

    for ext in ("pdf", "png"):
        out = f"figs/fig_feature_similarity.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
