"""
Census-only training curves: episode return, 2×2 grid (one panel per episode config).

Layout:
  [ep800/ph0]      [ep800/ph200 ★chosen]
  [ep1500/ph400 ★] [ep2000/ph600]

Mean ± std band across 5 seeds. Individual seed traces shown faintly.
Phase-2 boundary marked with a dashed red line.

Output: fig_training_curves_census.png  (paper_figures_v3/ and paper/figures/)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_V3  = "/home/epigou/cs_9170_project/paper_results_v3/training_runs"
OUT_DIR  = "/home/epigou/cs_9170_project/paper_figures_v3"
FIG_DIR  = "/home/epigou/cs_9170_project/paper/figures"

SEEDS = ["0", "2", "3", "5", "42"]
CHOSEN_KEY = "ep1500ph400"

# (key_fragment, display_label, gen_both_classes)
EP_CONFIGS = [
    ("ep1500ph400", "ep1500 / ph400", True),
    ("ep2000ph600", "ep2000 / ph600", True),
]

SMOOTH_WINDOW = 40
METRIC_COL    = "meta.episode_return"
COLOR_MAIN    = "#1a4f8a"
COLOR_SEED    = "#6699cc"


def find_run_dir(ep_key):
    fragment = f"SPECablation_census_{ep_key}_5s"
    matches = [d for d in os.listdir(BASE_V3) if fragment in d]
    if not matches:
        return None
    matches.sort(key=lambda d: -len([s for s in os.listdir(f"{BASE_V3}/{d}") if s.startswith("seed_")]))
    return os.path.join(BASE_V3, matches[0])


def smooth(s, w=SMOOTH_WINDOW):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def load_full(path):
    """Load metric with continuous x-axis across phase 1 and phase 2."""
    df = pd.read_csv(path, usecols=["episode", "meta.phase", METRIC_COL])
    phase1 = df[df["meta.phase"] == "phase1_class1"].copy()
    phase2 = df[df["meta.phase"] == "phase2_class0"].copy()
    offset = len(phase1)
    if len(phase2) > 0:
        phase2 = phase2.copy()
        phase2["episode"] = phase2["episode"] + offset
    combined = pd.concat([phase1, phase2])
    combined = combined[~combined["episode"].duplicated(keep="last")].set_index("episode")[METRIC_COL]
    return combined, offset  # offset = phase-1 length = phase-2 x boundary


# ── figure ─────────────────────────────────────────────────────────────────────

fig, axes_row = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
axes = np.array([[axes_row[0], axes_row[1]]])
fig.suptitle("Episode Return — Census Income (5 seeds)",
             fontsize=13, fontweight="bold")

panel_order = [(0,0), (0,1)]

for (row, col), (ep_key, cfg_label, gen_both) in zip(panel_order, EP_CONFIGS):
    ax = axes[row, col]
    is_chosen = (ep_key == CHOSEN_KEY)

    run_dir = find_run_dir(ep_key)
    if run_dir is None:
        ax.text(0.5, 0.5, f"Missing:\n{ep_key}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#cc0000")
        ax.set_title(cfg_label, fontsize=10)
        continue

    seed_curves = []
    actual_ph2_start = None

    for s in SEEDS:
        p = os.path.join(run_dir, f"seed_{s}", "metrics.csv")
        if not os.path.exists(p):
            continue
        try:
            series, ph2_x = load_full(p)
        except Exception as e:
            print(f"  Warning: seed {s} failed: {e}")
            continue

        if ph2_x > 0 and actual_ph2_start is None:
            actual_ph2_start = ph2_x

        smoothed = smooth(series)
        seed_curves.append(smoothed)

        # individual seed trace (faint)
        ax.plot(smoothed.index, smoothed.values,
                color=COLOR_SEED, linewidth=0.5, alpha=0.35, zorder=2)

    if not seed_curves:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        continue

    # Align to common index and aggregate
    min_len = min(len(c) for c in seed_curves)
    arr  = np.array([c.values[:min_len] for c in seed_curves])
    eps  = seed_curves[0].index[:min_len]
    mean = arr.mean(0)
    std  = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)

    ax.plot(eps, mean, color=COLOR_MAIN, linewidth=2.2, zorder=5,
            label=f"Mean (n={len(seed_curves)})")
    ax.fill_between(eps, mean - std, mean + std,
                    color=COLOR_MAIN, alpha=0.20, zorder=4, label="±1 std")

    # Phase-2 boundary
    if actual_ph2_start and actual_ph2_start > 0:
        ax.axvline(actual_ph2_start, color="#cc0000", linewidth=1.5,
                   linestyle="--", alpha=0.85, zorder=6, label="Phase 2 start")
        y_lo, y_hi = ax.get_ylim()
        ax.text(actual_ph2_start + eps[-1] * 0.015, y_hi * 0.98,
                "↓ ph2", color="#cc0000", fontsize=8.5, va="top", fontweight="bold")

    # Reference at 0.5 (beta = alpha, no improvement)
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.9, linestyle=":",
               label="β = α (0.5)", zorder=3)

    # Title styling
    star = "★ " if is_chosen else ""
    title_color = "#1a4f8a" if is_chosen else "black"
    title_weight = "bold" if is_chosen else "normal"
    ax.set_title(f"{star}{cfg_label}", fontsize=10,
                 fontweight=title_weight, color=title_color)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Episode return", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()

for fname in ["fig_training_curves_census.png",
              os.path.join(FIG_DIR, "fig_training_curves_census.png")]:
    out = fname if os.path.isabs(fname) else os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close()
print("Done.")
