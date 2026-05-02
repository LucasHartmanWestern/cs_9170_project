"""
Training curves: episode return for two panels.

Layout:
  [Census — ep1500/ph400 (chosen)]
  [Capture-24 — ep800/ph200 (chosen)]

Also generates a no-phase-2 variant showing only phase-1 portion of each.

Mean ± range band across seeds. Individual seed traces shown faintly.
Phase-2 boundary marked with a dashed red line (two-phase variant only).

Output:
  fig_training_curves_census.png        — two-phase version
  fig_training_curves_nophase2.png      — phase-1 only version
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_V3  = "/home/epigou/cs_9170_project/archive_runs/paper_results_v3/training_runs"
OUT_DIR  = "/home/epigou/cs_9170_project/figures"
FIG_DIR  = "/home/epigou/cs_9170_project/paper/figures"

SEEDS = {
    "census":    ["0", "2", "3", "5", "42"],
    "capture24": ["0", "3", "4", "5", "42"],
}

# (dataset, key_fragment, display_label, gen_both_classes)
PANELS = [
    ("census",    "ep1500ph400", "ep1500 / ph400", True),
    ("capture24", "ep800ph200",  "ep800 / ph200",  True),
]

SMOOTH_WINDOW = 40
METRIC_COL    = "meta.episode_return"
COLOR_MAIN    = {"census": "#1a4f8a", "capture24": "#27ae60"}
COLOR_SEED    = {"census": "#6699cc", "capture24": "#7ecfa0"}


def find_run_dir(ds, ep_key):
    fragment = f"SPECablation_{ds}_{ep_key}_5s"
    matches = [d for d in os.listdir(BASE_V3) if fragment in d]
    if not matches:
        return None
    matches.sort(key=lambda d: -len([s for s in os.listdir(f"{BASE_V3}/{d}") if s.startswith("seed_")]))
    return os.path.join(BASE_V3, matches[0])


def smooth(s, w=SMOOTH_WINDOW):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def load_full(path, phase1_only=False):
    """Load metric with continuous x-axis. If phase1_only, drop phase-2 rows."""
    df = pd.read_csv(path, usecols=["episode", "meta.phase", METRIC_COL])
    phase1 = df[df["meta.phase"] == "phase1_class1"].copy()
    phase2 = df[df["meta.phase"] == "phase2_class0"].copy()
    offset = len(phase1)
    if not phase1_only and len(phase2) > 0:
        phase2 = phase2.copy()
        phase2["episode"] = phase2["episode"] + offset
        combined = pd.concat([phase1, phase2])
    else:
        combined = phase1
    combined = combined[~combined["episode"].duplicated(keep="last")].set_index("episode")[METRIC_COL]
    return combined, offset


def make_fig(phase1_only=False):
    fig, axes_col = plt.subplots(1, 2, figsize=(9.0, 3.75), sharey=False)

    for panel_idx, (ax, (ds, ep_key, cfg_label, gen_both)) in enumerate(zip(axes_col, PANELS)):
        panel_label = ["a)", "b)"][panel_idx]
        ax.text(-0.20, 0.97, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)
        seeds = SEEDS[ds]
        seed_curves = []
        actual_ph2_start = None

        run_dir = find_run_dir(ds, ep_key)
        if run_dir is None:
            ax.text(0.5, 0.5, f"Missing:\n{ds} {ep_key}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#cc0000")
            continue

        for s in seeds:
            p = os.path.join(run_dir, f"seed_{s}", "metrics.csv")
            if not os.path.exists(p):
                continue
            try:
                series, ph2_x = load_full(p, phase1_only=phase1_only)
            except Exception as e:
                print(f"  Warning: {ds} seed {s} failed: {e}")
                continue

            if ph2_x > 0 and actual_ph2_start is None:
                actual_ph2_start = ph2_x

            smoothed = smooth(series)
            seed_curves.append(smoothed)

            ax.plot(smoothed.index, smoothed.values,
                    color=COLOR_SEED[ds], linewidth=0.5, alpha=0.35, zorder=2)

        if not seed_curves:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            continue

        min_len = min(len(c) for c in seed_curves)
        arr  = np.array([c.values[:min_len] for c in seed_curves])
        eps  = seed_curves[0].index[:min_len]
        mean    = arr.mean(0)
        arr_min = arr.min(0)
        arr_max = arr.max(0)

        ds_mean_label = {"census": "Mean Census Income", "capture24": "Mean Capture-24"}[ds]
        ax.plot(eps, mean, color=COLOR_MAIN[ds], linewidth=2.2, zorder=5, label=ds_mean_label)
        ax.fill_between(eps, arr_min, arr_max,
                        color=COLOR_MAIN[ds], alpha=0.20, zorder=4, label="Min-max range")

        if not phase1_only and actual_ph2_start and actual_ph2_start > 0:
            ax.axvline(actual_ph2_start, color="#cc0000", linewidth=1.5,
                       linestyle="--", alpha=0.85, zorder=6, label="Phase 2 start")

        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Episode return", fontsize=14)
        ax.tick_params(labelsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    return fig


# ── two-phase figure ───────────────────────────────────────────────────────────
fig = make_fig(phase1_only=False)
for fname in ["fig_training_curves_census.png",
              os.path.join(FIG_DIR, "fig_training_curves_census.png")]:
    out = fname if os.path.isabs(fname) else os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
plt.close()

# ── phase-1-only figure ────────────────────────────────────────────────────────
fig = make_fig(phase1_only=True)
for fname in ["fig_training_curves_nophase2.png",
              os.path.join(FIG_DIR, "fig_training_curves_nophase2.png")]:
    out = fname if os.path.isabs(fname) else os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
plt.close()

print("Done.")
