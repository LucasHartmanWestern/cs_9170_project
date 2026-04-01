"""
Generate per-episode EO learning curves for the episode ablation (Census + Capture-24).

Four configs per dataset: ep800ph0, ep800ph200, ep1500ph400★, ep2000ph600
Mean ± std band across 5 seeds. Smoothed with a rolling window.
COMPAS excluded pending investigation.

Output: fig_episode_learning_curves.png  (paper_figures_v3/ and paper/figures/)
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

EXPECTED_SEEDS = {
    "census":    {"0", "2", "3", "5", "42"},
    "capture24": {"0", "3", "4", "5", "42"},
}

# ep config: (key_fragment, display_label, phase2_start_episode)
# phase2_start_episode = None means no phase 2
EP_CONFIGS = [
    ("ep800ph0",    "ep800 / ph0",    None),
    ("ep800ph200",  "ep800 / ph200",  800),
    ("ep1500ph400", "ep1500 / ph400", 1500),
    ("ep2000ph600", "ep2000 / ph600", 2000),
]
# Mark chosen per dataset
CHOSEN = {"census": "ep1500ph400", "capture24": "ep800ph200"}

COLORS = ["#aec6e8", "#4a90d9", "#1a4f8a", "#f5a623"]
SMOOTH_WINDOW = 30  # rolling mean window (episodes)


def find_run_dir(ds_key, ep_key):
    """Return the path to the best-matching ablation run directory."""
    fragment = f"SPECablation_{ds_key}_{ep_key}_5s"
    matches = [d for d in os.listdir(BASE_V3) if fragment in d]
    if not matches:
        return None
    # Prefer the one with most seed dirs
    matches.sort(key=lambda d: -len([s for s in os.listdir(f"{BASE_V3}/{d}") if s.startswith("seed_")]))
    return os.path.join(BASE_V3, matches[0])


def load_seed_curves(run_dir, expected_seeds, col="fairness.eo_tpr_diff"):
    """Load per-episode metric column for each seed. Returns list of Series."""
    curves = []
    for seed in expected_seeds:
        p = os.path.join(run_dir, f"seed_{seed}", "metrics.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, usecols=["episode", "meta.phase", col])
        # Offset phase-2 episode numbers so they don't overwrite phase-1 entries
        if "meta.phase" in df.columns:
            ph1_len = (df["meta.phase"] == "phase1_class1").sum()
            mask2 = df["meta.phase"] == "phase2_class0"
            df.loc[mask2, "episode"] = df.loc[mask2, "episode"] + ph1_len
        df = df.set_index("episode")[col]
        df = df[~df.index.duplicated(keep="last")]
        curves.append(df)
    return curves


def smooth(series, window):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def align_and_aggregate(curves, smooth_window=SMOOTH_WINDOW):
    """Align curves to a common episode index, smooth, then compute mean/std."""
    if not curves:
        return None, None, None
    # Drop duplicate episode indices (keep last), then align
    curves = [c[~c.index.duplicated(keep="last")] for c in curves]
    all_ep = sorted(set().union(*[c.index for c in curves]))
    aligned = pd.DataFrame({i: c.reindex(all_ep).interpolate(method="index")
                             for i, c in enumerate(curves)})
    smoothed = aligned.apply(lambda col: smooth(col, smooth_window))
    mean = smoothed.mean(axis=1)
    std  = smoothed.std(axis=1, ddof=1)
    return mean.index.values, mean.values, std.values


# ── figure ────────────────────────────────────────────────────────────────────

DATASETS   = ["census", "capture24"]
DS_LABELS  = {"census": "Census Income", "capture24": "Capture-24"}

# (metric_col, ylabel, legend_loc, aux_col, aux_label, ylim_bottom)
# aux_col: horizontal reference line drawn from the first-episode mean across seeds
METRICS = [
    ("fairness.eo_tpr_diff",     "EO gap (↓)",               "upper right",
     "fairness.eo_alpha_baseline",          "Alpha (ERM)",   0.0),
    ("fairness.worst_loss_beta", "Worst-group BCE loss (↓)",  "upper right",
     "fairness.worst_loss_alpha_baseline",  "Alpha (ERM)",   None),
    ("meta.episode_return",      "Episode return (↑)",        "lower right",
     None,                                  None,            None),
]

fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=False)

for col_idx, ds in enumerate(DATASETS):
    seeds  = EXPECTED_SEEDS[ds]
    chosen = CHOSEN[ds]

    for row_idx, (metric_col, ylabel, legend_loc, aux_col, aux_label, ylim_bottom) in enumerate(METRICS):
        ax = axes[row_idx, col_idx]
        aux_vals     = []
        phase_markers = []

        for (ep_key, label, ph2_start), color in zip(EP_CONFIGS, COLORS):
            is_chosen = (ep_key == chosen)
            run_dir = find_run_dir(ds, ep_key)
            if run_dir is None:
                continue

            curves = load_seed_curves(run_dir, seeds, metric_col)
            if not curves:
                continue

            eps, mean, std = align_and_aggregate(curves)

            if aux_col and not aux_vals:
                for c in load_seed_curves(run_dir, seeds, aux_col):
                    if len(c) > 0:
                        aux_vals.append(float(c.iloc[0]))

            lw  = 1.8 if is_chosen else 1.0
            zo  = 3   if is_chosen else 2
            ls  = "-" if is_chosen else "--"
            lab = f"{label} ★" if is_chosen else label

            ax.plot(eps, mean, color=color, linewidth=lw, linestyle=ls,
                    label=lab, zorder=zo)
            ax.fill_between(eps, mean - std, mean + std,
                            color=color, alpha=0.15, zorder=zo - 1)

            if ph2_start is not None:
                phase_markers.append((ph2_start, color, is_chosen))

            if row_idx == 0:
                print(f"  {ds} {ep_key}: {len(curves)} seeds, final EO={mean[-1]:.3f}")

        # Auxiliary alpha baseline line
        if aux_vals:
            ax.axhline(np.mean(aux_vals), color="#555555", linewidth=1.1,
                       linestyle="--", label=f"{aux_label} = {np.mean(aux_vals):.3f}", zorder=1)

        # Return: reference at 0.5 (beta = alpha, no improvement)
        if metric_col == "meta.episode_return":
            ax.axhline(0.5, color="#aaaaaa", linewidth=0.9, linestyle=":",
                       label="beta = alpha (0.5)", zorder=1)

        # Phase-2 markers — must be drawn after data so ylim is set
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)
        ax.autoscale(enable=True, axis="y", tight=False)
        y_lo, y_hi = ax.get_ylim()
        for x_ph2, color, is_chosen in phase_markers:
            ax.axvline(x_ph2, color=color,
                       linewidth=1.4 if is_chosen else 1.0,
                       linestyle=":", alpha=0.85, zorder=1)
            ax.text(x_ph2 + 12, y_lo + (y_hi - y_lo) * 0.97, "→ph2",
                    color=color, fontsize=5.5, va="top", alpha=0.9)

        if row_idx == 0:
            ax.set_title(DS_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        ax.legend(fontsize=7.5, loc=legend_loc)

fig.suptitle("Episode / Phase-2 Budget Ablation — Learning Curves",
             fontsize=12, fontweight="bold")
plt.tight_layout()

for fname in ["fig_episode_learning_curves.png",
              os.path.join(FIG_DIR, "fig_episode_learning_curves.png")]:
    out = fname if os.path.isabs(fname) else os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close()
print("Done.")
