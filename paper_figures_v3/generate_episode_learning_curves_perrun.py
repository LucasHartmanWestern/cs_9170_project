"""
Per-run learning curves for the episode ablation (Census + Capture-24).

Each config gets its own column; each metric gets its own row.
Individual seed traces shown as thin lines + mean as thick line.
Annotates gen_both_classes to flag the algorithmic difference.

Two figures: fig_episode_perrun_census.png, fig_episode_perrun_capture24.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_V3 = "/home/epigou/cs_9170_project/paper_results_v3/training_runs"
OUT_DIR = "/home/epigou/cs_9170_project/paper_figures_v3"
FIG_DIR = "/home/epigou/cs_9170_project/paper/figures"

EXPECTED_SEEDS = {
    "census":    ["0", "2", "3", "5", "42"],
    "capture24": ["0", "3", "4", "5", "42"],
}

# (key_fragment, display_label, phase2_start, gen_both_classes)
EP_CONFIGS = [
    ("ep800ph0",    "ep800 / ph0\n(minority only)",  None, False),
    ("ep800ph200",  "ep800 / ph200\n(+majority)",    800,  True),
    ("ep1500ph400", "ep1500 / ph400\n(+majority)",   1500, True),
    ("ep2000ph600", "ep2000 / ph600\n(+majority)",   2000, True),
]
CHOSEN = {"census": "ep1500ph400", "capture24": "ep800ph200"}

METRICS = [
    ("fairness.eo_tpr_diff",     "EO gap (↓)",              "fairness.eo_alpha_baseline",         True),
    ("fairness.worst_loss_beta", "Worst-group BCE loss (↓)", "fairness.worst_loss_alpha_baseline", False),
    ("meta.episode_return",      "Episode return (↑)",       None,                                 False),
]

SMOOTH_WINDOW = 40


def find_run_dir(ds, ep_key):
    fragment = f"SPECablation_{ds}_{ep_key}_5s"
    matches = [d for d in os.listdir(BASE_V3) if fragment in d]
    if not matches:
        return None
    matches.sort(key=lambda d: -len([s for s in os.listdir(f"{BASE_V3}/{d}") if s.startswith("seed_")]))
    return os.path.join(BASE_V3, matches[0])


def smooth(s, w=SMOOTH_WINDOW):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def load_full(path, col):
    """Load col with a continuous x-axis across phase 1 and phase 2 (episode resets offset)."""
    df = pd.read_csv(path, usecols=["episode", "meta.phase", col])
    # Build continuous episode index: offset phase-2 episodes by phase-1 length
    phase1 = df[df["meta.phase"] == "phase1_class1"].copy()
    phase2 = df[df["meta.phase"] == "phase2_class0"].copy()
    offset = len(phase1)
    phase2 = phase2.copy()
    phase2["episode"] = phase2["episode"] + offset
    combined = pd.concat([phase1, phase2])
    combined = combined[~combined["episode"].duplicated(keep="last")].set_index("episode")[col]
    return combined, offset  # offset = phase-1 length = phase-2 start x


# ── one figure per dataset ────────────────────────────────────────────────────

for ds in ["census", "capture24"]:
    seeds  = EXPECTED_SEEDS[ds]
    chosen = CHOSEN[ds]
    n_cfg  = len(EP_CONFIGS)
    n_met  = len(METRICS)

    fig, axes = plt.subplots(n_met, n_cfg, figsize=(4.5 * n_cfg, 3.5 * n_met),
                             sharey="row", squeeze=False)
    ds_label = "Census Income" if ds == "census" else "Capture-24"
    fig.suptitle(f"{ds_label} — Per-Config Learning Curves (individual seeds)",
                 fontsize=12, fontweight="bold")

    for cfg_idx, (ep_key, cfg_label, ph2_start, gen_both) in enumerate(EP_CONFIGS):
        run_dir = find_run_dir(ds, ep_key)
        is_chosen = (ep_key == chosen)

        for met_idx, (metric_col, ylabel, aux_col, ylim_zero) in enumerate(METRICS):
            ax = axes[met_idx, cfg_idx]

            seed_curves = []
            actual_ph2_start = None
            aux_val = None

            for s in seeds:
                if run_dir is None:
                    continue
                p = os.path.join(run_dir, f"seed_{s}", "metrics.csv")
                if not os.path.exists(p):
                    continue
                try:
                    series, ph2_x = load_full(p, metric_col)
                except Exception:
                    continue

                if ph2_x > 0 and actual_ph2_start is None:
                    actual_ph2_start = ph2_x

                smoothed = smooth(series)
                seed_curves.append(smoothed)

                if aux_col and aux_val is None:
                    try:
                        aux_s, _ = load_full(p, aux_col)
                        aux_val = float(aux_s.iloc[0])
                    except Exception:
                        pass

            if not seed_curves:
                continue

            # Align seeds to common index, compute mean ± std
            min_len = min(len(c) for c in seed_curves)
            arr  = np.array([c.values[:min_len] for c in seed_curves])
            eps  = seed_curves[0].index[:min_len]
            mean = arr.mean(0)
            std  = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)

            ax.plot(eps, mean, color="#1a4f8a", linewidth=2.0, zorder=5, label="mean (5 seeds)")
            ax.fill_between(eps, mean - std, mean + std,
                            color="#1a4f8a", alpha=0.18, zorder=4)

            # Alpha baseline reference
            if aux_val is not None:
                ax.axhline(aux_val, color="#555555", linewidth=1.1,
                           linestyle="--", label=f"Alpha = {aux_val:.3f}", zorder=3)

            if metric_col == "meta.episode_return":
                ax.axhline(0.5, color="#aaaaaa", linewidth=0.8,
                           linestyle=":", label="beta = alpha (0.5)", zorder=3)

            # Phase-2 boundary
            if actual_ph2_start and actual_ph2_start > 0:
                ax.axvline(actual_ph2_start, color="#cc0000", linewidth=1.3,
                           linestyle="--", alpha=0.8, zorder=3, label="phase 2 start")
                ax.text(actual_ph2_start + (eps[-1] * 0.015), ax.get_ylim()[1] * 0.97,
                        "← ph2", color="#cc0000", fontsize=7, va="top")

            if ylim_zero:
                ax.set_ylim(bottom=0)

            # Column header (top row only)
            if met_idx == 0:
                star = "★ " if is_chosen else ""
                tag  = "[single-phase]" if not gen_both else "[two-phase]"
                ax.set_title(f"{star}{cfg_label}\n{tag}",
                             fontsize=8.5, fontweight="bold" if is_chosen else "normal",
                             color="#1a4f8a" if is_chosen else "black")

            ax.set_ylabel(ylabel if cfg_idx == 0 else "", fontsize=8)
            ax.set_xlabel("Episode", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linewidth=0.3, alpha=0.4)

            if met_idx == 0 and cfg_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    for fname in [f"fig_episode_perrun_{ds}.png",
                  os.path.join(FIG_DIR, f"fig_episode_perrun_{ds}.png")]:
        out = fname if os.path.isabs(fname) else os.path.join(OUT_DIR, fname)
        plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close()

print("Done.")
