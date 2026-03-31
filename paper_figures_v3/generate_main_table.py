"""
Generate main comparison table figure and metrics CSV for paper (Option C).

RL configs used:
  Census     : v3 ep1500/ph400 global-only
  COMPAS     : v3 dvrl (EP2000, lambda=[0.5,0.5])
  Capture-24 : v3 ep800/ph200 global-only

Baselines: paper_results_v2 (GroupDRO, OT Repair, FLB, FairTabDDPM, SMOTE, CTGAN)
Alpha baseline: alpha_* columns from RL runs (same data splits).
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

BASE_V2  = "/home/epigou/cs_9170_project/paper_results_v2"
BASE_V3  = "/home/epigou/cs_9170_project/paper_results_v3/training_runs"
OUT_DIR  = "/home/epigou/cs_9170_project/paper_figures_v3"

# Expected seeds per dataset (for validation)
EXPECTED_SEEDS = {
    "census":    {"0", "2", "3", "5", "42"},
    "compas":    {"1", "3", "6", "7", "42"},
    "capture24": {"0", "3", "4", "5", "42"},
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_ftm(base, dirname, expected_seeds=None):
    p = f"{base}/{dirname}/final_test_metrics.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if expected_seeds:
        df = df[df["seed"].astype(str).isin(expected_seeds)]
    return df if len(df) > 0 else None


def find_dir(base, key, expected_seeds=None, prefer_n=5):
    """Find best matching dir for key; prefer exact-n seed match."""
    matches = [d for d in os.listdir(base) if key in d]
    scored = []
    for d in matches:
        seeds = [s for s in os.listdir(f"{base}/{d}") if s.startswith("seed_")]
        scored.append((abs(len(seeds) - prefer_n), d))
    scored.sort()
    for _, d in scored:
        df = load_ftm(base, d, expected_seeds)
        if df is not None and len(df) >= prefer_n:
            return df
    # fallback: any
    for _, d in scored:
        df = load_ftm(base, d, expected_seeds)
        if df is not None:
            return df
    return None


def stats(df, col):
    """Return (mean, std) for a column, ignoring NaN."""
    v = df[col].dropna()
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


def get_metrics(df):
    """Extract all metrics from a final_test_metrics.csv dataframe."""
    return {
        "eo":  stats(df, "beta_eo_tpr_diff"),
        "auc": stats(df, "beta_roc_auc"),
        "f1w": stats(df, "beta_f1_weighted"),
        "dp":  stats(df, "beta_dp_diff"),
        "eod": stats(df, "beta_eod_max_diff"),
    }


def get_alpha_metrics(df):
    return {
        "eo":  stats(df, "alpha_eo_tpr_diff"),
        "auc": stats(df, "alpha_roc_auc"),
        "f1w": stats(df, "alpha_f1_weighted"),
        "dp":  stats(df, "alpha_dp_diff"),
        "eod": stats(df, "alpha_eod_max_diff"),
    }

# ── data loading ─────────────────────────────────────────────────────────────

datasets = ["census", "compas", "capture24"]
ds_labels = {"census": "Census", "compas": "COMPAS", "capture24": "Capture-24"}

# RL (Option C)
rl_keys = {
    "census":    ("census_ep1500ph400_5s_EP1500",  BASE_V3),
    "compas":    ("compas_dvrl_5s_EP2000",          BASE_V3),
    "capture24": ("capture24_ep800ph200_5s_EP800",  BASE_V3),
}

# Baselines
baseline_keys = {
    "Alpha":       {ds: ("", BASE_V3, True) for ds in datasets},   # special
    "GroupDRO":    {ds: (f"group_dro_v2_{ds}", BASE_V2, False) for ds in datasets},
    "OT Repair":   {ds: (f"gaussian_ot_repair_v2_{ds}", BASE_V2, False) for ds in datasets},
    "FLB":         {ds: (f"fairness_loss_balancing_v2_{ds}", BASE_V2, False) for ds in datasets},
    "FairTabDDPM": {ds: (f"fairtabddpm_v2_{ds}", BASE_V2, False) for ds in datasets},
    "SMOTE":       {ds: (f"smote_v2_{ds}", BASE_V2, False) for ds in datasets},
    "CTGAN":       {ds: (f"ctgan_v2_{ds}", BASE_V2, False) for ds in datasets},
    "RL (ours)":   {ds: rl_keys[ds] + (False,) for ds in datasets},
}

# Method display order
METHOD_ORDER = [
    "Alpha", "GroupDRO", "OT Repair", "FLB",
    "FairTabDDPM", "SMOTE", "CTGAN", "RL (ours)"
]

print("Loading data...")
data = {}  # data[method][dataset] = metrics dict

for method in METHOD_ORDER:
    data[method] = {}
    for ds in datasets:
        seeds = EXPECTED_SEEDS[ds]
        if method == "Alpha":
            # Get alpha metrics from the RL run for this dataset
            key, base = rl_keys[ds]
            df = find_dir(base, key, seeds)
            if df is not None:
                data[method][ds] = get_alpha_metrics(df)
                print(f"  Alpha {ds}: n={len(df)}, seeds={sorted(df['seed'].astype(str).tolist())}")
            else:
                print(f"  MISSING: Alpha {ds}")
        elif method == "RL (ours)":
            key, base, _ = baseline_keys[method][ds]
            df = find_dir(base, key, seeds)
            if df is not None:
                data[method][ds] = get_metrics(df)
                print(f"  RL {ds}: n={len(df)}, seeds={sorted(df['seed'].astype(str).tolist())}")
            else:
                print(f"  MISSING: RL {ds}")
        else:
            key, base, _ = baseline_keys[method][ds]
            df = find_dir(base, key, seeds)
            if df is not None:
                data[method][ds] = get_metrics(df)
                print(f"  {method} {ds}: n={len(df)}, seeds={sorted(df['seed'].astype(str).tolist())}")
            else:
                print(f"  MISSING: {method} {ds}")

# ── build CSV ─────────────────────────────────────────────────────────────────

print("\nBuilding CSV...")
csv_rows = []
for method in METHOD_ORDER:
    for ds in datasets:
        m = data.get(method, {}).get(ds)
        if m is None:
            continue
        row = {
            "method":       method,
            "dataset":      ds_labels[ds],
            "eo_mean":      round(m["eo"][0], 4),
            "eo_std":       round(m["eo"][1], 4),
            "auc_mean":     round(m["auc"][0], 4),
            "auc_std":      round(m["auc"][1], 4),
            "f1w_mean":     round(m["f1w"][0], 4),
            "f1w_std":      round(m["f1w"][1], 4),
            "dp_mean":      round(m["dp"][0], 4),
            "dp_std":       round(m["dp"][1], 4),
            "eod_mean":     round(m["eod"][0], 4),
            "eod_std":      round(m["eod"][1], 4),
        }
        csv_rows.append(row)

csv_df = pd.DataFrame(csv_rows)
csv_path = os.path.join(OUT_DIR, "main_table_metrics.csv")
csv_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# ── determine best per metric per dataset (for bolding) ───────────────────────

# Best = lowest EO, highest AUC, highest F1w
# Exclude Alpha row from "best" competition
COMPETING = [m for m in METHOD_ORDER if m != "Alpha"]

best = {}  # best[ds][metric] = best_value
for ds in datasets:
    best[ds] = {}
    for metric, better in [("eo", min), ("auc", max), ("f1w", max)]:
        vals = []
        for method in COMPETING:
            m = data.get(method, {}).get(ds)
            if m:
                vals.append(m[metric][0])
        best[ds][metric] = better(vals) if vals else None

def is_best(method, ds, metric, tol=1e-4):
    if method == "Alpha":
        return False
    m = data.get(method, {}).get(ds)
    if m is None:
        return False
    b = best[ds].get(metric)
    if b is None:
        return False
    better = min if metric == "eo" else max
    return better(m[metric][0], b) == m[metric][0] and abs(m[metric][0] - b) < tol

# ── figure ───────────────────────────────────────────────────────────────────

print("\nGenerating figure...")

METRICS    = ["eo", "auc", "f1w"]
COL_LABELS = ["EO ↓", "AUC ↑", "F1w ↑"]
N_METHODS  = len(METHOD_ORDER)
N_DS       = len(datasets)
N_METRICS  = len(METRICS)

# Colour palette
COL_HEADER_DS  = "#2b4590"    # dark blue for dataset group headers
COL_HEADER_MET = "#4a7fbf"    # lighter blue for metric sub-headers
COL_RL         = "#fff3cd"    # warm yellow for RL row
COL_ALPHA      = "#f0f0f0"    # light grey for alpha row
COL_ODD        = "#ffffff"
COL_EVEN       = "#f7f9fc"
COL_BEST       = "#d4edda"    # light green for best cell
COL_BORDER     = "#c0c0c0"

# Figure dimensions
FIG_W  = 16
FIG_H  = 6.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Column layout: method label | [census: EO AUC F1w] | [compas: ...] | [capture24: ...]
LEFT_COL  = 1.90   # width of method name column
METRIC_W  = 1.55   # width per metric cell
GROUP_W   = N_METRICS * METRIC_W   # width per dataset group = 4.65
TOTAL_W   = LEFT_COL + N_DS * GROUP_W  # should be ~16

# Row layout (bottom-up from 0)
ROW_H     = 0.48
TOP_PAD   = 0.30
HDR1_H    = 0.52   # dataset header row
HDR2_H    = 0.40   # metric header row
DATA_H    = ROW_H
TOTAL_H   = TOP_PAD + HDR1_H + HDR2_H + N_METHODS * DATA_H + 0.20

# x positions of left edge of each metric column
def col_x(ds_idx, met_idx):
    return LEFT_COL + ds_idx * GROUP_W + met_idx * METRIC_W

# y position (top edge) for row i (0=first data row at top)
def row_y(i):
    return FIG_H - TOP_PAD - HDR1_H - HDR2_H - i * DATA_H

def draw_cell(ax, x, y, w, h, text, fontsize=8.5, bold=False, bg=None,
              color="black", ha="center", va="center", border_color=COL_BORDER):
    if bg:
        rect = plt.Rectangle((x, y - h), w, h,
                              facecolor=bg, edgecolor=border_color, linewidth=0.5)
        ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y - h / 2, text,
            ha=ha, va=va, fontsize=fontsize, fontweight=weight, color=color,
            clip_on=True)

def fmt(mean, std, is_pct=False):
    if is_pct:
        return f"{mean*100:.1f}±{std*100:.1f}"
    return f"{mean:.3f}±{std:.3f}"

# ── draw header row 1: dataset names ─────────────────────────────────────────
y_hdr1 = FIG_H - TOP_PAD
# method column top-left empty
draw_cell(ax, 0, y_hdr1, LEFT_COL, HDR1_H + HDR2_H, "", bg="#f0f0f0")

for di, ds in enumerate(datasets):
    x = col_x(di, 0)
    draw_cell(ax, x, y_hdr1, GROUP_W, HDR1_H,
              ds_labels[ds], fontsize=10, bold=True,
              bg=COL_HEADER_DS, color="white")

# ── draw header row 2: metric names ──────────────────────────────────────────
y_hdr2 = y_hdr1 - HDR1_H
# method col label
draw_cell(ax, 0, y_hdr2, LEFT_COL, HDR2_H,
          "Method", fontsize=9.5, bold=True, bg="#e8e8e8")

for di in range(N_DS):
    for mi, label in enumerate(COL_LABELS):
        x = col_x(di, mi)
        draw_cell(ax, x, y_hdr2, METRIC_W, HDR2_H,
                  label, fontsize=8.5, bold=True,
                  bg=COL_HEADER_MET, color="white")

# ── draw data rows ────────────────────────────────────────────────────────────
for ri, method in enumerate(METHOD_ORDER):
    y = row_y(ri)
    is_rl    = method == "RL (ours)"
    is_alpha = method == "Alpha"
    row_bg   = COL_RL if is_rl else (COL_ALPHA if is_alpha else (COL_ODD if ri % 2 == 0 else COL_EVEN))

    # method name cell
    display = method if not is_rl else "RL (ours) ★"
    draw_cell(ax, 0, y, LEFT_COL, DATA_H,
              display, fontsize=9, bold=is_rl,
              bg=row_bg, ha="left",
              color="#1a1a2e" if is_rl else "black")
    # shift text slightly right
    ax.texts[-1].set_position((0.12, y - DATA_H / 2))

    for di, ds in enumerate(datasets):
        m = data.get(method, {}).get(ds)
        for mi, metric in enumerate(METRICS):
            x = col_x(di, mi)
            if m is None:
                draw_cell(ax, x, y, METRIC_W, DATA_H, "—", bg=row_bg, fontsize=8)
                continue
            mean, std = m[metric]
            txt = fmt(mean, std)
            cell_best = is_best(method, ds, metric)
            cell_bg = COL_BEST if cell_best else row_bg
            draw_cell(ax, x, y, METRIC_W, DATA_H, txt,
                      fontsize=8, bold=cell_best, bg=cell_bg)

# ── outer border ──────────────────────────────────────────────────────────────
border_h = HDR1_H + HDR2_H + N_METHODS * DATA_H
outer = plt.Rectangle(
    (0, FIG_H - TOP_PAD - border_h), TOTAL_W, border_h,
    facecolor="none", edgecolor="#555555", linewidth=1.2
)
ax.add_patch(outer)

# ── vertical group separators ─────────────────────────────────────────────────
for di in range(1, N_DS):
    x = col_x(di, 0)
    ax.plot([x, x],
            [FIG_H - TOP_PAD - border_h, FIG_H - TOP_PAD],
            color="#555555", linewidth=1.0)

# ── footnote ─────────────────────────────────────────────────────────────────
note = ("★ RL configs: Census = ep1500/ph400, λ=[1,1] (global-only);  "
        "COMPAS = ep2000/ph600, λ=[0.5,0.5] (DVRL local reward);  "
        "Capture-24 = ep800/ph200, λ=[1,1] (global-only).  "
        "Bold = best per metric per dataset (excluding Alpha).  "
        "All results: 5 seeds, mean ± std.")
ax.text(0.0, FIG_H - TOP_PAD - border_h - 0.16, note,
        fontsize=6.5, color="#444444", ha="left", va="top",
        style="italic")

plt.tight_layout(pad=0)
out_fig = os.path.join(OUT_DIR, "fig_main_table.png")
plt.savefig(out_fig, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved figure: {out_fig}")
print("Done.")
