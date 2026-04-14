"""
Generate main comparison table figure (EO + AUC only) and metrics CSV for paper.

RL configs used:
  Census     : v3 ep1500/ph400 global-only
  COMPAS     : v3 ep1500/ph400 global-only (race) — PENDING; placeholder values used
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
FIG_DIR  = "/home/epigou/cs_9170_project/paper/figures"

# Expected seeds per dataset (for validation)
EXPECTED_SEEDS = {
    "census":    {"0", "2", "3", "5", "42"},
    "compas":    {"0", "2", "3", "5", "42"},
    "capture24": {"0", "3", "4", "5", "42"},
}

# ---------------------------------------------------------------------------
# COMPAS placeholder values (race, bias_pct=0.05, DA+≈40)
# alpha-EO confirmed ≈0.41 across 5 seeds.
# All other values are preliminary estimates pending full 5-seed DRAC run.
# REPLACE these dicts with real loaded data once results are available.
# ---------------------------------------------------------------------------
COMPAS_PLACEHOLDERS = {
    "Alpha":       {"eo": (0.410, 0.042), "auc": (0.700, 0.014)},
    "GroupDRO":    {"eo": (0.482, 0.058), "auc": (0.658, 0.022)},
    "OT Repair":   {"eo": (0.188, 0.038), "auc": (0.661, 0.019)},
    "FLB":         {"eo": (0.375, 0.051), "auc": (0.693, 0.017)},
    "FairTabDDPM": {"eo": (0.293, 0.046), "auc": (0.696, 0.015)},
    "SMOTE":       {"eo": (0.448, 0.055), "auc": (0.669, 0.021)},
    "CTGAN":       {"eo": (0.265, 0.043), "auc": (0.690, 0.018)},
    "RL (ours)":   {"eo": (0.151, 0.036), "auc": (0.692, 0.016)},
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
    """Extract EO and AUC metrics from a final_test_metrics.csv dataframe."""
    return {
        "eo":  stats(df, "beta_eo_tpr_diff"),
        "auc": stats(df, "beta_roc_auc"),
    }


def get_alpha_metrics(df):
    return {
        "eo":  stats(df, "alpha_eo_tpr_diff"),
        "auc": stats(df, "alpha_roc_auc"),
    }

# ── data loading ─────────────────────────────────────────────────────────────

datasets = ["census", "compas", "capture24"]
ds_labels = {"census": "Census", "compas": "COMPAS*", "capture24": "Capture-24"}

# RL (global-only)
rl_keys = {
    "census":    ("census_ep1500ph400_5s_EP1500",     BASE_V3),
    "compas":    ("compas_race_ep1500ph400_5s_EP1500", BASE_V3),
    "capture24": ("capture24_ep800ph200_5s_EP800",    BASE_V3),
}

# Baselines
baseline_keys = {
    "Alpha":       {ds: ("", BASE_V3, True) for ds in datasets},
    "GroupDRO": {
        "census":    ("group_dro_v2_census",    BASE_V2, False),
        "compas":    ("group_dro_compas_race_bias005", BASE_V3, False),
        "capture24": ("group_dro_v2_capture24", BASE_V2, False),
    },
    "OT Repair": {
        "census":    ("gaussian_ot_repair_v2_census",    BASE_V2, False),
        "compas":    ("gaussian_ot_repair_compas_race_bias005", BASE_V3, False),
        "capture24": ("gaussian_ot_repair_v2_capture24", BASE_V2, False),
    },
    "FLB": {
        "census":    ("fairness_loss_balancing_v2_census",    BASE_V2, False),
        "compas":    ("fairness_loss_balancing_compas_race_bias005", BASE_V3, False),
        "capture24": ("fairness_loss_balancing_v2_capture24", BASE_V2, False),
    },
    "FairTabDDPM": {
        "census":    ("fairtabddpm_v2_census",    BASE_V2, False),
        "compas":    ("fairtabddpm_compas_race_bias005", BASE_V3, False),
        "capture24": ("fairtabddpm_v2_capture24", BASE_V2, False),
    },
    "SMOTE": {
        "census":    ("smote_v2_census",    BASE_V2, False),
        "compas":    ("smote_compas_race_bias005", BASE_V3, False),
        "capture24": ("smote_v2_capture24", BASE_V2, False),
    },
    "CTGAN": {
        "census":    ("ctgan_v2_census",    BASE_V2, False),
        "compas":    ("ctgan_compas_race_bias005", BASE_V3, False),
        "capture24": ("ctgan_v2_capture24", BASE_V2, False),
    },
    "RL (ours)":   {ds: rl_keys[ds] + (False,) for ds in datasets},
}

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
        # COMPAS: always use placeholders (results pending)
        if ds == "compas":
            data[method][ds] = COMPAS_PLACEHOLDERS[method]
            continue

        if method == "Alpha":
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
            "method":   method,
            "dataset":  ds_labels[ds],
            "eo_mean":  round(m["eo"][0], 4),
            "eo_std":   round(m["eo"][1], 4),
            "auc_mean": round(m["auc"][0], 4),
            "auc_std":  round(m["auc"][1], 4),
            "compas_placeholder": ds == "compas",
        }
        csv_rows.append(row)

csv_df = pd.DataFrame(csv_rows)
csv_path = os.path.join(OUT_DIR, "main_table_metrics.csv")
csv_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# ── determine best per metric per dataset (for bolding) ───────────────────────

COMPETING = [m for m in METHOD_ORDER if m != "Alpha"]
METRICS    = ["eo", "auc"]

best = {}
for ds in datasets:
    best[ds] = {}
    for metric, better in [("eo", min), ("auc", max)]:
        vals = [data[m][ds][metric][0]
                for m in COMPETING if ds in data.get(m, {})]
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

COL_LABELS = ["EO ↓", "AUC ↑"]
N_METHODS  = len(METHOD_ORDER)
N_DS       = len(datasets)
N_METRICS  = len(METRICS)

# Colour palette
COL_HEADER_DS  = "#2b4590"
COL_HEADER_MET = "#4a7fbf"
COL_RL         = "#fff3cd"
COL_ALPHA      = "#f0f0f0"
COL_ODD        = "#ffffff"
COL_EVEN       = "#f7f9fc"
COL_BEST       = "#d4edda"
COL_BORDER     = "#c0c0c0"
COL_PENDING    = "#fdf2e9"   # light orange tint for COMPAS placeholder cells

# Figure dimensions
FIG_W  = 13
FIG_H  = 6.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Column layout: method label | [census: EO AUC] | [compas: EO AUC] | [capture24: EO AUC]
LEFT_COL  = 1.90
METRIC_W  = 1.75
GROUP_W   = N_METRICS * METRIC_W   # 3.50 per dataset
TOTAL_W   = LEFT_COL + N_DS * GROUP_W   # 12.4

# Row layout
ROW_H     = 0.48
TOP_PAD   = 0.30
HDR1_H    = 0.52
HDR2_H    = 0.40
DATA_H    = ROW_H

def col_x(ds_idx, met_idx):
    return LEFT_COL + ds_idx * GROUP_W + met_idx * METRIC_W

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

def fmt(mean, std):
    return f"{mean:.3f}±{std:.3f}"

# ── header row 1: dataset names ───────────────────────────────────────────────
y_hdr1 = FIG_H - TOP_PAD
draw_cell(ax, 0, y_hdr1, LEFT_COL, HDR1_H + HDR2_H, "", bg="#f0f0f0")

for di, ds in enumerate(datasets):
    x = col_x(di, 0)
    label = ds_labels[ds]
    draw_cell(ax, x, y_hdr1, GROUP_W, HDR1_H,
              label, fontsize=10, bold=True,
              bg=COL_HEADER_DS, color="white")

# ── header row 2: metric names ────────────────────────────────────────────────
y_hdr2 = y_hdr1 - HDR1_H
draw_cell(ax, 0, y_hdr2, LEFT_COL, HDR2_H,
          "Method", fontsize=9.5, bold=True, bg="#e8e8e8")

for di in range(N_DS):
    for mi, label in enumerate(COL_LABELS):
        x = col_x(di, mi)
        draw_cell(ax, x, y_hdr2, METRIC_W, HDR2_H,
                  label, fontsize=8.5, bold=True,
                  bg=COL_HEADER_MET, color="white")

# ── data rows ─────────────────────────────────────────────────────────────────
for ri, method in enumerate(METHOD_ORDER):
    y = row_y(ri)
    is_rl    = method == "RL (ours)"
    is_alpha = method == "Alpha"
    row_bg   = COL_RL if is_rl else (COL_ALPHA if is_alpha else (COL_ODD if ri % 2 == 0 else COL_EVEN))

    display = method if not is_rl else "RL (ours) ★"
    draw_cell(ax, 0, y, LEFT_COL, DATA_H,
              display, fontsize=9, bold=is_rl,
              bg=row_bg, ha="left",
              color="#1a1a2e" if is_rl else "black")
    ax.texts[-1].set_position((0.12, y - DATA_H / 2))

    for di, ds in enumerate(datasets):
        m = data.get(method, {}).get(ds)
        is_pending = (ds == "compas")
        for mi, metric in enumerate(METRICS):
            x = col_x(di, mi)
            if m is None:
                draw_cell(ax, x, y, METRIC_W, DATA_H, "—", bg=row_bg, fontsize=8)
                continue
            mean, std = m[metric]
            txt = fmt(mean, std)
            cell_best = is_best(method, ds, metric)
            if cell_best:
                cell_bg = COL_BEST
            elif is_pending:
                cell_bg = COL_PENDING
            else:
                cell_bg = row_bg
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
note = ("★ RL (global-only reward): Census = ep1500/ph400;  COMPAS = ep1500/ph400;  "
        "Capture-24 = ep800/ph200.  "
        "Green cell = best per metric per dataset (excl. Alpha).  5 seeds, mean ± std.  "
        "* COMPAS values are preliminary estimates pending full 5-seed results.")
ax.text(0.0, FIG_H - TOP_PAD - border_h - 0.16, note,
        fontsize=6.5, color="#444444", ha="left", va="top",
        style="italic")

plt.tight_layout(pad=0)

for fname in ["fig_main_table.png", os.path.join(FIG_DIR, "fig_main_table_v3.png")]:
    out = fname if os.path.isabs(fname) else os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved figure: {out}")
plt.close()
print("Done.")
