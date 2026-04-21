"""Generate sweep comparison report with figures."""
import os, re, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "analysis_figures/sweep_report"
os.makedirs(OUT_DIR, exist_ok=True)
RUNS_DIR = "training_runs"

# ── run index ────────────────────────────────────────────────────────────────
RUNS = {
    ("census",    "no sigmoid"): "SPECcensus_wgl_k0_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132257_4e035c66",
    ("census",    "k=3"):        "SPECcensus_wgl_k3_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_1ba3fd5c",
    ("census",    "k=5"):        "SPECcensus_wgl_k5_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_d96ace32",
    ("census",    "k=10"):       "SPECcensus_wgl_k10_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_97dd3630",
    ("census",    "λ=0.3"):      "SPECcensus_roc_eo_lam03_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132314_f70958de",
    ("census",    "λ=0.5"):      "SPECcensus_roc_eo_lam05_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132327_6f8ed797",
    ("census",    "λ=0.7"):      "SPECcensus_roc_eo_lam07_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132330_6680a7d8",
    ("compas",    "no sigmoid"): "SPECcompas_wgl_k0_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_294539a3",
    ("compas",    "k=3"):        "SPECcompas_wgl_k3_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_be5e33ae",
    ("compas",    "k=5"):        "SPECcompas_wgl_k5_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_5e3f63f1",
    ("compas",    "k=10"):       "SPECcompas_wgl_k10_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_80bc39af",
    ("compas",    "λ=0.3"):      "SPECcompas_roc_eo_lam03_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_8afd81e8",
    ("compas",    "λ=0.5"):      "SPECcompas_roc_eo_lam05_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_a400fcf4",
    ("compas",    "λ=0.7"):      "SPECcompas_roc_eo_lam07_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_73dab287",
    ("capture24", "no sigmoid"): "SPECcapture24_wgl_k0_5000ep_EP5000_PCA10_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202604132330_96ec1827",
    ("capture24", "k=3"):        "SPECcapture24_wgl_k3_5000ep_EP5000_PCA10_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202604132333_66d7b41c",
    ("capture24", "k=5"):        "SPECcapture24_wgl_k5_5000ep_EP5000_PCA10_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202604132347_04d4b2e0",
    ("capture24", "k=10"):       "SPECcapture24_wgl_k10_5000ep_EP5000_PCA10_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202604140000_99396ee7",
    ("capture24", "λ=0.3"):      "SPECcapture24_roc_eo_lam03_5000ep_EP5000_PCA10_REWroc_eo_minID1_majID0_TRJ2000_REAL3000_GG202604140000_6df7adb4",
    ("capture24", "λ=0.5"):      "SPECcapture24_roc_eo_lam05_5000ep_EP5000_PCA10_REWroc_eo_minID1_majID0_TRJ2000_REAL3000_GG202604140000_ff1d0dda",
    ("capture24", "λ=0.7"):      "SPECcapture24_roc_eo_lam07_5000ep_EP5000_PCA10_REWroc_eo_minID1_majID0_TRJ2000_REAL3000_GG202604140003_cce953ab",
}

BASELINE_LABEL_MAP = {
    "group_dro":               "GroupDRO",
    "fairness_loss_balancing": "FLB",
    "gaussian_ot_repair":      "OT Repair",
    "ctgan":                   "CTGAN",
    "smote":                   "SMOTE",
    "fairtabddpm":             "FairTabDDPM",
}

SEEDS = ["seed_0", "seed_1", "seed_42"]
DATASETS = ["census", "compas", "capture24"]
DATASET_LABELS = {"census": "Census", "compas": "COMPAS", "capture24": "Capture-24"}
K_LABELS   = ["no sigmoid", "k=3", "k=5", "k=10"]
LAM_LABELS = ["λ=0.3", "λ=0.5", "λ=0.7"]
BEST_CONFIG = {"census": "k=3", "compas": "λ=0.5", "capture24": "k=3"}

DS_COLORS  = {"census": "#2196F3", "compas": "#E91E63", "capture24": "#4CAF50"}
K_COLORS   = {"no sigmoid": "#9E9E9E", "k=3": "#1565C0", "k=5": "#42A5F5", "k=10": "#B3E5FC"}
LAM_COLORS = {"λ=0.3": "#BF360C", "λ=0.5": "#EF6C00", "λ=0.7": "#FFA726"}

# ── helpers ──────────────────────────────────────────────────────────────────
def read_summary(run_dir):
    path = os.path.join(RUNS_DIR, run_dir, "analysis", "summary.txt")
    result = {}
    with open(path) as f:
        text = f.read()
    for m in re.finditer(r'([\w\-α-βΔ]+)=([-\d.]+)±([\d.]+)%?', text):
        key, val, std = m.group(1), float(m.group(2)), float(m.group(3))
        result[key] = (val, std)
    dz = re.search(r'Deadzone=([\d.]+)%±([\d.]+)%', text)
    if dz:
        result["deadzone_mean"] = float(dz.group(1))
    si = re.search(r'Seeds improved:\s*(\S+)', text)
    if si:
        result["seeds_improved"] = si.group(1)
    return result

def read_per_seed(run_dir):
    path = os.path.join(RUNS_DIR, run_dir, "analysis", "summary.txt")
    rows = []
    with open(path) as f:
        lines = f.readlines()
    in_table = False
    for line in lines:
        if "Seed" in line and "α-EO" in line:
            in_table = True; continue
        if in_table and line.strip().startswith("---"): continue
        if in_table and line.strip() == "": break
        if in_table:
            parts = line.split()
            if len(parts) >= 11:
                try:
                    rows.append({"seed": int(parts[0]), "alpha_eo": float(parts[1]),
                                 "beta_eo": float(parts[2]), "eo_delta": float(parts[3]),
                                 "beta_f1w": float(parts[5])})
                except (ValueError, IndexError):
                    pass
    return rows

def read_final_test_metrics(run_dir):
    """Return mean of fairness cols across all seeds from experiment-level CSV."""
    p = os.path.join(RUNS_DIR, run_dir, "final_test_metrics.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    out = {}
    for col in ["beta_dp_diff", "beta_eod_max_diff", "alpha_dp_diff", "alpha_eod_max_diff",
                "beta_eo_tpr_diff", "alpha_eo_tpr_diff", "beta_f1_weighted", "beta_roc_auc"]:
        if col in df.columns:
            out[col + "_mean"] = df[col].mean()
            out[col + "_std"]  = df[col].std()
    return out

def load_metrics(run_dir):
    dfs = []
    for seed in SEEDS:
        p = os.path.join(RUNS_DIR, run_dir, seed, "metrics.csv")
        if os.path.exists(p):
            df = pd.read_csv(p); df["seed"] = seed; dfs.append(df)
    return pd.concat(dfs) if dfs else None

def fmt(val, std=None):
    if pd.isna(val): return "—"
    s = f"{val:.3f}"
    if std is not None and not pd.isna(std):
        s += f" ± {std:.3f}"
    return s

# ── collect RL summary stats ──────────────────────────────────────────────────
records = []
for (ds, label), run_dir in RUNS.items():
    try:
        s   = read_summary(run_dir)
        ftm = read_final_test_metrics(run_dir)
        records.append({
            "dataset": ds, "label": label,
            "beta_eo_mean":    s.get("β-EO",  (np.nan,)*2)[0],
            "beta_eo_std":     s.get("β-EO",  (np.nan,)*2)[1],
            "eo_delta_mean":   s.get("EO-Δ",  (np.nan,)*2)[0],
            "alpha_eo_mean":   s.get("α-EO",  (np.nan,)*2)[0],
            "beta_f1w_mean":   s.get("β-F1w", (np.nan,)*2)[0],
            "beta_f1w_std":    s.get("β-F1w", (np.nan,)*2)[1],
            "beta_auc_mean":   s.get("β-AUC", (np.nan,)*2)[0],
            "beta_dp_mean":    ftm.get("beta_dp_diff_mean",      np.nan),
            "beta_dp_std":     ftm.get("beta_dp_diff_std",       np.nan),
            "beta_eod_mean":   ftm.get("beta_eod_max_diff_mean", np.nan),
            "beta_eod_std":    ftm.get("beta_eod_max_diff_std",  np.nan),
            "alpha_dp_mean":   ftm.get("alpha_dp_diff_mean",     np.nan),
            "alpha_eod_mean":  ftm.get("alpha_eod_max_diff_mean",np.nan),
            "deadzone":        s.get("deadzone_mean", np.nan),
            "seeds_improved":  s.get("seeds_improved", "?"),
            "per_seed":        read_per_seed(run_dir),
        })
    except Exception as e:
        print(f"WARNING: {ds}/{label}: {e}")

df_rl = pd.DataFrame(records)

# ── collect baseline stats ────────────────────────────────────────────────────
bl_records = []
for d in sorted(os.listdir(RUNS_DIR)):
    if not d.startswith("BASELINE_"): continue
    fpath = f"{RUNS_DIR}/{d}/final_test_metrics.csv"
    if not os.path.exists(fpath): continue
    df = pd.read_csv(fpath)
    dataset = next((ds for ds in DATASETS if ds in d), None)
    if not dataset: continue
    raw_method = d.split(f"_{dataset}")[0].replace("BASELINE_", "")
    method = BASELINE_LABEL_MAP.get(raw_method, raw_method)
    row = {"dataset": dataset, "method": method, "n_seeds": len(df)}
    for col in ["alpha_eo_tpr_diff", "beta_eo_tpr_diff", "beta_dp_diff",
                "beta_eod_max_diff", "beta_f1_weighted", "beta_roc_auc",
                "alpha_dp_diff", "alpha_eod_max_diff"]:
        if col in df.columns:
            row[col + "_mean"] = df[col].mean()
            row[col + "_std"]  = df[col].std()
    bl_records.append(row)
df_bl = pd.DataFrame(bl_records)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1: K-sweep β-EO bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, ds in zip(axes, DATASETS):
    sub = df_rl[(df_rl.dataset == ds) & (df_rl.label.isin(K_LABELS))].set_index("label").loc[K_LABELS]
    x = np.arange(len(K_LABELS))
    ax.bar(x, sub["beta_eo_mean"], yerr=sub["beta_eo_std"],
           color=[K_COLORS[k] for k in K_LABELS], capsize=4, width=0.6,
           error_kw={"linewidth": 1.2})
    ax.set_xticks(x); ax.set_xticklabels(K_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
    if ds == "census": ax.set_ylabel("β-EO (mean ± std)")
    ax.set_ylim(0, None)
fig.suptitle("β-EO by sigmoid k — WGL reward sweep (3 seeds, 5000 ep)", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_k_sweep_eo.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig1_k_sweep_eo.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: Lambda-sweep β-EO bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for ax, ds in zip(axes, DATASETS):
    sub = df_rl[(df_rl.dataset == ds) & (df_rl.label.isin(LAM_LABELS))].set_index("label").loc[LAM_LABELS]
    x = np.arange(len(LAM_LABELS))
    ax.bar(x, sub["beta_eo_mean"], yerr=sub["beta_eo_std"],
           color=[LAM_COLORS[l] for l in LAM_LABELS], capsize=4, width=0.5,
           error_kw={"linewidth": 1.2})
    ax.set_xticks(x); ax.set_xticklabels(LAM_LABELS, fontsize=10)
    ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
    if ds == "census": ax.set_ylabel("β-EO (mean ± std)")
    ax.set_ylim(0, None)
fig.suptitle("β-EO by ROC-EO lambda — EO-reward sweep (3 seeds, 5000 ep)", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_lambda_sweep_eo.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig2_lambda_sweep_eo.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3: Best config vs baselines bar chart
# ═══════════════════════════════════════════════════════════════════════════════
BL_ORDER = ["Alpha", "GroupDRO", "SMOTE", "FLB", "OT Repair", "CTGAN", "FairTabDDPM", "FORGE"]
BL_COLORS_MAP = {
    "Alpha": "#BDBDBD", "GroupDRO": "#7986CB", "SMOTE": "#4DB6AC",
    "FLB": "#FFD54F", "OT Repair": "#FF8A65", "CTGAN": "#BA68C8",
    "FairTabDDPM": "#F06292", "FORGE": "#D32F2F",
}
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, ds in zip(axes, DATASETS):
    best_lbl = BEST_CONFIG[ds]
    rl_row   = df_rl[(df_rl.dataset == ds) & (df_rl.label == best_lbl)].iloc[0]
    bl_ds    = df_bl[df_bl.dataset == ds].set_index("method")
    alpha_eo = rl_row["alpha_eo_mean"]
    rows = {"Alpha": (alpha_eo, 0)}
    for m in ["GroupDRO", "SMOTE", "FLB", "OT Repair", "CTGAN", "FairTabDDPM"]:
        if m in bl_ds.index:
            rows[m] = (bl_ds.loc[m, "beta_eo_tpr_diff_mean"], bl_ds.loc[m, "beta_eo_tpr_diff_std"])
    rows["FORGE"] = (rl_row["beta_eo_mean"], rl_row["beta_eo_std"])
    present = [m for m in BL_ORDER if m in rows]
    means = [rows[m][0] for m in present]; stds = [rows[m][1] for m in present]
    x = np.arange(len(present))
    ax.bar(x, means, yerr=stds, color=[BL_COLORS_MAP[m] for m in present],
           capsize=4, width=0.65, error_kw={"linewidth": 1.2},
           edgecolor=["#333" if m == "FORGE" else "none" for m in present],
           linewidth=[1.5 if m == "FORGE" else 0 for m in present])
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=35, ha="right", fontsize=8.5)
    ax.set_title(f"{DATASET_LABELS[ds]}\n(best: {best_lbl})", fontsize=11, fontweight="bold")
    if ds == "census": ax.set_ylabel("β-EO (mean ± std, 3 seeds)")
    ax.set_ylim(0, None)
fig.suptitle("Best RL config vs baselines — β-EO (lower is better)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_best_vs_baselines.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig3_best_vs_baselines.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 FIGURES: copy learning + gen curves from analysis dirs
# ═══════════════════════════════════════════════════════════════════════════════
for ds, cfg in BEST_CONFIG.items():
    run_dir = RUNS[(ds, cfg)]
    analysis = os.path.join(RUNS_DIR, run_dir, "analysis")
    ds_safe = ds.replace("24", "24")
    for fig_name, out_name in [("fig_learning.png", f"fig_learning_{ds}.png"),
                                ("fig_gen_curve.png", f"fig_gen_curve_{ds}.png")]:
        src = os.path.join(analysis, fig_name)
        dst = os.path.join(OUT_DIR, out_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fig_name} -> {out_name}")
        else:
            print(f"WARNING: {src} not found")

# ═══════════════════════════════════════════════════════════════════════════════
# WGL ANALYSIS FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
WGL_RUNS = {
    "Census\n(k=3)":           ("census",    "k=3"),
    "Census\n(no sigmoid)":    ("census",    "no sigmoid"),
    "COMPAS\n(λ=0.5)":         ("compas",    "λ=0.5"),
    "COMPAS\n(no sigmoid)":    ("compas",    "no sigmoid"),
    "Capture-24\n(k=3)":       ("capture24", "k=3"),
    "Capture-24\n(no sigmoid)":("capture24", "no sigmoid"),
}
wgl_data = {}
for label, (ds, rl) in WGL_RUNS.items():
    df = load_metrics(RUNS[(ds, rl)])
    alpha = df["fairness.worst_loss_alpha_baseline"].dropna()
    beta  = df["fairness.worst_loss_beta"].dropna()
    wgl_data[label] = {"alpha": alpha, "diff": alpha - beta, "seed": df["seed"], "ds": ds}

# Paired colours: solid = best config, lighter = no sigmoid
colors_box = [
    DS_COLORS["census"],   "#90CAF9",   # census k=3, no-sig
    DS_COLORS["compas"],   "#F48FB1",   # compas λ=0.5, no-sig
    DS_COLORS["capture24"],"#A5D6A7",   # capture24 k=3, no-sig
]
labels_wgl = list(wgl_data.keys())
seed_dirs  = ["seed_0", "seed_1", "seed_42"]
seed_names = ["seed 0", "seed 1", "seed 42"]
seed_colors = ["#1565C0", "#E53935", "#2E7D32"]

# FIG 4: wgl_alpha pooled (6 boxes: best + no-sigmoid per dataset)
fig, ax = plt.subplots(figsize=(12, 4.5))
bp = ax.boxplot([wgl_data[l]["alpha"].values for l in labels_wgl],
                tick_labels=[l.replace("\n", " ") for l in labels_wgl],
                patch_artist=True, widths=0.55,
                medianprops={"color": "black", "linewidth": 2})
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="BCE = 1.0")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, label="BCE = 0.0")
ax.set_ylabel("Worst-group BCE loss (wgl_alpha)")
ax.set_title("wgl_alpha across all episodes and seeds — all datasets well above 1")
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=9); ax.set_ylim(-0.1, None)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_wgl_alpha_dist.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig4_wgl_alpha_dist.png")

# FIG 5: wgl_alpha per seed — all datasets
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (label, (ds, rl)) in zip(axes, WGL_RUNS.items()):
    per_seed_alphas = []
    for sd in seed_dirs:
        p = os.path.join(RUNS_DIR, RUNS[(ds, rl)], sd, "metrics.csv")
        per_seed_alphas.append(pd.read_csv(p)["fairness.worst_loss_alpha_baseline"].dropna().values)
    bp = ax.boxplot(per_seed_alphas, tick_labels=seed_names, patch_artist=True, widths=0.45,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], seed_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    for i, a in enumerate(per_seed_alphas):
        ax.text(i+1, np.max(a) + 0.03, f"{a.mean():.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_title(label.replace("\n", " "), fontsize=10, fontweight="bold")
    if ds == "census": ax.set_ylabel("wgl_alpha")
fig.suptitle("wgl_alpha per seed — Census and COMPAS stable; Capture-24 seed 1 is 40% lower\n"
             "(wgl_alpha is constant within a seed, reflecting the fixed alpha model for that data split)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_wgl_alpha_per_seed.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig5_wgl_alpha_per_seed.png")

# FIG 6: wgl diff pooled (6 boxes)
fig, ax = plt.subplots(figsize=(12, 4.5))
bp = ax.boxplot([wgl_data[l]["diff"].values for l in labels_wgl],
                tick_labels=[l.replace("\n", " ") for l in labels_wgl],
                patch_artist=True, widths=0.55,
                medianprops={"color": "black", "linewidth": 2})
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="diff = 0 (β = α)")
ax.set_ylabel("wgl_alpha − wgl_beta")
ax.set_title("Reward input (wgl_alpha − wgl_beta) across all episodes and seeds")
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_wgl_diff_dist.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig6_wgl_diff_dist.png")

# FIG 7: wgl diff per seed — 2x3 layout (best config + no sigmoid for each dataset)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (label, (ds, rl)) in zip(axes.flat, WGL_RUNS.items()):
    per_seed_diffs = []
    for sd in seed_dirs:
        p = os.path.join(RUNS_DIR, RUNS[(ds, rl)], sd, "metrics.csv")
        df_s = pd.read_csv(p)
        per_seed_diffs.append((df_s["fairness.worst_loss_alpha_baseline"] -
                               df_s["fairness.worst_loss_beta"]).dropna().values)
    bp = ax.boxplot(per_seed_diffs, tick_labels=seed_names, patch_artist=True, widths=0.45,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], seed_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
    for i, d in enumerate(per_seed_diffs):
        dead = 100 * np.mean(1/(1+np.exp(-3*d)) < 0.5)
        ax.text(i+1, np.percentile(d, 75) + 0.05, f"dead={dead:.0f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(label.replace("\n", " "), fontsize=10, fontweight="bold")
    ax.set_ylabel("wgl_alpha − wgl_beta")
fig.suptitle("Reward input per seed — best config (top row) vs no sigmoid (bottom row)", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig7_wgl_diff_per_seed.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig7_wgl_diff_per_seed.png")

# FIG 8: reward distribution — 2x3 layout (best config uses sigmoid(3×diff); no sigmoid uses norm diff)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
wgl_keys = list(WGL_RUNS.keys())
for ax, (label, (ds, rl)) in zip(axes.flat, WGL_RUNS.items()):
    diff  = wgl_data[label]["diff"].values
    alpha = wgl_data[label]["alpha"].values
    is_nosig = "no sigmoid" in rl
    color = DS_COLORS[ds] if not is_nosig else colors_box[wgl_keys.index(label)]
    if is_nosig:
        vals   = diff / alpha
        xlabel = "norm reward (diff / wgl_alpha)"
        vline  = 0.0
        annot  = f"Mean: {vals.mean():.3f}\nStd: {vals.std():.3f}"
    else:
        vals   = 1 / (1 + np.exp(-3.0 * diff))
        xlabel = "sigmoid(3 × diff)"
        vline  = 0.5
        annot  = f"Dead: {100*np.mean(vals<0.5):.1f}%\nMean: {vals.mean():.3f}"
    ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(vline, color="red", linestyle="--", linewidth=1.2)
    ax.text(0.03, 0.95, annot, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_title(label.replace("\n", " "), fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode count")
fig.suptitle("Reward distribution — best config uses sigmoid(3×diff) (top); no sigmoid uses norm diff (bottom)\n"
             "Red line = 0.5 (sigmoid) or 0.0 (norm diff)", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig8_sigmoid_dist.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig8_sigmoid_dist.png")

print(f"\nAll figures saved to {OUT_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# Markdown table builders
# ═══════════════════════════════════════════════════════════════════════════════
SWEEP_HDR = ("| Config | β-EO | EO-Δ | β-EOd | β-DP | β-F1w | β-AUC |\n"
             "|--------|------|------|-------|------|-------|-------|\n")

def make_sweep_row(label, r, star=False):
    tag = " ★" if star else ""
    return (f"| {label}{tag} | {fmt(r.beta_eo_mean, r.beta_eo_std)} | "
            f"{fmt(r.eo_delta_mean)} | {fmt(r.beta_eod_mean, r.beta_eod_std)} | "
            f"{fmt(r.beta_dp_mean, r.beta_dp_std)} | "
            f"{fmt(r.beta_f1w_mean, r.beta_f1w_std)} | {fmt(r.beta_auc_mean)} |")

def make_k_table(ds):
    rows = []
    for lbl in K_LABELS:
        r = df_rl[(df_rl.dataset == ds) & (df_rl.label == lbl)]
        if r.empty: continue
        rows.append(make_sweep_row(lbl, r.iloc[0], star=(lbl == BEST_CONFIG[ds])))
    return SWEEP_HDR + "\n".join(rows)

def make_lam_table(ds):
    rows = []
    for lbl in LAM_LABELS:
        r = df_rl[(df_rl.dataset == ds) & (df_rl.label == lbl)]
        if r.empty: continue
        rows.append(make_sweep_row(lbl, r.iloc[0], star=(lbl == BEST_CONFIG[ds])))
    return SWEEP_HDR.replace("Config", "λ (ROC-EO)") + "\n".join(rows)

def make_per_seed_table(ds, lbl):
    r = df_rl[(df_rl.dataset == ds) & (df_rl.label == lbl)]
    if r.empty: return ""
    rows = []
    for s in r.iloc[0]["per_seed"]:
        rows.append(f"| {s['seed']} | {s['alpha_eo']:.4f} | {s['beta_eo']:.4f} | "
                    f"{s['eo_delta']:.4f} | {s['beta_f1w']:.4f} |")
    return ("| Seed | α-EO | β-EO | EO-Δ | β-F1w |\n"
            "|------|------|------|------|-------|\n" + "\n".join(rows))

def make_baseline_table(ds):
    best_lbl = BEST_CONFIG[ds]
    rl_row   = df_rl[(df_rl.dataset == ds) & (df_rl.label == best_lbl)].iloc[0]
    bl_ds    = df_bl[df_bl.dataset == ds].set_index("method")
    alpha_eo = rl_row["alpha_eo_mean"]
    alpha_dp = rl_row["alpha_dp_mean"]
    alpha_eod= rl_row["alpha_eod_mean"]

    hdr = ("| Method | β-EO | EO-Δ | β-EOd | β-DP | β-F1w | β-AUC |\n"
           "|--------|------|------|-------|------|-------|-------|\n")
    rows = [f"| Alpha (no intervention) | {alpha_eo:.3f} | — | {alpha_eod:.3f} | {alpha_dp:.3f} | — | — |"]
    for m in ["GroupDRO", "SMOTE", "FLB", "OT Repair", "CTGAN", "FairTabDDPM"]:
        if m not in bl_ds.index: continue
        b = bl_ds.loc[m]
        delta = b["beta_eo_tpr_diff_mean"] - alpha_eo
        rows.append(
            f"| {m} | {fmt(b['beta_eo_tpr_diff_mean'], b['beta_eo_tpr_diff_std'])} | "
            f"{delta:+.3f} | {fmt(b.get('beta_eod_max_diff_mean', np.nan), b.get('beta_eod_max_diff_std', np.nan))} | "
            f"{fmt(b.get('beta_dp_diff_mean', np.nan), b.get('beta_dp_diff_std', np.nan))} | "
            f"{b['beta_f1_weighted_mean']:.3f} | {b['beta_roc_auc_mean']:.3f} |"
        )
    rows.append(
        f"| **FORGE ({best_lbl})** | **{fmt(rl_row.beta_eo_mean, rl_row.beta_eo_std)}** | "
        f"**{rl_row.eo_delta_mean:+.3f}** | **{fmt(rl_row.beta_eod_mean, rl_row.beta_eod_std)}** | "
        f"**{fmt(rl_row.beta_dp_mean, rl_row.beta_dp_std)}** | "
        f"**{rl_row.beta_f1w_mean:.3f}** | **{rl_row.beta_auc_mean:.3f}** |"
    )
    return hdr + "\n".join(rows)

def make_wgl_stats_table():
    rows = []
    for label, (ds, rl) in WGL_RUNS.items():
        df    = load_metrics(RUNS[(ds, rl)])
        alpha = df["fairness.worst_loss_alpha_baseline"].dropna()
        diff  = (alpha - df["fairness.worst_loss_beta"]).dropna()
        lbl   = label.replace("\n", " ")
        is_nosig = "no sigmoid" in rl
        if is_nosig:
            norm_reward = (diff / alpha).mean()
            sig_str = "N/A"
            reward_str = f"{norm_reward:.3f}"
        else:
            sig_str = f"{(1 / (1 + np.exp(-3.0 * diff))).mean():.3f}"
            reward_str = "N/A"
        rows.append(f"| {lbl} | {alpha.mean():.3f} | [{alpha.min():.3f}, {alpha.max():.3f}] | "
                    f"{diff.mean():.3f} ± {diff.std():.3f} | [{diff.min():.3f}, {diff.max():.3f}] | "
                    f"{sig_str} | {reward_str} |")
    return ("| Run | wgl_alpha mean | wgl_alpha range | diff mean±std | diff range | sigmoid(3×diff) mean | norm reward mean |\n"
            "|-----|---------------|----------------|--------------|-----------|---------------------|----------------|\n"
            + "\n".join(rows))

def make_capture24_per_seed_wgl_table():
    rows = []
    for sd, sname in zip(seed_dirs, [0, 1, 42]):
        p = os.path.join(RUNS_DIR, RUNS[("capture24", "k=3")], sd, "metrics.csv")
        df_s = pd.read_csv(p)
        alpha = df_s["fairness.worst_loss_alpha_baseline"].dropna()
        diff  = (alpha - df_s["fairness.worst_loss_beta"]).dropna()
        sig   = 1 / (1 + np.exp(-3.0 * diff))
        rows.append(f"| {sname} | {alpha.mean():.3f} (constant) | "
                    f"{diff.mean():.3f} ± {diff.std():.3f} | "
                    f"{sig.mean():.3f} |")
    return ("| Seed | wgl_alpha | diff mean±std | sigmoid(3×diff) mean |\n"
            "|------|-----------|--------------|---------------------|\n" + "\n".join(rows))

# ═══════════════════════════════════════════════════════════════════════════════
# Write report
# ═══════════════════════════════════════════════════════════════════════════════
report = f"""# Reward Sweep Comparison Report

All runs: 5000 episodes, 3 seeds (0, 1, 42), real_data_size=3000.
"no sigmoid" = k=0 (normalised delta reward). ★ = best config for that dataset.

---

## 1. WGL Sigmoid k Sweep

### 1.1 Census

{make_k_table("census")}

![k sweep EO](fig1_k_sweep_eo.png)

#### Per-seed detail — Census k=3

{make_per_seed_table("census", "k=3")}

---

### 1.2 COMPAS

{make_k_table("compas")}

#### Per-seed detail — COMPAS k=3

{make_per_seed_table("compas", "k=3")}

---

### 1.3 Capture-24

{make_k_table("capture24")}

#### Per-seed detail — Capture-24 k=3

{make_per_seed_table("capture24", "k=3")}

---

## 2. ROC-EO Lambda Sweep

### 2.1 Census

{make_lam_table("census")}

#### Per-seed detail — Census λ=0.7

{make_per_seed_table("census", "λ=0.7")}

---

### 2.2 COMPAS

{make_lam_table("compas")}

#### Per-seed detail — COMPAS λ=0.5

{make_per_seed_table("compas", "λ=0.5")}

---

### 2.3 Capture-24

{make_lam_table("capture24")}

![lambda sweep EO](fig2_lambda_sweep_eo.png)

---

## 3. Best Config vs Baselines

Best config per dataset: Census → k=3, COMPAS → λ=0.5, Capture-24 → k=3. Alpha = no-intervention baseline.

![best vs baselines](fig3_best_vs_baselines.png)

### 3.1 Census (best: k=3)

{make_baseline_table("census")}

### 3.2 COMPAS (best: λ=0.5)

{make_baseline_table("compas")}

### 3.3 Capture-24 (best: k=3)

{make_baseline_table("capture24")}

---

## 4. Learning and Generalization Curves — Best Config per Dataset

### 4.1 Census (k=3) — Learning curve

Episode return and validation EO per seed across training.

![Census learning curve](fig_learning_census.png)

### 4.2 Census (k=3) — Generalization curve (EO gap)

Test-set EO gap at snapshot intervals (every 150 episodes).

![Census gen curve](fig_gen_curve_census.png)

---

### 4.3 COMPAS (λ=0.5) — Learning curve

![COMPAS learning curve](fig_learning_compas.png)

### 4.4 COMPAS (λ=0.5) — Generalization curve (EO gap)

![COMPAS gen curve](fig_gen_curve_compas.png)

---

### 4.5 Capture-24 (k=3) — Learning curve

![Capture-24 learning curve](fig_learning_capture24.png)

### 4.6 Capture-24 (k=3) — Generalization curve (EO gap)

![Capture-24 gen curve](fig_gen_curve_capture24.png)

---

## 5. WGL Reward Scale Analysis

Addressing the question of whether the reward input scale (wgl_alpha − wgl_beta) differs materially across datasets and could explain the performance gap.

### 5.1 wgl_alpha is not bounded in [0, 1] on any dataset

BCE loss is unbounded; all three datasets have wgl_alpha well above 1.0. The reward input (diff mean) is the raw, unnormalized difference wgl_alpha − wgl_beta averaged across all training episodes and seeds. Sigmoid mean is the mean value of sigmoid(k × diff) across all episodes and seeds; a value near 1.0 means the agent was consistently improving the worst-group loss over alpha, while a value near 0.5 means near-zero improvement.

{make_wgl_stats_table()}

![wgl_alpha pooled](fig4_wgl_alpha_dist.png)

### 5.2 Distribution of reward input (wgl_alpha − wgl_beta) — pooled across all seeds

{make_capture24_per_seed_wgl_table()}

![wgl diff pooled](fig6_wgl_diff_dist.png)

### 5.3 Sigmoid(k=3) reward value distribution — all episodes, all seeds

The sigmoid distribution shows how often the agent receives an informative reward signal. Census and COMPAS cluster tightly near 1.0, indicating the agent reliably improves wgl on most episodes. Capture-24 is bimodal: seeds 0 and 42 cluster near 1.0 (strong signal), but seed 1 clusters near 0.5 (near-zero diff, uninformative reward). This per-seed divergence in Capture-24 traces back to wgl_alpha varying 40% across seeds due to data-split instability — seed 1 produces an alpha model that is already nearly optimal on the worst group, leaving little room for beta to improve.

![sigmoid distribution](fig8_sigmoid_dist.png)
"""

report_path = f"{OUT_DIR}/sweep_report.md"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport saved to {report_path}")

# ── Section 5 per-seed CSV export ────────────────────────────────────────────
sec5_rows = []
for label, (ds, rl) in WGL_RUNS.items():
    ds_label = DATASET_LABELS[ds]
    run_dir  = RUNS[(ds, rl)]
    for sd in SEEDS:
        p = os.path.join(RUNS_DIR, run_dir, sd, "metrics.csv")
        if not os.path.exists(p):
            continue
        df_s  = pd.read_csv(p)
        alpha = df_s["fairness.worst_loss_alpha_baseline"].dropna()
        beta  = df_s["fairness.worst_loss_beta"].dropna()
        diff  = alpha - beta
        sig   = 1 / (1 + np.exp(-3.0 * diff))
        ep    = df_s.loc[alpha.index, "episode"] if "episode" in df_s.columns else alpha.index
        seed_num = int(sd.split("_")[1])
        is_nosig = "no sigmoid" in rl
        for i, idx in enumerate(alpha.index):
            a_val = float(alpha.iloc[i])
            d_val = float(diff.iloc[i])
            sec5_rows.append({
                "dataset":      ds_label,
                "config":       rl,
                "seed":         seed_num,
                "episode":      int(ep.iloc[i]) if hasattr(ep, "iloc") else idx,
                "wgl_alpha":    a_val,
                "wgl_beta":     float(beta.iloc[i]),
                "diff":         d_val,
                "sigmoid_k3":   float("nan") if is_nosig else 1 / (1 + np.exp(-3.0 * d_val)),
                "norm_reward":  d_val / a_val if is_nosig else float("nan"),
            })

csv_path = f"{OUT_DIR}/section5_per_seed.csv"
pd.DataFrame(sec5_rows).to_csv(csv_path, index=False)
print(f"Section 5 CSV saved to {csv_path}")
