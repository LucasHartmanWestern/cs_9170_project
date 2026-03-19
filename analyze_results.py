#!/usr/bin/env python3
"""
analyze_results.py — Automated results analysis and EXPERIMENTS.md updater.

Usage:
    python analyze_results.py \\
        --runs training_runs/SPECv17c_... training_runs/SPECv17c_... \\
        --label "v17c bias=0.10 ds=0.20" \\
        [--out plots/v17c_bias010_ds020/] \\
        [--experiments-md EXPERIMENTS.md] \\
        [--no-append]   # dry-run: print section, don't write to EXPERIMENTS.md

Produces:
    plots/<label_slug>/
        deadzone.png        — global_obj < 0.5 fraction over training per seed
        eo_comparison.png   — EO bar chart vs all known baselines
        utility.png         — F1w / AUC bar chart vs baselines
        per_seed_eo.png     — per-seed EO strip plot

Appends a dated section to EXPERIMENTS.md with tables + Claude's interpretation.
"""

import argparse
import json
import re
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded baselines (census_income only; extend as new datasets are added)
# Keys: (dataset, bias_pct)  →  list of dicts with keys: label, eo, eo_std, f1w, auc
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_BASELINES = {
    ("census_income", 0.10): [
        {"label": "Alpha (no aug)", "eo": 0.144, "eo_std": None, "f1w": 0.821, "auc": 0.891},
        {"label": "GroupDRO",       "eo": 0.115, "eo_std": 0.024, "f1w": 0.821, "auc": 0.891},
        {"label": "OT Repair",      "eo": 0.025, "eo_std": 0.020, "f1w": 0.788, "auc": 0.819},
        {"label": "CTGAN",          "eo": 0.075, "eo_std": 0.027, "f1w": 0.789, "auc": 0.842},
        {"label": "RL v16",         "eo": 0.084, "eo_std": 0.043, "f1w": 0.788, "auc": 0.848},
        {"label": "RL v17a",        "eo": 0.071, "eo_std": 0.067, "f1w": 0.807, "auc": 0.867},
    ],
    ("census_income", 0.05): [
        {"label": "Alpha (no aug)", "eo": 0.075, "eo_std": None, "f1w": 0.810, "auc": 0.875},
        {"label": "GroupDRO",       "eo": 0.247, "eo_std": 0.120, "f1w": 0.807, "auc": 0.878},
        {"label": "OT Repair",      "eo": 0.050, "eo_std": 0.038, "f1w": 0.788, "auc": 0.819},
        {"label": "CTGAN",          "eo": 0.020, "eo_std": 0.010, "f1w": 0.756, "auc": 0.812},
        {"label": "RL v16",         "eo": 0.097, "eo_std": 0.067, "f1w": 0.787, "auc": 0.850},
        {"label": "RL v17a",        "eo": 0.071, "eo_std": 0.051, "f1w": 0.771, "auc": 0.860},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_seed_results(seed_dir: Path) -> dict | None:
    """Load test_results.json + meta.json from a single seed directory."""
    results_path = seed_dir / "test_results.json"
    meta_path    = seed_dir / "meta.json"
    if not results_path.exists():
        return None
    r = json.loads(results_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    r["_meta"] = meta
    r["_seed_dir"] = str(seed_dir)
    return r


def load_run(run_dir: Path) -> list[dict]:
    """Return list of per-seed result dicts for a run directory."""
    run_dir = Path(run_dir)
    seed_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    results = []
    for sd in seed_dirs:
        r = load_seed_results(sd)
        if r is not None:
            results.append(r)
    return results


def load_metrics_csv(run_dir: Path) -> dict[int, pd.DataFrame]:
    """Return {seed_id: metrics_df} for all seeds in a run directory."""
    out = {}
    for sd in sorted(Path(run_dir).iterdir()):
        if not (sd.is_dir() and sd.name.startswith("seed_")):
            continue
        mc = sd / "metrics.csv"
        if mc.exists():
            try:
                seed_id = int(sd.name.split("_")[1])
            except (IndexError, ValueError):
                seed_id = -1
            out[seed_id] = pd.read_csv(mc)
    return out


def infer_bias_pct(results: list[dict]) -> float | None:
    """Try to read bias_pct from meta.json embedded in results."""
    for r in results:
        bp = r.get("_meta", {}).get("BIAS_PCT") or r.get("_meta", {}).get("bias_pct")
        if bp is not None:
            return float(bp)
    return None


def infer_dataset(results: list[dict]) -> str:
    for r in results:
        ds = r.get("_meta", {}).get("dataset_name") or r.get("dataset")
        if ds:
            return ds
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_beta_metrics(r: dict) -> dict:
    return {
        "eo":  r.get("beta_eo_tpr_diff"),
        "f1w": r.get("beta_f1_weighted"),
        "auc": r.get("beta_roc_auc"),
        "f1m": r.get("beta_f1_minority"),
        "brier": r.get("beta_brier"),
    }


def get_alpha_metrics(r: dict) -> dict:
    return {
        "eo":  r.get("alpha_eo_tpr_diff"),
        "f1w": r.get("alpha_f1_weighted"),
        "auc": r.get("alpha_roc_auc"),
    }


def aggregate(values: list[float | None]) -> tuple[float, float]:
    """Return (mean, std) ignoring None."""
    v = [x for x in values if x is not None]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def deadzone_fraction(df: pd.DataFrame, phase: int = 1) -> float:
    """Fraction of Phase-1 episodes with global_obj < 0.5."""
    col = "global.global_obj"
    if col not in df.columns:
        return float("nan")
    phase_col = "global.phase"
    if phase_col in df.columns:
        df = df[df[phase_col] == phase]
    vals = df[col].dropna()
    if len(vals) == 0:
        return float("nan")
    return float((vals < 0.5).mean())


def escape_epoch(df: pd.DataFrame) -> int | None:
    """First episode where global_obj crosses 0.5."""
    col = "global.global_obj"
    ep_col = "episode"
    if col not in df.columns or ep_col not in df.columns:
        return None
    above = df[df[col] >= 0.5]
    if above.empty:
        return None
    return int(above[ep_col].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "ours":    "#2077B4",
    "ctgan":   "#FF7F0E",
    "otrepair":"#2CA02C",
    "gdro":    "#D62728",
    "v16":     "#9467BD",
    "v17a":    "#8C564B",
    "alpha":   "#7F7F7F",
}

def _color_for_label(label: str) -> str:
    lo = label.lower()
    if "alpha" in lo or "no aug" in lo: return COLORS["alpha"]
    if "groupdro" in lo or "group dro" in lo: return COLORS["gdro"]
    if "ot repair" in lo or "otrepair" in lo: return COLORS["otrepair"]
    if "ctgan" in lo: return COLORS["ctgan"]
    if "v16" in lo: return COLORS["v16"]
    if "v17a" in lo: return COLORS["v17a"]
    return COLORS["ours"]


def plot_deadzone(metrics_by_seed: dict[int, pd.DataFrame], run_label: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.tab10.colors
    for i, (seed_id, df) in enumerate(sorted(metrics_by_seed.items())):
        col = "global.global_obj"
        if col not in df.columns:
            continue
        # Phase 1 only
        phase_col = "global.phase"
        if phase_col in df.columns:
            df_p1 = df[df[phase_col] == 1].copy()
        else:
            df_p1 = df[df["episode"] <= 800].copy() if "episode" in df.columns else df.copy()

        ep  = df_p1["episode"].values if "episode" in df_p1.columns else np.arange(len(df_p1))
        obj = df_p1[col].values
        ax.plot(ep, obj, lw=0.9, alpha=0.85, label=f"seed {seed_id}", color=colors[i % 10])
        ax.fill_between(ep, obj, 0.5, where=obj < 0.5, alpha=0.12, color=colors[i % 10])

    ax.axhline(0.5, color="black", lw=1.2, ls="--", label="Deadzone threshold (0.5)")
    ax.set_xlabel("Episode (Phase 1)")
    ax.set_ylabel("global_obj  [sigmoid(β better than α)]")
    ax.set_title(f"Global objective over training — {run_label}\n"
                 "Shaded region = deadzone (agent gets no positive learning signal)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_eo_comparison(our_eos: list[float], baselines: list[dict], run_label: str, out_path: Path):
    our_mean, our_std = aggregate(our_eos)

    all_methods = baselines + [{"label": run_label, "eo": our_mean, "eo_std": our_std}]
    labels = [m["label"] for m in all_methods]
    means  = [m["eo"] for m in all_methods]
    stds   = [m.get("eo_std") or 0 for m in all_methods]
    colors = [_color_for_label(l) for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.axhline(means[0], color=colors[0], lw=1.0, ls=":", alpha=0.6)  # alpha line for reference
    ax.set_ylabel("EO gap (Equal Opportunity, lower = fairer)")
    ax.set_title(f"Equal Opportunity gap — {run_label}\n(error bars = ±std across seeds; lower is better)")
    ax.set_ylim(0, max(means) * 1.35 if max(means) > 0 else 0.3)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_utility(our_results: list[dict], baselines: list[dict], run_label: str, out_path: Path):
    our_f1ws = [get_beta_metrics(r)["f1w"] for r in our_results if get_beta_metrics(r)["f1w"] is not None]
    our_aucs  = [get_beta_metrics(r)["auc"]  for r in our_results if get_beta_metrics(r)["auc"]  is not None]
    our_f1w_m, our_f1w_s = aggregate(our_f1ws)
    our_auc_m, our_auc_s = aggregate(our_aucs)

    all_methods = baselines + [{"label": run_label, "f1w": our_f1w_m, "auc": our_auc_m}]
    labels = [m["label"] for m in all_methods]
    f1ws   = [m.get("f1w") or 0 for m in all_methods]
    aucs   = [m.get("auc")  or 0 for m in all_methods]
    colors = [_color_for_label(l) for l in labels]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5))
    ax.bar(x - width/2, f1ws, width, label="F1-weighted", color=colors, alpha=0.80, edgecolor="black", lw=0.7)
    ax.bar(x + width/2, aucs, width, label="AUC",         color=colors, alpha=0.45, edgecolor="black", lw=0.7, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score (higher = better utility)")
    ax.set_ylim(0.65, min(1.02, max(max(f1ws), max(aucs)) * 1.06))
    ax.set_title(f"Utility metrics — {run_label}\nSolid = F1-weighted, hatched = AUC")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    # Value labels
    for xi, (f, a) in enumerate(zip(f1ws, aucs)):
        ax.text(xi - width/2, f + 0.002, f"{f:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + width/2, a + 0.002, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_per_seed_eo(our_results: list[dict], baselines: list[dict], run_label: str, out_path: Path):
    """Strip plot of per-seed beta EO, with reference lines for key baselines."""
    eos = [get_beta_metrics(r)["eo"] for r in our_results if get_beta_metrics(r)["eo"] is not None]
    alpha_eos = [get_alpha_metrics(r)["eo"] for r in our_results if get_alpha_metrics(r)["eo"] is not None]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Reference lines from baselines
    ref_colors = ["#D62728", "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B", "#7F7F7F"]
    for j, bl in enumerate(baselines):
        ax.axhline(bl["eo"], color=ref_colors[j % len(ref_colors)], lw=1.2, ls="--", alpha=0.7,
                   label=f"{bl['label']} ({bl['eo']:.3f})")

    # Alpha per-seed
    for i, aeo in enumerate(alpha_eos):
        ax.scatter(0.8 + i * 0.15, aeo, marker="^", s=90, color="#7F7F7F", zorder=5, alpha=0.8)

    # Beta per-seed
    for i, eo in enumerate(eos):
        seed_id = our_results[i].get("seed", i)
        ax.scatter(1 + i * 0.1, eo, marker="o", s=100, color="#2077B4", zorder=5)
        ax.annotate(f"s{seed_id}", (1 + i * 0.1, eo), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    mean_eo, std_eo = aggregate(eos)
    ax.axhline(mean_eo, color="#2077B4", lw=2.0, ls="-", label=f"Our mean ({mean_eo:.3f} ±{std_eo:.3f})")
    ax.fill_between([0.7, 1 + len(eos) * 0.1 + 0.1], mean_eo - std_eo, mean_eo + std_eo,
                    alpha=0.12, color="#2077B4")

    ax.set_xlim(0.6, 1 + len(eos) * 0.1 + 0.3)
    ax.set_ylabel("EO gap (lower = fairer)")
    ax.set_title(f"Per-seed EO gap — {run_label}\n"
                 "Blue circles = beta (our method)  |  Grey triangles = alpha baseline  |  "
                 "Dashed = published baselines")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Automatic interpretation
# ─────────────────────────────────────────────────────────────────────────────

def build_interpretation(
    our_results: list[dict],
    metrics_by_seed: dict,
    run_label: str,
    baselines: list[dict],
    dataset: str,
    bias_pct: float | None,
) -> str:
    """Generate Claude's interpretation of the run results."""
    eos   = [get_beta_metrics(r)["eo"]  for r in our_results if get_beta_metrics(r)["eo"]  is not None]
    f1ws  = [get_beta_metrics(r)["f1w"] for r in our_results if get_beta_metrics(r)["f1w"] is not None]
    aucs  = [get_beta_metrics(r)["auc"] for r in our_results if get_beta_metrics(r)["auc"] is not None]
    aeos  = [get_alpha_metrics(r)["eo"] for r in our_results if get_alpha_metrics(r)["eo"] is not None]

    eo_mean, eo_std = aggregate(eos)
    f1w_mean, _ = aggregate(f1ws)
    auc_mean, _ = aggregate(aucs)
    aeo_mean, _ = aggregate(aeos)

    n_seeds = len(eos)
    n_improve = sum(1 for eo, aeo in zip(eos, aeos) if eo < aeo)
    rogue_count = n_seeds - n_improve

    # Deadzone stats
    dz_fracs = [deadzone_fraction(df) for df in metrics_by_seed.values()]
    dz_fracs = [x for x in dz_fracs if not np.isnan(x)]
    mean_dz = np.mean(dz_fracs) if dz_fracs else float("nan")

    # Find reference baselines
    def bl(label_substr):
        for b in baselines:
            if label_substr.lower() in b["label"].lower():
                return b
        return None

    alpha_bl  = bl("alpha") or bl("no aug")
    ctgan_bl  = bl("ctgan")
    v17a_bl   = bl("v17a")
    otrepair_bl = bl("ot repair") or bl("otrepair")

    lines = []

    # Overall summary
    delta_eo = eo_mean - aeo_mean
    if delta_eo < -0.01:
        verdict = f"EO improves by {abs(delta_eo):.3f} vs alpha (mean {eo_mean:.3f} ← {aeo_mean:.3f})"
    elif abs(delta_eo) <= 0.01:
        verdict = f"EO roughly unchanged vs alpha (mean {eo_mean:.3f} vs {aeo_mean:.3f})"
    else:
        verdict = f"⚠️ EO degrades by {delta_eo:.3f} vs alpha (mean {eo_mean:.3f} vs {aeo_mean:.3f})"

    lines.append(f"**Summary:** {n_seeds}-seed run. {verdict}. "
                 f"F1w={f1w_mean:.3f}, AUC={auc_mean:.3f}. "
                 f"{n_improve}/{n_seeds} seeds improve EO, {rogue_count} rogue.")

    # Deadzone
    if not np.isnan(mean_dz):
        dz_pct = mean_dz * 100
        if dz_pct < 10:
            lines.append(f"**Deadzone:** {dz_pct:.1f}% — excellent, well below 20% threshold. "
                         "gamma=1.0 is fully effective; agent gets positive signal from early episodes.")
        elif dz_pct < 20:
            lines.append(f"**Deadzone:** {dz_pct:.1f}% — acceptable. "
                         "Mild residual deadzone but below the 20% concern threshold.")
        elif dz_pct < 40:
            lines.append(f"**Deadzone:** {dz_pct:.1f}% — elevated. "
                         "A significant fraction of Phase-1 episodes produce no learning signal. "
                         "Check whether beta's training loss is collapsing on certain seeds.")
        else:
            lines.append(f"**Deadzone:** {dz_pct:.1f}% — 🔴 severe. "
                         "Majority of Phase-1 episodes waste the policy gradient. "
                         "Something is wrong with credit assignment or beta training.")

    # Comparison to v17a reference
    if v17a_bl and not np.isnan(eo_mean):
        diff = eo_mean - v17a_bl["eo"]
        if diff < -0.005:
            lines.append(f"**vs v17a baseline:** EO {abs(diff):.3f} better than v17a ({v17a_bl['eo']:.3f}). "
                         "This config is an improvement — consider as new best.")
        elif abs(diff) <= 0.005:
            lines.append(f"**vs v17a baseline:** EO essentially identical to v17a ({v17a_bl['eo']:.3f}). "
                         "Not a regression; v17a advantages (F1w, AUC) still apply.")
        else:
            lines.append(f"**vs v17a baseline:** EO {diff:.3f} worse than v17a ({v17a_bl['eo']:.3f}). "
                         "This config does not improve on v17a. Check for confound or hyperparameter sensitivity.")

    # Comparison to CTGAN
    if ctgan_bl and not np.isnan(eo_mean):
        diff = eo_mean - ctgan_bl["eo"]
        if diff < 0:
            lines.append(f"**vs CTGAN:** EO {abs(diff):.3f} better than CTGAN ({ctgan_bl['eo']:.3f}). "
                         "Fairness advantage strengthened. Good for competitive performance claim.")
        elif abs(diff) <= 0.01:
            lines.append(f"**vs CTGAN:** EO roughly matched ({eo_mean:.3f} vs {ctgan_bl['eo']:.3f}). "
                         f"Utility comparison is the differentiator: F1w {f1w_mean:.3f} vs CTGAN {ctgan_bl['f1w']:.3f}.")
        else:
            lines.append(f"**vs CTGAN:** EO {diff:.3f} worse than CTGAN ({ctgan_bl['eo']:.3f}). "
                         "Fairness claim relative to CTGAN weakens here; investigate whether "
                         "the delta_scale change hurts utility-fairness balance.")

    # Rogue seed analysis
    if rogue_count > 0:
        rogue_seeds = [our_results[i].get("seed", i) for i, (eo, aeo) in enumerate(zip(eos, aeos)) if eo >= aeo]
        rogue_eos   = [eos[i] for i, (eo, aeo) in enumerate(zip(eos, aeos)) if eo >= aeo]
        lines.append(f"**Rogue seeds:** {rogue_seeds} (EO: {[f'{e:.3f}' for e in rogue_eos]}). "
                     "Seed-level instability persists. Before the 5-seed final run, check whether "
                     "rogue seeds correlate with low alpha-EO (little room to improve) or "
                     "high deadzone fraction on that seed.")

    # Utility comment
    if alpha_bl and not np.isnan(f1w_mean):
        f1w_delta = f1w_mean - alpha_bl["f1w"]
        if f1w_delta > 0:
            lines.append(f"**Utility:** F1w {f1w_mean:.3f} (+{f1w_delta:.3f} vs alpha). "
                         "Utility preserved and slightly improved — strong paper story.")
        elif f1w_delta > -0.015:
            lines.append(f"**Utility:** F1w {f1w_mean:.3f} ({f1w_delta:.3f} vs alpha). "
                         "Utility well-preserved within noise bounds.")
        else:
            lines.append(f"**Utility:** F1w {f1w_mean:.3f} ({f1w_delta:.3f} vs alpha). "
                         "⚠️ Noticeable utility drop. Check whether the delta_scale increase "
                         "is generating samples that hurt beta's overall classification ability.")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS.md section builder
# ─────────────────────────────────────────────────────────────────────────────

def build_experiments_section(
    our_results: list[dict],
    metrics_by_seed: dict,
    run_label: str,
    baselines: list[dict],
    run_dirs: list[Path],
    out_dir: Path,
    dataset: str,
    bias_pct: float | None,
) -> str:
    today = date.today().isoformat()

    eos   = [get_beta_metrics(r)["eo"]  for r in our_results if get_beta_metrics(r)["eo"]  is not None]
    f1ws  = [get_beta_metrics(r)["f1w"] for r in our_results if get_beta_metrics(r)["f1w"] is not None]
    aucs  = [get_beta_metrics(r)["auc"] for r in our_results if get_beta_metrics(r)["auc"] is not None]
    aeos  = [get_alpha_metrics(r)["eo"] for r in our_results if get_alpha_metrics(r)["eo"] is not None]

    eo_mean, eo_std = aggregate(eos)
    f1w_mean, f1w_std = aggregate(f1ws)
    auc_mean, auc_std = aggregate(aucs)
    aeo_mean, _ = aggregate(aeos)

    dz_fracs = [deadzone_fraction(df) for df in metrics_by_seed.values()]
    mean_dz = np.nanmean(dz_fracs) if dz_fracs else float("nan")

    # Per-seed table
    seed_rows = []
    for r in our_results:
        bm = get_beta_metrics(r)
        am = get_alpha_metrics(r)
        seed = r.get("seed", "?")
        dz = float("nan")
        try:
            sidx = int(str(seed))
            if sidx in metrics_by_seed:
                dz = deadzone_fraction(metrics_by_seed[sidx])
        except (ValueError, TypeError):
            pass
        delta = (bm["eo"] - am["eo"]) if (bm["eo"] is not None and am["eo"] is not None) else float("nan")
        flag = "✅" if delta < -0.005 else ("🔴" if delta > 0.005 else "~")
        seed_rows.append(
            f"| {seed} | {am['eo']:.3f} | {bm['eo']:.3f} | {delta:+.3f} {flag} | "
            f"{bm['f1w']:.3f} | {bm['auc']:.3f} | {dz*100:.1f}% |"
        )

    per_seed_table = "\n".join([
        "| seed | α-EO | β-EO | Δ-EO | β-F1w | β-AUC | dead% |",
        "|---|---|---|---|---|---|---|",
    ] + seed_rows)

    # Summary aggregate row
    summary_row = (
        f"| **{run_label}** | {aeo_mean:.3f} | **{eo_mean:.3f}** | ±{eo_std:.3f} | "
        f"**{f1w_mean:.3f}** | **{auc_mean:.3f}** | {mean_dz*100:.1f}% |"
    )

    # Baseline comparison table (append our row)
    bl_rows = []
    for bl in baselines:
        std_str = f"±{bl['eo_std']:.3f}" if bl.get("eo_std") is not None else "—"
        bl_rows.append(f"| {bl['label']} | — | {bl['eo']:.3f} | {std_str} | {bl.get('f1w', '—'):.3f} | {bl.get('auc', '—'):.3f} | — |")
    bl_rows.append(summary_row)

    baseline_table = "\n".join([
        "| Method | α-EO | β-EO | ±std | β-F1w | β-AUC | dead% |",
        "|---|---|---|---|---|---|---|",
    ] + bl_rows)

    # Plot links
    rel_plots = [
        f"  - `{out_dir / 'deadzone.png'}`",
        f"  - `{out_dir / 'eo_comparison.png'}`",
        f"  - `{out_dir / 'utility.png'}`",
        f"  - `{out_dir / 'per_seed_eo.png'}`",
    ]

    interpretation = build_interpretation(
        our_results, metrics_by_seed, run_label, baselines, dataset, bias_pct
    )

    run_dir_list = "\n".join(f"  - `{rd}`" for rd in run_dirs)

    bias_str = f"{bias_pct:.2f}" if bias_pct is not None else "unknown"

    section = textwrap.dedent(f"""
    ---

    ### {run_label} ({today})

    **Run dirs:**
    {run_dir_list}

    **Dataset:** {dataset}  |  **bias_pct:** {bias_str}  |  **Seeds:** {len(our_results)}

    #### Per-seed results

    {per_seed_table}

    #### Comparison vs baselines

    {baseline_table}

    #### Plots

    {"".join(chr(10) + l for l in rel_plots)}

    #### Analysis

    {interpretation}

    ---
    """).strip()

    return "\n\n" + section + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s).strip("_")


def main():
    parser = argparse.ArgumentParser(description="Analyze RL experiment results and update EXPERIMENTS.md")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more run directories (each contains seed_N subdirs)")
    parser.add_argument("--label", required=True,
                        help='Human-readable label, e.g. "v17c bias=0.10 ds=0.20"')
    parser.add_argument("--out", default=None,
                        help="Output directory for plots (default: plots/<label_slug>/)")
    parser.add_argument("--experiments-md", default=None,
                        help="Path to EXPERIMENTS.md (default: auto-detect from project root)")
    parser.add_argument("--no-append", action="store_true",
                        help="Print the EXPERIMENTS.md section but do not write it")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    run_dirs = [Path(r) for r in args.runs]
    for rd in run_dirs:
        if not rd.exists():
            print(f"ERROR: run dir not found: {rd}", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.out) if args.out else script_dir / "plots" / slugify(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiments_md:
        exp_md_path = Path(args.experiments_md)
    else:
        exp_md_path = script_dir / "EXPERIMENTS.md"
        if not exp_md_path.exists():
            exp_md_path = Path("EXPERIMENTS.md")

    print(f"Analyzing {len(run_dirs)} run(s) for: {args.label}")
    print(f"Output dir: {out_dir}")

    # Load all results
    all_results = []
    all_metrics: dict[int, pd.DataFrame] = {}

    for rd in run_dirs:
        seed_results = load_run(rd)
        all_results.extend(seed_results)
        seed_metrics = load_metrics_csv(rd)
        # merge: if same seed exists, prefer first occurrence
        for sid, df in seed_metrics.items():
            if sid not in all_metrics:
                all_metrics[sid] = df

    if not all_results:
        print("ERROR: No test_results.json found in any seed subdir.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_results)} seed results, {len(all_metrics)} metrics CSVs")

    dataset   = infer_dataset(all_results)
    bias_pct  = infer_bias_pct(all_results)

    print(f"Dataset: {dataset}  |  bias_pct: {bias_pct}")

    # Find baselines
    bl_key = (dataset, round(bias_pct, 2)) if bias_pct is not None else None
    baselines = KNOWN_BASELINES.get(bl_key, [])
    if not baselines:
        print(f"  Note: no hardcoded baselines for {bl_key}; comparison plots will show run only")

    # Plots
    print("Generating plots...")
    eos = [get_beta_metrics(r)["eo"] for r in all_results if get_beta_metrics(r)["eo"] is not None]

    if all_metrics:
        plot_deadzone(all_metrics, args.label, out_dir / "deadzone.png")
    else:
        print("  Skipping deadzone.png (no metrics.csv found)")

    plot_eo_comparison(eos, baselines, args.label, out_dir / "eo_comparison.png")
    plot_utility(all_results, baselines, args.label, out_dir / "utility.png")
    plot_per_seed_eo(all_results, baselines, args.label, out_dir / "per_seed_eo.png")

    # Build EXPERIMENTS.md section
    section = build_experiments_section(
        all_results, all_metrics, args.label, baselines,
        run_dirs, out_dir, dataset, bias_pct
    )

    if args.no_append:
        print("\n" + "="*70)
        print("EXPERIMENTS.md section (not written — --no-append flag):")
        print("="*70)
        print(section)
    else:
        if not exp_md_path.exists():
            print(f"WARNING: {exp_md_path} not found; creating a new one.")
            exp_md_path.write_text("# Experiment Findings\n")
        with open(exp_md_path, "a") as f:
            f.write(section)
        print(f"\nAppended section to {exp_md_path}")

    # Print summary to terminal
    eo_mean, eo_std = aggregate(eos)
    f1ws = [get_beta_metrics(r)["f1w"] for r in all_results if get_beta_metrics(r)["f1w"] is not None]
    aucs = [get_beta_metrics(r)["auc"] for r in all_results if get_beta_metrics(r)["auc"] is not None]
    f1w_mean, _ = aggregate(f1ws)
    auc_mean, _  = aggregate(aucs)
    dz_fracs = [deadzone_fraction(df) for df in all_metrics.values()]
    mean_dz = np.nanmean(dz_fracs) if dz_fracs else float("nan")

    print(f"\n{'─'*50}")
    print(f"  β-EO  : {eo_mean:.3f} ± {eo_std:.3f}")
    print(f"  β-F1w : {f1w_mean:.3f}")
    print(f"  β-AUC : {auc_mean:.3f}")
    print(f"  dead% : {mean_dz*100:.1f}%")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
