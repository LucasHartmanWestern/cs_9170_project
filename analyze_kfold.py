#!/usr/bin/env python3
"""
analyze_kfold.py — Aggregate FORGE and baseline results across k-fold runs.

Usage:
    # Summarise a set of FORGE fold dirs (one dir per fold):
    python analyze_kfold.py forge \
        training_runs/fold0_dir training_runs/fold1_dir ... [--n-folds 5]

    # Summarise a baseline results dir that contains fold_N subdirs:
    python analyze_kfold.py baseline \
        training_runs/baseline_capture24_kfold_dir [--n-folds 5]

    # Summarise both FORGE and all baselines from a root dir:
    python analyze_kfold.py all  <root_dir>  [--n-folds 5]

Output (printed + saved to <root>/kfold_summary.txt):
  Per-fold and mean±std of alpha-EO, beta-EO, EOd, F1w, AUC for each method.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_forge_fold(fold_dir: Path) -> dict | None:
    """Return metrics from a FORGE seed dir (reads final_test_metrics.csv + meta)."""
    # fold dir may contain seed_N subdirs (one seed per fold in kfold mode)
    candidates = [fold_dir] + sorted(fold_dir.glob("seed_*"))
    for sd in candidates:
        ftm = sd / "final_test_metrics.csv"
        if ftm.exists():
            try:
                df = pd.read_csv(ftm)
                row = df.iloc[0]
                meta_p = sd / "meta.json"
                fold_idx = None
                if meta_p.exists():
                    with open(meta_p) as f:
                        meta = json.load(f)
                    fold_idx = meta.get("fold_idx")
                return {
                    "fold_idx":   fold_idx,
                    "alpha_eo":   float(row.get("alpha_eo_tpr_diff", float("nan"))),
                    "beta_eo":    float(row.get("beta_eo_tpr_diff",  float("nan"))),
                    "beta_eod":   float(row.get("beta_eod_max_diff", float("nan"))),
                    "beta_f1w":   float(row.get("beta_f1_weighted",  float("nan"))),
                    "beta_auc":   float(row.get("beta_roc_auc",      float("nan"))),
                }
            except Exception:
                continue
    return None


def _load_baseline_fold(fold_dir: Path) -> dict | None:
    """Return metrics from a baseline seed dir (reads test_results.json or final_test_metrics.csv)."""
    candidates = [fold_dir] + sorted(fold_dir.glob("seed_*"))
    for sd in candidates:
        # baselines write test_results.json
        trj = sd / "test_results.json"
        if trj.exists():
            try:
                with open(trj) as f:
                    d = json.load(f)
                meta_p = sd / "meta.json"
                fold_idx = None
                if meta_p.exists():
                    with open(meta_p) as f2:
                        meta = json.load(f2)
                    fold_idx = meta.get("fold_idx")
                return {
                    "fold_idx": fold_idx,
                    "alpha_eo": float(d.get("alpha_eo_tpr",  d.get("alpha_eo", float("nan")))),
                    "beta_eo":  float(d.get("beta_eo_tpr",   d.get("eo",       float("nan")))),
                    "beta_eod": float(d.get("beta_eod_max",  d.get("eod_max",  float("nan")))),
                    "beta_f1w": float(d.get("beta_f1w",      d.get("f1w",      float("nan")))),
                    "beta_auc": float(d.get("beta_auc",      d.get("auc",      float("nan")))),
                }
            except Exception:
                continue
        # fallback: final_test_metrics.csv (some baselines write this)
        ftm = sd / "final_test_metrics.csv"
        if ftm.exists():
            return _load_forge_fold(sd)
    return None


def _aggregate(rows: list[dict]) -> dict:
    """Given a list of per-fold dicts, return mean±std for each metric."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "fold_idx"]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if not np.isnan(float(r[k]))]
        out[f"{k}_mean"] = np.mean(vals) if vals else float("nan")
        out[f"{k}_std"]  = (np.std(vals, ddof=1) if len(vals) > 1 else 0.0) if vals else float("nan")
        out[f"{k}_vals"] = vals
    return out


def _fmt(mean, std):
    if np.isnan(mean):
        return "  n/a  "
    return f"{mean:.3f}±{std:.3f}"


def print_summary(method: str, agg: dict, fold_rows: list[dict]):
    print(f"\n{'─'*70}")
    print(f"  {method}")
    print(f"{'─'*70}")
    n = len(fold_rows)
    print(f"  Folds completed: {n}")
    if not agg:
        print("  No data.")
        return

    # Per-fold table
    print(f"\n  {'fold':>5} {'α-EO':>8} {'β-EO':>8} {'EOd':>8} {'F1w':>8} {'AUC':>8}")
    print(f"  {'─'*45}")
    for r in sorted(fold_rows, key=lambda x: x.get("fold_idx") or 0):
        fi = r.get("fold_idx", "?")
        print(f"  {str(fi):>5} "
              f"{r['alpha_eo']:8.3f} {r['beta_eo']:8.3f} {r['beta_eod']:8.3f} "
              f"{r['beta_f1w']:8.3f} {r['beta_auc']:8.3f}")

    print(f"\n  {'mean':>5} "
          f"{_fmt(agg['alpha_eo_mean'], agg['alpha_eo_std']):>12} "
          f"{_fmt(agg['beta_eo_mean'],  agg['beta_eo_std']):>12} "
          f"{_fmt(agg['beta_eod_mean'], agg['beta_eod_std']):>12} "
          f"{_fmt(agg['beta_f1w_mean'], agg['beta_f1w_std']):>12} "
          f"{_fmt(agg['beta_auc_mean'], agg['beta_auc_std']):>12}")


# ── subcommand: forge ─────────────────────────────────────────────────────────

def cmd_forge(args):
    fold_rows = []
    for d in args.dirs:
        r = _load_forge_fold(Path(d))
        if r:
            fold_rows.append(r)
        else:
            print(f"  WARNING: no results found in {d}")
    agg = _aggregate(fold_rows)
    print_summary("FORGE", agg, fold_rows)
    return agg, fold_rows


# ── subcommand: baseline ──────────────────────────────────────────────────────

def cmd_baseline(args):
    root = Path(args.root)
    # Look for fold_N subdirs or seed_N subdirs directly
    fold_dirs = sorted(root.glob("fold_*")) or sorted(root.glob("seed_*")) or [root]
    fold_rows = []
    for d in fold_dirs:
        r = _load_baseline_fold(d)
        if r:
            fold_rows.append(r)
    method = args.method or root.name
    agg = _aggregate(fold_rows)
    print_summary(method, agg, fold_rows)
    return agg, fold_rows


# ── subcommand: all ───────────────────────────────────────────────────────────

def cmd_all(args):
    """
    Expects root_dir to contain:
      forge/fold_0/, forge/fold_1/, ...
      baselines/group_dro/fold_0/, ...
      baselines/smote/fold_0/, ...
      etc.
    Or: each method in a subdirectory with seed_N dirs inside.
    """
    root = Path(args.root)
    results = {}

    # FORGE
    forge_dirs = sorted((root / "forge").glob("*")) if (root / "forge").exists() else []
    if forge_dirs:
        rows = [r for d in forge_dirs if (r := _load_forge_fold(d)) is not None]
        results["FORGE"] = (_aggregate(rows), rows)
        print_summary("FORGE", *results["FORGE"])

    # Baselines
    bl_root = root / "baselines"
    if bl_root.exists():
        for method_dir in sorted(bl_root.iterdir()):
            if not method_dir.is_dir():
                continue
            fold_dirs = sorted(method_dir.glob("fold_*")) or sorted(method_dir.glob("seed_*")) or [method_dir]
            rows = [r for d in fold_dirs if (r := _load_baseline_fold(d)) is not None]
            if rows:
                agg = _aggregate(rows)
                results[method_dir.name] = (agg, rows)
                print_summary(method_dir.name, agg, rows)

    # Comparison table
    if results:
        print(f"\n{'='*70}")
        print("  COMPARISON TABLE (mean±std across folds)")
        print(f"{'='*70}")
        print(f"  {'Method':<22} {'α-EO':>12} {'β-EO':>12} {'EOd':>12} {'F1w':>12} {'AUC':>12}")
        print(f"  {'─'*64}")
        for method, (agg, _) in results.items():
            if not agg:
                continue
            print(f"  {method:<22} "
                  f"{_fmt(agg['alpha_eo_mean'], agg['alpha_eo_std']):>12} "
                  f"{_fmt(agg['beta_eo_mean'],  agg['beta_eo_std']):>12} "
                  f"{_fmt(agg['beta_eod_mean'], agg['beta_eod_std']):>12} "
                  f"{_fmt(agg['beta_f1w_mean'], agg['beta_f1w_std']):>12} "
                  f"{_fmt(agg['beta_auc_mean'], agg['beta_auc_std']):>12}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate k-fold results.")
    sub = ap.add_subparsers(dest="cmd")

    p_forge = sub.add_parser("forge", help="Aggregate FORGE fold dirs")
    p_forge.add_argument("dirs", nargs="+", help="One run dir per fold")
    p_forge.add_argument("--n-folds", type=int, default=5)

    p_bl = sub.add_parser("baseline", help="Aggregate a single baseline across folds")
    p_bl.add_argument("root", help="Root dir containing fold_N or seed_N subdirs")
    p_bl.add_argument("--method", default=None, help="Method name (default: dir name)")
    p_bl.add_argument("--n-folds", type=int, default=5)

    p_all = sub.add_parser("all", help="Aggregate all methods from a structured root dir")
    p_all.add_argument("root", help="Root dir containing forge/ and baselines/ subdirs")
    p_all.add_argument("--n-folds", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "forge":
        cmd_forge(args)
    elif args.cmd == "baseline":
        cmd_baseline(args)
    elif args.cmd == "all":
        cmd_all(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
