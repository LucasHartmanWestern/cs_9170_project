#!/usr/bin/env python3
"""
check_smoke.py  —  Pass/fail diagnostic for a completed smoke (or full) run.

Usage:
    python check_smoke.py <run_dir_or_prefix> [--seed N] [--threshold loose|normal|strict]

Examples:
    python check_smoke.py training_runs/SPECv7_census_hard_anchors_smoke_...
    python check_smoke.py v7_census_hard_anchors_smoke  --seed 42
    python check_smoke.py v7_census_hard_anchors_smoke  --all-seeds
    python check_smoke.py v7_census_hard_anchors_smoke  --compare v3_census_ablation_global_anchors
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_ROOT = Path("training_runs")

# ── Thresholds (tune as you gather more signal) ────────────────────────────
THRESHOLDS = {
    "loose": {
        "anchor_reward_floor":   0.02,   # anchor_reward_mean in last 30% of run
        "anchor_dist_sigma_ratio": 10.0, # mean(dist / sigma) in last 30%
        "anchor_hard_corr":      -0.30,  # Pearson r(anchor_reward, hard_reward)
        "eo_regress_pct":         0.20,  # best EO <= alpha_EO * (1 + this)
        "gen_radius_cap":         50.0,  # mean gen_radius in last 30% (if no clip)
    },
    "normal": {
        "anchor_reward_floor":   0.05,
        "anchor_dist_sigma_ratio": 5.0,
        "anchor_hard_corr":      -0.15,
        "eo_regress_pct":         0.05,
        "gen_radius_cap":         15.0,
    },
    "strict": {
        "anchor_reward_floor":   0.10,
        "anchor_dist_sigma_ratio": 3.0,
        "anchor_hard_corr":       0.00,
        "eo_regress_pct":         0.00,
        "gen_radius_cap":          5.0,
    },
}

PASS  = "\033[92m✓ PASS\033[0m"
FAIL  = "\033[91m✗ FAIL\033[0m"
WARN  = "\033[93m⚠ WARN\033[0m"
INFO  = "\033[94mℹ\033[0m"

def _pf(ok, strict=True):
    return PASS if ok else (FAIL if strict else WARN)


def find_run_dir(name_or_path: str) -> list[Path]:
    p = Path(name_or_path)
    if p.exists() and p.is_dir():
        return [p]
    # prefix search in RUNS_ROOT
    matches = [d for d in sorted(RUNS_ROOT.iterdir())
               if d.is_dir() and d.name.startswith("SPEC" + name_or_path.lstrip("SPEC"))]
    if not matches:
        matches = [d for d in sorted(RUNS_ROOT.iterdir())
                   if d.is_dir() and name_or_path in d.name]
    return matches


def check_seed(seed_dir: Path, thr: dict, baseline_dir: Path | None = None) -> dict[str, bool]:
    metrics_path = seed_dir / "metrics.csv"
    test_path    = seed_dir / "final_test_metrics.csv"
    meta_path    = seed_dir / "meta.json"

    if not metrics_path.exists():
        print(f"  {FAIL}  metrics.csv not found in {seed_dir}")
        return {}

    df   = pd.read_csv(metrics_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n    = len(df)
    tail = df.iloc[int(n * 0.70):]   # last 30%

    results = {}
    lines   = []

    # ── 1. Generation radius ──────────────────────────────────────────────
    radius_clip = meta.get("radius_clip", None)
    if "extra.diag_gen_radius_mean" in df.columns:
        rad_tail = tail["extra.diag_gen_radius_mean"].mean()
        if radius_clip:
            ok = rad_tail <= radius_clip * 1.5
            lines.append(f"  {_pf(ok)}  Gen radius (last 30% mean): {rad_tail:.1f}  "
                         f"[cap={radius_clip}, limit={radius_clip*1.5:.1f}]")
        else:
            ok = rad_tail <= thr["gen_radius_cap"]
            lines.append(f"  {_pf(ok, strict=False)}  Gen radius (last 30% mean): {rad_tail:.1f}  "
                         f"[no radius_clip set — consider adding one!  limit={thr['gen_radius_cap']}]")
        results["radius"] = ok
    else:
        lines.append(f"  {INFO}  extra.diag_gen_radius_mean not logged")

    # ── 2. Anchor reward alive ────────────────────────────────────────────
    if "local.anchor_reward_mean" in df.columns:
        anc_tail = tail["local.anchor_reward_mean"].mean()
        ok = anc_tail >= thr["anchor_reward_floor"]
        lines.append(f"  {_pf(ok)}  Anchor reward (last 30% mean): {anc_tail:.4f}  "
                     f"[floor={thr['anchor_reward_floor']}]")
        results["anchor_alive"] = ok

    # ── 3. Anchor distance / sigma ratio ─────────────────────────────────
    if "local.min_anchor_dist_mean" in df.columns and "local.sigma_anchor_used" in df.columns:
        dist_v  = tail["local.min_anchor_dist_mean"].values
        sigma_v = tail["local.sigma_anchor_used"].values
        # avoid div-by-zero
        valid   = sigma_v > 1e-6
        if valid.any():
            ratio = (dist_v[valid] / sigma_v[valid]).mean()
            ok    = ratio <= thr["anchor_dist_sigma_ratio"]
            lines.append(f"  {_pf(ok)}  Anchor dist/sigma ratio (last 30%): {ratio:.2f}  "
                         f"[limit={thr['anchor_dist_sigma_ratio']}]")
            results["anchor_dist"] = ok
        else:
            lines.append(f"  {WARN}  sigma_anchor was near-zero throughout")
            results["anchor_dist"] = False

    # ── 4. Anchor–hard reward correlation ─────────────────────────────────
    if "local.anchor_reward_mean" in df.columns and "local.hard_reward_mean" in df.columns:
        a = df["local.anchor_reward_mean"].values
        h = df["local.hard_reward_mean"].values
        mask = ~(np.isnan(a) | np.isnan(h))
        if mask.sum() >= 10:
            r = float(np.corrcoef(a[mask], h[mask])[0, 1])
            ok = r >= thr["anchor_hard_corr"]
            lines.append(f"  {_pf(ok)}  r(anchor_reward, hard_reward): {r:+.3f}  "
                         f"[floor={thr['anchor_hard_corr']}]  "
                         f"{'← aligned ✓' if r > 0 else '← anti-correlated' if r < -0.15 else '← near-zero'}")
            results["anchor_hard_corr"] = ok

    # ── 5. Local–global alignment ─────────────────────────────────────────
    if "align.corr_local_delta" in df.columns:
        cld = df["align.corr_local_delta"].dropna()
        if len(cld) >= 5:
            mean_cld = float(cld.mean())
            late_cld = float(df["align.corr_local_delta"].iloc[int(n*0.5):].dropna().mean()) if n > 10 else mean_cld
            ok_warn  = late_cld >= -0.10
            lines.append(f"  {_pf(ok_warn, strict=False)}  corr_local_delta "
                         f"mean={mean_cld:+.3f}  late={late_cld:+.3f}  "
                         f"[warn if late < -0.10]")
            results["local_global_align"] = ok_warn

    # ── 6. EO regression vs alpha ─────────────────────────────────────────
    if test_path.exists():
        test_df  = pd.read_csv(test_path)
        eo_alpha = test_df["alpha_eo_tpr_diff"].mean()
        eo_beta  = test_df["beta_eo_tpr_diff"].mean()
        eo_best_train = df["fairness.eo_tpr_diff"].min() if "fairness.eo_tpr_diff" in df.columns else np.nan
        limit    = eo_alpha * (1 + thr["eo_regress_pct"])
        ok       = eo_beta <= limit
        delta_pct = (eo_alpha - eo_beta) / (eo_alpha + 1e-9) * 100
        lines.append(f"  {_pf(ok)}  EO final:  alpha={eo_alpha:.4f}  beta={eo_beta:.4f}  "
                     f"Δ={delta_pct:+.1f}%  [limit≤{limit:.4f}]")
        if not np.isnan(eo_best_train):
            best_delta = (eo_alpha - eo_best_train) / (eo_alpha + 1e-9) * 100
            lines.append(f"  {INFO}  EO best checkpoint (training): {eo_best_train:.4f}  "
                         f"Δ={best_delta:+.1f}%")
        results["eo_not_regressed"] = ok

        # Utility check
        for col in ["beta_f1_weighted", "beta_roc_auc", "beta_acc"]:
            a_col = col.replace("beta_", "alpha_")
            if col in test_df.columns and a_col in test_df.columns:
                delta_pp = (test_df[col].mean() - test_df[a_col].mean()) * 100
                ok_u = delta_pp >= -0.5
                lines.append(f"  {_pf(ok_u, strict=False)}  {col.replace('beta_','').upper()} "
                             f"Δ={delta_pp:+.2f}pp  [warn if < -0.5pp]")
    else:
        lines.append(f"  {INFO}  final_test_metrics.csv not found (run may not be complete)")

    # ── 7. Comparison to baseline ──────────────────────────────────────────
    if baseline_dir is not None:
        b_test = baseline_dir / "final_test_metrics.csv"
        if b_test.exists() and test_path.exists():
            b_df   = pd.read_csv(b_test)
            b_eo   = b_df["beta_eo_tpr_diff"].mean()
            my_eo  = pd.read_csv(test_path)["beta_eo_tpr_diff"].mean()
            ok_cmp = my_eo <= b_eo
            lines.append(f"  {_pf(ok_cmp, strict=False)}  vs baseline EO: "
                         f"this={my_eo:.4f}  baseline={b_eo:.4f}  "
                         f"{'better ↓' if ok_cmp else 'worse ↑'}")
            results["beats_baseline"] = ok_cmp

    for l in lines:
        print(l)
    return results


def overall_verdict(all_results: list[dict]) -> None:
    if not all_results:
        print(f"\n{FAIL}  No results collected.")
        return

    # Merge: a check fails if it fails in ANY seed
    merged = {}
    for r in all_results:
        for k, v in r.items():
            merged[k] = merged.get(k, True) and v

    hard_checks  = ["radius", "anchor_alive", "anchor_dist", "eo_not_regressed"]
    soft_checks  = ["anchor_hard_corr", "local_global_align"]

    hard_pass = all(merged.get(k, True) for k in hard_checks)
    soft_pass = all(merged.get(k, True) for k in soft_checks)

    print("\n" + "─" * 60)
    if hard_pass and soft_pass:
        print(f"\033[92m{'━'*20}  OVERALL: PASS — ESCALATE TO FULL RUN  {'━'*20}\033[0m")
    elif hard_pass:
        print(f"\033[93m{'━'*20}  OVERALL: PASS (with warnings) — consider iterating  {'━'*20}\033[0m")
    else:
        print(f"\033[91m{'━'*20}  OVERALL: FAIL — iterate before escalating  {'━'*20}\033[0m")

    failed_hard = [k for k in hard_checks if not merged.get(k, True)]
    failed_soft = [k for k in soft_checks if not merged.get(k, True)]
    if failed_hard:
        print(f"  Failed (hard): {', '.join(failed_hard)}")
    if failed_soft:
        print(f"  Warning (soft): {', '.join(failed_soft)}")
    print("─" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Smoke / run pass-fail checker")
    ap.add_argument("run",  help="Run dir path or name prefix")
    ap.add_argument("--seed",       type=int,   default=None)
    ap.add_argument("--all-seeds",  action="store_true")
    ap.add_argument("--threshold",  choices=["loose", "normal", "strict"], default="normal")
    ap.add_argument("--compare",    default=None, help="Baseline run prefix to compare against")
    args = ap.parse_args()

    thr       = THRESHOLDS[args.threshold]
    run_dirs  = find_run_dir(args.run)

    if not run_dirs:
        print(f"No runs found matching '{args.run}'")
        sys.exit(1)

    baseline_seed_dir = None
    if args.compare:
        b_dirs = find_run_dir(args.compare)
        if b_dirs:
            seed_n = args.seed or 42
            baseline_seed_dir = b_dirs[0] / f"seed_{seed_n}"
            if not baseline_seed_dir.exists():
                baseline_seed_dir = next((b_dirs[0] / d for d in b_dirs[0].iterdir()
                                          if d.is_dir() and d.name.startswith("seed_")), None)

    all_results = []
    for run_dir in run_dirs:
        print(f"\n{'═'*70}")
        print(f"  Run: {run_dir.name}")
        print(f"  Threshold: {args.threshold}")
        print(f"{'═'*70}")

        seed_dirs = sorted(run_dir.glob("seed_*"))
        if not seed_dirs:
            print(f"  {FAIL}  No seed_ subdirs found.")
            continue

        target_seeds = seed_dirs
        if args.seed is not None:
            target_seeds = [s for s in seed_dirs if s.name == f"seed_{args.seed}"]
        elif not args.all_seeds:
            target_seeds = seed_dirs[:1]   # default: first seed only

        for sd in target_seeds:
            print(f"\n  ── {sd.name} ──")
            r = check_seed(sd, thr, baseline_seed_dir)
            all_results.append(r)

    overall_verdict(all_results)


if __name__ == "__main__":
    main()
