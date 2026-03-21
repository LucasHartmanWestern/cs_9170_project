#!/usr/bin/env python3
"""
Analyze curriculum learning smoke-test results.

Run from project root after all 4 runs complete:
    python3 curriculum_specs/analyze_curriculum.py

Compares learning curves across:
  - episode_return  (is the agent learning?)
  - fairness.eo_tpr_diff  (EO gap during training)
  - fairness.worst_loss_beta  (worst-group BCE — what the reward optimises)
  - global.local_reward  (global reward signal)

Also checks for late-training regression and best-checkpoint timing.
"""

import csv, glob, json, os, statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR     = PROJECT_ROOT / "training_runs"

CONFIGS = ["curr_control", "curr_gentle", "curr_standard", "curr_aggressive"]

CONFIG_LABELS = {
    "curr_control":    "control (no curriculum, start_dim=10)",
    "curr_gentle":     "gentle  (start_dim=5,  stages=2)",
    "curr_standard":   "standard (start_dim=2, stages=5)",
    "curr_aggressive": "aggressive (start_dim=1, stages=10)",
}


def find_run_dir(config_name: str) -> Path | None:
    """Find the most recent training_run directory for a given spec name."""
    pattern = str(RUNS_DIR / f"SPEC{config_name}_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])


def load_metrics(run_dir: Path, seed: str = "seed_42") -> list[dict]:
    path = run_dir / seed / "metrics.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def smooth(vals: list[float], w: int = 10) -> list[float]:
    out = []
    for i in range(len(vals)):
        window = vals[max(0, i - w + 1):i + 1]
        out.append(statistics.mean(window))
    return out


def quartile_means(vals: list[float], n: int = 4) -> list[float]:
    q = max(1, len(vals) // n)
    return [statistics.mean(vals[i * q:(i + 1) * q]) for i in range(n)]


def regression_check(vals: list[float]) -> tuple[float, int]:
    """Return (regression_amount, best_episode) — regression = final - best."""
    best = min(vals)
    best_ep = vals.index(best) + 1
    return vals[-1] - best, best_ep


def col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan", "None")]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {}
    missing = []

    for name in CONFIGS:
        run_dir = find_run_dir(name)
        if run_dir is None:
            missing.append(name)
            continue
        rows = load_metrics(run_dir)
        if not rows:
            missing.append(name)
            continue
        results[name] = rows

    if missing:
        print(f"Missing results for: {missing}")
        print("Run: bash curriculum_specs/run_all.sh\n")
        if not results:
            return

    # ── Per-metric comparison ─────────────────────────────────────────────
    METRICS = [
        ("meta.episode_return",        "Episode return",       "higher=better"),
        ("fairness.eo_tpr_diff",       "EO gap (tpr_diff)",    "lower=better"),
        ("fairness.worst_loss_beta",   "Worst-group BCE (β)",  "lower=better"),
    ]

    for key, label, direction in METRICS:
        print(f"\n{'─'*60}")
        print(f"{label}  [{direction}]  — quartile means (Q1→Q4) + final")
        print(f"{'─'*60}")
        for name in CONFIGS:
            if name not in results:
                print(f"  {CONFIG_LABELS[name]:<45}  MISSING")
                continue
            vals = col(results[name], key)
            if not vals:
                print(f"  {CONFIG_LABELS[name]:<45}  no data for {key}")
                continue
            sm = smooth(vals, w=10)
            qs = quartile_means(sm)
            reg, best_ep = regression_check(vals)
            reg_str = f"reg={reg:+.4f}" if key == "fairness.worst_loss_beta" else ""
            print(f"  {CONFIG_LABELS[name]:<45}  "
                  f"Q1={qs[0]:.4f}  Q2={qs[1]:.4f}  Q3={qs[2]:.4f}  Q4={qs[3]:.4f}  "
                  f"final={vals[-1]:.4f}  best_ep={best_ep}  {reg_str}")

    # ── Final test metrics ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Final test metrics (from final_test_metrics.csv)")
    print(f"{'─'*60}")
    print(f"  {'Config':<45}  {'Beta EO':>8}  {'F1w':>8}  {'AUC':>8}  {'Alpha EO':>10}")
    for name in CONFIGS:
        run_dir = find_run_dir(name)
        if run_dir is None:
            print(f"  {CONFIG_LABELS[name]:<45}  MISSING")
            continue
        csv_path = run_dir / "final_test_metrics.csv"
        if not csv_path.exists():
            # try per-seed
            csv_path = run_dir / "seed_42" / "final_test_metrics.csv"
        if not csv_path.exists():
            print(f"  {CONFIG_LABELS[name]:<45}  no final_test_metrics.csv")
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r = rows[-1]
        eo  = float(r.get("beta_eo_tpr_diff", r.get("eo_gap", "nan")))
        f1w = float(r.get("beta_f1_weighted", r.get("f1_weighted", "nan")))
        auc = float(r.get("beta_roc_auc", r.get("roc_auc", "nan")))
        aeo = float(r.get("alpha_eo_tpr_diff", "nan"))
        print(f"  {CONFIG_LABELS[name]:<45}  {eo:>8.4f}  {f1w:>8.4f}  {auc:>8.4f}  {aeo:>10.4f}")

    # ── Summary judgement ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Summary")
    print(f"{'─'*60}")
    print("Key questions answered by this experiment:")
    print("  1. Does curriculum help or hurt vs no-curriculum at gamma=1.0?")
    print("     → Compare control vs gentle/standard/aggressive on EO and worst_loss_beta.")
    print("  2. Does any curriculum config show better convergence speed?")
    print("     → Check if Q1 EO is lower for curriculum configs (faster early learning).")
    print("  3. Is there late-training regression (policy collapse)?")
    print("     → Check reg= values for worst_loss_beta — should be near 0 for stable runs.")
    print("  4. Does aggressive curriculum cause instability (start_dim=1)?")
    print("     → If Q1 is high and Q2+ doesn't recover, aggressive is too restrictive.")
    print("\nIf curriculum helps: add curriculum variants to paper_specs_v1 before SLURM submission.")
    print("If neutral/harmful: confirms current v18 design choice. No spec changes needed.")


if __name__ == "__main__":
    main()
