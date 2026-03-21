#!/usr/bin/env python3
"""
Analyze no-PCA smoke-test results.

Run from project root after runs complete:
    python3 curriculum_specs/analyze_nopca.py

Compares:
  - Can the agent learn at all in raw feature space (nopca_control)?
  - Does curriculum help/hurt with arbitrary feature ordering?
  - Does a larger RL network help in the high-D action space?
  - How does credit_card (~29D) fare compared to census_income (~100D)?
"""

import csv, glob, statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR     = PROJECT_ROOT / "training_runs"

CONFIGS = [
    "nopca_control",
    "nopca_large_net_control",
    "nopca_curr_standard",
    "nopca_curr_gentle",
    "nopca_large_net_curr",
    "nopca_credit_control",
]

CONFIG_LABELS = {
    "nopca_control":           "census  no-curr  [64,64]   (~100D)",
    "nopca_large_net_control": "census  no-curr  [256,256] (~100D)",
    "nopca_curr_standard":     "census  curr-s2  [64,64]   start=2  stages=5",
    "nopca_curr_gentle":       "census  curr-g   [64,64]   start=5  stages=2",
    "nopca_large_net_curr":    "census  curr-s2  [256,256] start=2  stages=5",
    "nopca_credit_control":    "credit  no-curr  [64,64]   (~29D)",
}


def find_run_dir(config_name: str) -> Path | None:
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


def col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan", "None")]


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
        print("Run: bash curriculum_specs/run_nopca.sh\n")
        if not results:
            return

    METRICS = [
        ("meta.episode_return",      "Episode return",      "higher=better"),
        ("fairness.eo_tpr_diff",     "EO gap (tpr_diff)",   "lower=better"),
        ("fairness.worst_loss_beta", "Worst-group BCE (β)", "lower=better"),
    ]

    for key, label, direction in METRICS:
        print(f"\n{'─'*70}")
        print(f"{label}  [{direction}]  — quartile means (Q1→Q4) + final")
        print(f"{'─'*70}")
        for name in CONFIGS:
            if name not in results:
                print(f"  {CONFIG_LABELS[name]:<50}  MISSING")
                continue
            vals = col(results[name], key)
            if not vals:
                print(f"  {CONFIG_LABELS[name]:<50}  no data for {key}")
                continue
            sm = smooth(vals, w=10)
            qs = quartile_means(sm)
            best = min(vals) if direction == "lower=better" else max(vals)
            best_ep = (vals.index(min(vals)) if direction == "lower=better"
                       else vals.index(max(vals))) + 1
            print(f"  {CONFIG_LABELS[name]:<50}  "
                  f"Q1={qs[0]:.4f}  Q2={qs[1]:.4f}  Q3={qs[2]:.4f}  Q4={qs[3]:.4f}  "
                  f"final={vals[-1]:.4f}  best={best:.4f}@ep{best_ep}")

    # ── Final test metrics ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Final test metrics (final_test_metrics.csv)")
    print(f"{'─'*70}")
    print(f"  {'Config':<50}  {'Beta EO':>8}  {'F1w':>8}  {'AUC':>8}  {'Alpha EO':>10}")
    for name in CONFIGS:
        run_dir = find_run_dir(name)
        if run_dir is None:
            print(f"  {CONFIG_LABELS[name]:<50}  MISSING")
            continue
        csv_path = run_dir / "final_test_metrics.csv"
        if not csv_path.exists():
            csv_path = run_dir / "seed_42" / "final_test_metrics.csv"
        if not csv_path.exists():
            print(f"  {CONFIG_LABELS[name]:<50}  no final_test_metrics.csv")
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r = rows[-1]
        eo  = float(r.get("beta_eo_tpr_diff",  r.get("eo_gap",      "nan")))
        f1w = float(r.get("beta_f1_weighted",  r.get("f1_weighted",  "nan")))
        auc = float(r.get("beta_roc_auc",      r.get("roc_auc",      "nan")))
        aeo = float(r.get("alpha_eo_tpr_diff", "nan"))
        print(f"  {CONFIG_LABELS[name]:<50}  {eo:>8.4f}  {f1w:>8.4f}  {auc:>8.4f}  {aeo:>10.4f}")

    # ── Summary questions ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Key questions answered by this experiment:")
    print("  1. Can the agent learn at all in raw feature space?")
    print("     → If nopca_control final EO < alpha_eo, the agent learns despite ~100D actions.")
    print("  2. Does a larger network help for high-D action space?")
    print("     → Compare nopca_control [64,64] vs nopca_large_net_control [256,256].")
    print("  3. Does curriculum hurt (as expected) with arbitrary feature ordering?")
    print("     → Compare nopca_control vs nopca_curr_standard / nopca_curr_gentle.")
    print("     → Without PCA's principled ordering, curriculum start_dim=2 is arbitrary (age, fnlwgt).")
    print("  4. Does credit (~29D) work better than census (~100D) without PCA?")
    print("     → Compare nopca_credit_control final EO vs nopca_control.")
    print("\nIf no-PCA works: consider dropping PCA requirement from final framework.")
    print("If no-PCA fails: PCA is a necessary component (dimensionality control for RL policy).")


if __name__ == "__main__":
    main()
