"""
Aggregate final_test_metrics.csv across all k=0 Huron census grid runs.
Groups by (ffnn_epochs, pca_components, traj_length), deduplicates by seed
(latest timestamp wins if a seed appears in multiple dirs), then ranks by mean β-EO.

Usage:  python analyze_k0_grid.py [--storage /storage_1/epigou_storage/FORGE/training_runs]
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

STORAGE = Path("/storage_1/epigou_storage/FORGE/training_runs")
TARGET_K = 0.0


def collect_rows(storage: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(storage.glob("SPECcensus_k0*")):
        for seed_dir in sorted(run_dir.glob("seed_*")):
            meta_path = seed_dir / "meta.json"
            csv_path  = seed_dir / "final_test_metrics.csv"
            if not meta_path.exists() or not csv_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                df   = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue
            row = df.iloc[-1].to_dict()  # one row per seed
            row["ffnn_epochs"]    = meta.get("ffnn_epochs")
            row["pca_components"] = meta.get("pca_components")
            row["traj_length"]    = meta.get("TRAJ_LENGTH")
            row["real_size"]      = meta.get("REAL_DATA_SIZE")
            row["global_k"]       = meta.get("global_sigmoid_k")
            row["seed"]           = meta.get("seed")
            row["run_dir"]        = run_dir.name
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", default=str(STORAGE))
    args = parser.parse_args()

    storage = Path(args.storage)
    raw = collect_rows(storage)

    # Keep only k=0 runs
    raw = raw[raw["global_k"] == TARGET_K].copy()
    print(f"Collected {len(raw)} seed-rows for k={TARGET_K}")

    # Deduplicate: if same (ffnn_epochs, pca, traj, seed) appears in multiple dirs,
    # keep the one with the most recent timestamp (run_dir name encodes timestamp).
    raw = raw.sort_values("run_dir")
    key_cols = ["ffnn_epochs", "pca_components", "traj_length", "seed"]
    raw = raw.drop_duplicates(subset=key_cols, keep="last")
    print(f"After deduplication: {len(raw)} unique (epochs, pca, traj, seed) rows")

    # Group and aggregate
    metric_cols = {
        "alpha_eo":  "alpha_eo_tpr_diff",
        "beta_eo":   "beta_eo_tpr_diff",
        "beta_f1w":  "beta_f1_weighted",
        "beta_auc":  "beta_roc_auc",
        "alpha_f1w": "alpha_f1_weighted",
    }
    group_cols = ["ffnn_epochs", "pca_components", "traj_length"]
    agg_rows = []
    for key, grp in raw.groupby(group_cols):
        n = len(grp)
        rec = dict(zip(group_cols, key))
        rec["n_seeds"] = n
        for label, col in metric_cols.items():
            if col in grp.columns:
                rec[f"{label}_mean"] = grp[col].mean()
                rec[f"{label}_std"]  = grp[col].std(ddof=1) if n > 1 else float("nan")
        # EO improvement
        if "alpha_eo_mean" in rec and "beta_eo_mean" in rec:
            rec["eo_delta_mean"] = rec["alpha_eo_mean"] - rec["beta_eo_mean"]
        agg_rows.append(rec)

    agg = pd.DataFrame(agg_rows).sort_values("beta_eo_mean")

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("k=0 census grid — ranked by mean β-EO (lower is fairer)")
    print("=" * 90)
    hdr = (f"{'ep':>4} {'pca':>4} {'traj':>5} {'n':>2}  "
           f"{'α-EO':>7} {'β-EO':>14} {'Δ-EO':>7}  {'β-F1w':>7}  {'β-AUC':>7}")
    print(hdr)
    print("-" * 90)
    for _, r in agg.iterrows():
        a_eo = r.get("alpha_eo_mean", float("nan"))
        b_eo = r.get("beta_eo_mean",  float("nan"))
        b_eo_s = r.get("beta_eo_std",  float("nan"))
        deo  = r.get("eo_delta_mean", float("nan"))
        b_f1 = r.get("beta_f1w_mean", float("nan"))
        b_auc= r.get("beta_auc_mean", float("nan"))
        print(f"{int(r['ffnn_epochs']):>4} {int(r['pca_components']):>4} {int(r['traj_length']):>5} "
              f"{int(r['n_seeds']):>2}  "
              f"{a_eo:>7.3f} {b_eo:>7.3f}±{b_eo_s:.3f} {deo:>+7.3f}  "
              f"{b_f1:>7.3f}  {b_auc:>7.3f}")
    print("=" * 90)

    # ── Top 5 ─────────────────────────────────────────────────────────────
    print("\nTop 5 configurations by β-EO:")
    for i, (_, r) in enumerate(agg.head(5).iterrows(), 1):
        print(f"  {i}. ep={int(r['ffnn_epochs'])} pca={int(r['pca_components'])} "
              f"traj={int(r['traj_length'])}  "
              f"β-EO={r['beta_eo_mean']:.3f}±{r['beta_eo_std']:.3f}  "
              f"β-F1w={r['beta_f1w_mean']:.3f}  β-AUC={r['beta_auc_mean']:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path("analysis_k0_grid.csv")
    agg.to_csv(out, index=False)
    print(f"\nFull table saved to {out}")


if __name__ == "__main__":
    main()
