"""
Rank census grid configs by validation EO (not test EO).

For each run dir, reads best_beta_meta_phase1_class1.json to get the best-reward
episode, then looks up fairness.eo_tpr_diff at that episode in metrics.csv.
This gives val EO at the best-reward checkpoint — no re-runs needed.

Also extracts test EO from final_test_metrics.csv for comparison.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CENSUS_SEARCH_DIRS = [
    Path("/storage_1/epigou_storage/FORGE/training_runs_k10"),
    Path("/storage_1/epigou_storage/FORGE/training_runs"),
]

CAPTURE24_SEARCH_DIRS = [
    Path("/storage_1/epigou_storage/FORGE/training_runs"),
    Path("/storage_1/epigou_storage/FORGE/aulavik_runs/capture_24/k0"),
    Path("/storage_1/epigou_storage/FORGE/aulavik_runs/capture_24/k5"),
    Path("/storage_1/epigou_storage/FORGE/lambda_runs/capture_24/k3"),
    Path("/storage_1/epigou_storage/FORGE/lambda_runs/capture_24/k5"),
]

def parse_dir_name(d: Path):
    """Extract k, pca, ep, traj from directory name."""
    name = d.name
    # k from prefix like SPECcensus_k10_ or k=0/3/5
    km = re.search(r'_k(\d+)_', name)
    pcam = re.search(r'PCA(\d+)', name)
    trjm = re.search(r'TRJ(\d+)', name)
    epm = re.search(r'EP(\d+)', name)
    k = int(km.group(1)) if km else None
    pca = int(pcam.group(1)) if pcam else None
    traj = int(trjm.group(1)) if trjm else None
    ep_total = int(epm.group(1)) if epm else None
    return k, pca, traj, ep_total

def get_classifier_epochs(run_dir: Path):
    """Read ffnn epochs from any seed's meta.json."""
    for sd in sorted(run_dir.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("seed_"):
            continue
        meta_p = sd / "meta.json"
        if meta_p.exists():
            with open(meta_p) as f:
                meta = json.load(f)
            return meta.get("ffnn_epochs", meta.get("ffnn", {}).get("epochs", meta.get("epochs", None)))
    return None

def process_run_dir(run_dir: Path):
    """Return list of per-seed rows with val_eo, test_eo, alpha_eo."""
    rows = []
    for sd in sorted(run_dir.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("seed_"):
            continue
        seed = int(sd.name.split("_")[1])

        # Best checkpoint episode
        meta_p = sd / "best_beta_meta_phase1_class1.json"
        if not meta_p.exists():
            continue
        with open(meta_p) as f:
            meta = json.load(f)
        best_ep = meta.get("episode")

        # Val EO at best-reward checkpoint
        metrics_p = sd / "metrics.csv"
        if not metrics_p.exists():
            continue
        try:
            df = pd.read_csv(metrics_p)
        except Exception:
            continue
        ep_col = "episode" if "episode" in df.columns else "global_ep"
        if "fairness.eo_tpr_diff" not in df.columns or ep_col not in df.columns:
            continue

        row_ep = df[df[ep_col] == best_ep]
        if row_ep.empty:
            idx = (df[ep_col] - best_ep).abs().idxmin()
            row_ep = df.iloc[[idx]]
        val_eo = float(row_ep["fairness.eo_tpr_diff"].iloc[0])
        alpha_val_eo = float(row_ep["fairness.eo_alpha_baseline"].iloc[0]) if "fairness.eo_alpha_baseline" in row_ep.columns else float("nan")

        # Test EO from final_test_metrics.csv
        test_p = sd / "final_test_metrics.csv"
        test_eo = float("nan")
        alpha_test_eo = float("nan")
        if test_p.exists():
            try:
                tdf = pd.read_csv(test_p)
                test_eo = float(tdf["beta_eo_tpr_diff"].iloc[0])
                alpha_test_eo = float(tdf["alpha_eo_tpr_diff"].iloc[0])
            except Exception:
                pass

        rows.append({
            "seed": seed,
            "best_ep": best_ep,
            "val_eo": val_eo,
            "alpha_val_eo": alpha_val_eo,
            "test_eo": test_eo,
            "alpha_test_eo": alpha_test_eo,
        })
    return rows


def collect_configs(search_dirs, name_prefix, skip_restart=True):
    """Scan directories, return seen_configs dict keyed by (k, pca, traj, ep_cls)."""
    seen = {}
    for base in search_dirs:
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if not name.startswith(name_prefix):
                continue
            if skip_restart and "restart" in name:
                continue

            k, pca, traj, ep_total = parse_dir_name(run_dir)
            if k is None:
                continue
            ep_cls = get_classifier_epochs(run_dir)
            if ep_cls is None:
                continue

            seed_rows = process_run_dir(run_dir)
            if not seed_rows:
                continue

            config_key = (k, pca, traj, ep_cls)
            existing = seen.get(config_key)
            if existing is None or len(seed_rows) > existing["n_seeds"]:
                seen[config_key] = {
                    "n_seeds": len(seed_rows),
                    "seed_rows": seed_rows,
                    "run_dir": run_dir.name,
                }
    return seen


def aggregate_configs(seen_configs):
    records = []
    for (k, pca, traj, ep_cls), info in seen_configs.items():
        rows = info["seed_rows"]
        val_eos  = [r["val_eo"]       for r in rows if not np.isnan(r["val_eo"])]
        test_eos = [r["test_eo"]      for r in rows if not np.isnan(r["test_eo"])]
        a_test   = [r["alpha_test_eo"] for r in rows if not np.isnan(r["alpha_test_eo"])]
        if not val_eos:
            continue
        records.append({
            "k": k, "pca": pca, "traj": traj, "ep_cls": ep_cls,
            "n_seeds": len(val_eos),
            "mean_val_eo":  np.mean(val_eos),
            "std_val_eo":   np.std(val_eos,  ddof=1) if len(val_eos)  > 1 else 0.0,
            "mean_test_eo": np.mean(test_eos) if test_eos else float("nan"),
            "std_test_eo":  np.std(test_eos,  ddof=1) if len(test_eos) > 1 else 0.0,
            "mean_alpha_test_eo": np.mean(a_test) if a_test else float("nan"),
            "run_dir": info["run_dir"],
        })
    return pd.DataFrame(records)


def print_ranking(df, label):
    if df.empty:
        print(f"No data for {label}")
        return

    df3 = df[df["n_seeds"] >= 3].copy()
    if df3.empty:
        print(f"No configs with ≥3 seeds for {label}")
        return

    df3["val_upper"] = df3["mean_val_eo"] + df3["std_val_eo"]
    df3_val   = df3.sort_values("mean_val_eo").reset_index(drop=True)
    df3_test  = df3.sort_values("mean_test_eo").reset_index(drop=True)
    df3_upper = df3.sort_values("val_upper").reset_index(drop=True)

    print("=" * 90)
    print(f"DATASET: {label}  — TOP 10 BY VALIDATION EO (≥3 seeds)")
    print("=" * 90)
    print(f"{'k':>3} {'pca':>4} {'traj':>5} {'ep':>4} {'n':>2}  "
          f"{'val_EO (mean±std)':>18} {'val+std':>8} {'test_EO':>12} {'α_test':>8}")
    print("-" * 90)
    for _, r in df3_val.head(10).iterrows():
        print(f"{r['k']:>3} {r['pca']:>4} {r['traj']:>5} {r['ep_cls']:>4} {r['n_seeds']:>2}  "
              f"{r['mean_val_eo']:.3f}±{r['std_val_eo']:.3f}{'':>7}"
              f"{r['val_upper']:.3f}   "
              f"{r['mean_test_eo']:.3f}±{r['std_test_eo']:.3f}  "
              f"{r['mean_alpha_test_eo']:.3f}")

    print()
    print(f"TOP 10 BY TEST EO (≥3 seeds) — for comparison")
    print("-" * 90)
    print(f"{'k':>3} {'pca':>4} {'traj':>5} {'ep':>4} {'n':>2}  "
          f"{'test_EO':>12} {'val_EO':>12} {'val+std':>8}")
    print("-" * 90)
    for _, r in df3_test.head(10).iterrows():
        print(f"{r['k']:>3} {r['pca']:>4} {r['traj']:>5} {r['ep_cls']:>4} {r['n_seeds']:>2}  "
              f"{r['mean_test_eo']:.3f}±{r['std_test_eo']:.3f}  "
              f"{r['mean_val_eo']:.3f}±{r['std_val_eo']:.3f}  "
              f"{r['val_upper']:.3f}")

    top_val   = df3_val.iloc[0]
    top_test  = df3_test.iloc[0]
    top_upper = df3_upper.iloc[0]

    print()
    print("SELECTION CRITERION COMPARISON:")
    print(f"  By mean val EO:      k={top_val['k']},pca={top_val['pca']},traj={top_val['traj']},ep={top_val['ep_cls']}"
          f" → val={top_val['mean_val_eo']:.3f}±{top_val['std_val_eo']:.3f}, test={top_val['mean_test_eo']:.3f}")
    print(f"  By val+std (stable): k={top_upper['k']},pca={top_upper['pca']},traj={top_upper['traj']},ep={top_upper['ep_cls']}"
          f" → val={top_upper['mean_val_eo']:.3f}±{top_upper['std_val_eo']:.3f}, val+std={top_upper['val_upper']:.3f}, test={top_upper['mean_test_eo']:.3f}")
    print(f"  By test EO (ref):    k={top_test['k']},pca={top_test['pca']},traj={top_test['traj']},ep={top_test['ep_cls']}"
          f" → val={top_test['mean_val_eo']:.3f}±{top_test['std_val_eo']:.3f}, test={top_test['mean_test_eo']:.3f}")

    upper_matches_test = (top_upper['k'] == top_test['k'] and top_upper['pca'] == top_test['pca'] and
                          top_upper['traj'] == top_test['traj'] and top_upper['ep_cls'] == top_test['ep_cls'])
    print(f"  val+std winner == test winner: {'YES' if upper_matches_test else 'NO'}")
    print(f"\n  Total configs with ≥3 seeds: {len(df3)}  |  Total overall: {len(df)}")


def main():
    print("\n### CENSUS ###")
    census_seen = collect_configs(CENSUS_SEARCH_DIRS, "SPECcensus_k")
    census_df = aggregate_configs(census_seen)
    print_ranking(census_df, "census_income")

    print("\n\n### CAPTURE24 ###")
    # capture24 runs use SPECcapture24_k prefix; training_runs has k=0 only
    cap_seen = collect_configs(CAPTURE24_SEARCH_DIRS, "SPECcapture24_k")
    cap_df = aggregate_configs(cap_seen)
    print_ranking(cap_df, "capture24")


if __name__ == "__main__":
    main()
