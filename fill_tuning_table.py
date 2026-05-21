"""
Aggregate completed grid-search runs and patch the tuning table (tab:tuning)
in Journal_Paper_v9.tex with available marginal results.

Marginal: for each (dataset, parameter, value), average beta_eo and beta_auc
across all completed (k, pca, ratio, epochs, seed) combinations that share
that parameter value.  Deduplicates by (dataset, k, pca, epochs, ratio, seed),
keeping the latest run_dir (restart runs supersede originals).

Usage:
    python fill_tuning_table.py [--dry-run]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

STORAGE_DIRS = [
    Path("/storage_1/epigou_storage/FORGE/training_runs"),
    Path("/home/epigou/cs_9170_project/training_runs"),
]

TEX_FILE = Path("/home/epigou/cs_9170_project/paper/Journal_Paper_v9.tex")

TOTAL_DATA = 5000  # ratio_trajectory = TRAJ_LENGTH / TOTAL_DATA


def ratio_label(traj_length: int) -> float:
    return round(traj_length / TOTAL_DATA, 2)


def collect(storage_dirs) -> pd.DataFrame:
    rows = []
    for storage in storage_dirs:
        if not storage.exists():
            continue
        pattern = "SPECcensus_k*" if "training_runs" in str(storage) else "SPEC*"
        for run_dir in sorted(storage.glob("SPEC*")):
            name = run_dir.name
            if not (name.startswith("SPECcensus_k") or name.startswith("SPECcapture24_k")):
                continue
            for seed_dir in sorted(run_dir.glob("seed_*")):
                meta_p = seed_dir / "meta.json"
                csv_p  = seed_dir / "final_test_metrics.csv"
                if not meta_p.exists() or not csv_p.exists():
                    continue
                try:
                    meta = json.loads(meta_p.read_text())
                    df   = pd.read_csv(csv_p)
                except Exception:
                    continue
                if df.empty:
                    continue
                row = df.iloc[-1].to_dict()
                row["k"]       = float(meta.get("global_sigmoid_k", -1))
                row["pca"]     = int(meta.get("pca_components", -1))
                row["epochs"]  = int(meta.get("ffnn_epochs", -1))
                row["ratio"]   = ratio_label(int(meta.get("TRAJ_LENGTH", 0)))
                row["seed"]    = int(meta.get("seed", -1))
                row["dataset"] = meta.get("dataset_name", "")
                row["run_dir"] = run_dir.name
                rows.append(row)

    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "k", "pca", "epochs", "ratio", "seed"]
    df = df.sort_values("run_dir")
    return df.drop_duplicates(subset=key, keep="last")


def marginal(df: pd.DataFrame, dataset: str, param: str, value) -> dict:
    """Mean ± std of beta_eo and beta_auc over all runs with param==value."""
    sub = df[(df["dataset"] == dataset) & (df[param] == value)]
    if sub.empty:
        return {}
    eo  = sub["beta_eo_tpr_diff"].dropna()
    auc = sub["beta_roc_auc"].dropna()
    return {
        "eo_mean":  eo.mean(),
        "eo_std":   eo.std(ddof=1) if len(eo) > 1 else float("nan"),
        "auc_mean": auc.mean(),
        "auc_std":  auc.std(ddof=1) if len(auc) > 1 else float("nan"),
        "n":        len(sub),
    }


def fmt(val, decimals=3) -> str:
    if np.isnan(val):
        return "--"
    return f"{val:.{decimals}f}"


def patch_cell(line: str, col_idx: int, new_val: str) -> str:
    """Replace the col_idx-th & … & cell (0-based) in a LaTeX table row."""
    parts = line.split("&")
    if col_idx >= len(parts):
        return line
    parts[col_idx] = f" {new_val} "
    return "&".join(parts)


DATASET_NAMES = {
    "census_income": "census",
    "capture24":     "capture24",
}

# Parameter name → column used in df
PARAM_MAP = {
    "k":      "k",
    "pca":    "pca",
    "ratio":  "ratio",
    "epochs": "epochs",
}


def build_results(df: pd.DataFrame):
    """Return dict: results[dataset][param][value] = {eo_mean, eo_std, auc_mean, auc_std, n}."""
    results = {}
    for dataset_name, short in DATASET_NAMES.items():
        results[short] = {}
        sub = df[df["dataset"] == dataset_name]
        for param_label, col in PARAM_MAP.items():
            results[short][param_label] = {}
            for val in sorted(sub[col].unique()):
                results[short][param_label][val] = marginal(sub, dataset_name, col, val)
    return results


def print_summary(results):
    for ds, params in results.items():
        print(f"\n=== {ds} ===")
        for param, vals in params.items():
            print(f"  {param}:")
            for v, stats in sorted(vals.items()):
                if stats:
                    print(f"    {v:>5}  EO={fmt(stats['eo_mean'])}±{fmt(stats['eo_std'])}  "
                          f"AUC={fmt(stats['auc_mean'])}±{fmt(stats['auc_std'])}  n={stats['n']}")
                else:
                    print(f"    {v:>5}  --")


# ── LaTeX table patching ─────────────────────────────────────────────────────
# The table has columns: Parameter | Value | census EO | std | AUC | std | capture24 EO | std | AUC | std
# (0-indexed cells after splitting on &)
# Census:   EO=col2, std=col3, AUC=col4, std=col5
# Capture24: EO=col6, std=col7, AUC=col8, std=col9

COL_CENSUS_EO   = 2
COL_CENSUS_ESTD = 3
COL_CENSUS_AUC  = 4
COL_CENSUS_ASTD = 5
COL_C24_EO      = 6
COL_C24_ESTD    = 7
COL_C24_AUC     = 8
COL_C24_ASTD    = 9

# Map LaTeX row markers to (param_col, value)
# Patterns to detect which row we're in:
ROW_PATTERNS = [
    # (regex to match value cell content, param, value)
    (r"\|\{0\}",          "k",      0.0),
    (r"\|\{3\}",          "k",      3.0),
    (r"\|\{5\}",          "k",      5.0),
    (r"\|\{10\}",         "k",      10.0),
    (r"\|\{5\}",          "pca",    5),    # duplicate — distinguished by block context
    (r"\|\{10\}",         "pca",    10),
    (r"\|\{15\}",         "pca",    15),
    (r"\|\{20\\%\}",      "ratio",  0.2),
    (r"\|\{40\\%\}",      "ratio",  0.4),
    (r"\|\{60\\%\}",      "ratio",  0.6),
    (r"\|\{10\}",         "epochs", 10),
    (r"\|\{20\}",         "epochs", 20),
    (r"\|\{30\}",         "epochs", 30),
]

# Simpler: parse the table by tracking which \multirow block we're in.
BLOCK_ORDER = ["k", "pca", "ratio", "epochs"]


def fill_table(tex: str, results: dict, dry_run: bool) -> str:
    lines = tex.split("\n")
    out   = []
    current_block = None
    block_idx     = -1

    for line in lines:
        # Detect block transitions by \multirow
        if r"\multirow" in line:
            block_idx += 1
            if block_idx < len(BLOCK_ORDER):
                current_block = BLOCK_ORDER[block_idx]

        if current_block and r"\multicolumn{1}{c|}" in line:
            # Extract the value from the first \multicolumn{1}{c|}{VALUE} cell
            m = re.search(r"\\multicolumn\{1\}\{c\|\}\{([^}]+)\}", line)
            if m:
                raw_val = m.group(1).strip().replace("\\%", "").replace("\\textbf{", "").replace("}", "")
                try:
                    if current_block in ("k", "pca", "epochs"):
                        val = int(raw_val) if "." not in raw_val else float(raw_val)
                    else:  # ratio
                        val = float(raw_val) / 100.0
                except ValueError:
                    val = None

                if val is not None:
                    c_stats  = results.get("census",   {}).get(current_block, {}).get(val, {})
                    c24_stats = results.get("capture24", {}).get(current_block, {}).get(val, {})

                    parts = line.split("&")
                    if len(parts) >= 10:
                        if c_stats:
                            parts[COL_CENSUS_EO]   = f" {fmt(c_stats['eo_mean'])} "
                            parts[COL_CENSUS_ESTD] = f" $\\pm${fmt(c_stats['eo_std'])} "
                            if not np.isnan(c_stats['auc_mean']):
                                parts[COL_CENSUS_AUC]  = f" {fmt(c_stats['auc_mean'])} "
                                parts[COL_CENSUS_ASTD] = f" \\multicolumn{{1}}{{l|}}{{$\\pm${fmt(c_stats['auc_std'])}}} "
                        if c24_stats:
                            parts[COL_C24_EO]   = f" {fmt(c24_stats['eo_mean'])} "
                            parts[COL_C24_ESTD] = f" $\\pm${fmt(c24_stats['eo_std'])} "
                            if not np.isnan(c24_stats['auc_mean']):
                                parts[COL_C24_AUC]  = f" {fmt(c24_stats['auc_mean'])} "
                                parts[COL_C24_ASTD] = f" -- "
                        line = "&".join(parts)

        out.append(line)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't patch tex")
    args = parser.parse_args()

    print("Collecting runs...")
    raw = collect(STORAGE_DIRS)
    print(f"  Raw rows: {len(raw)}")
    df = deduplicate(raw)
    print(f"  After dedup: {len(df)} seed-rows")

    results = build_results(df)
    print_summary(results)

    if args.dry_run:
        print("\n[dry-run] Not patching tex.")
        return

    tex = TEX_FILE.read_text()
    patched = fill_table(tex, results, dry_run=False)
    TEX_FILE.write_text(patched)
    print(f"\nPatched {TEX_FILE}")


if __name__ == "__main__":
    main()
