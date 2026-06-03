"""
Shared utilities for grid-search result loading used by fig_grid_sensitivity.py
and fig_grid_scatter.py.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path


def parse_traj_from_dirname(name: str) -> int | None:
    m = re.search(r"_TRJ(\d+)_", name)
    return int(m.group(1)) if m else None


def read_run(run_dir: Path) -> dict | None:
    """
    Return hyperparams + mean beta-EO (and AUC if available) for one run
    directory, or None if the run is incomplete or unparseable.
    """
    run_dir = Path(run_dir)
    final = run_dir / "final_test_metrics.csv"
    if not final.exists():
        return None

    meta = {}
    for seed_dir in sorted(run_dir.iterdir()):
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            mp = seed_dir / "meta.json"
            if mp.exists():
                with open(mp) as f:
                    meta = json.load(f)
                break

    k    = meta.get("global_sigmoid_k")
    pca  = meta.get("pca_components")
    ep   = meta.get("ffnn_epochs")
    traj = parse_traj_from_dirname(run_dir.name)

    if k is None:
        m = re.search(r"_k(\d+)_", run_dir.name)
        k = int(m.group(1)) if m else None
    if pca is None:
        m = re.search(r"_PCA(\d+)_", run_dir.name)
        pca = int(m.group(1)) if m else None
    if ep is None:
        m = re.search(r"_e(\d+)(?:_|$)", run_dir.name)
        ep = int(m.group(1)) if m else 10

    if any(v is None for v in [k, pca, traj, ep]):
        return None

    try:
        df   = pd.read_csv(final)
        eo   = df["beta_eo_tpr_diff"].dropna().values
        auc  = df["beta_roc_auc"].dropna().values if "beta_roc_auc" in df.columns else np.array([])
        if len(eo) == 0:
            return None
        return dict(
            k=float(k), pca=int(pca), traj=int(traj), ep=int(ep),
            syn_pct=round(traj / 5000 * 100),
            beta_eo_mean=float(np.mean(eo)),
            beta_eo_std=float(np.std(eo)),
            eo=float(np.mean(eo)),
            auc=float(np.mean(auc)) if len(auc) else np.nan,
            n_seeds=len(eo),
        )
    except Exception as e:
        print(f"  [warn] {run_dir.name}: {e}")
        return None


def load_dataset_runs(dirs: list[Path], name_filter, min_seeds: int = 1) -> pd.DataFrame:
    """
    Scan a list of directories for run dirs matching name_filter, load each
    via read_run, and return a DataFrame of completed runs (>= min_seeds seeds).
    """
    records = []
    for root in dirs:
        root = Path(root)
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and name_filter(d.name):
                rec = read_run(d)
                if rec is not None:
                    records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[df["n_seeds"] >= min_seeds].reset_index(drop=True)
    return df
