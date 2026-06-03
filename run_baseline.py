# run_baseline.py
"""
Entry point for standalone fairness baselines.

Usage:
    python run_baseline.py --spec experiment_specs/census/baselines/group_dro.yaml --device cuda:0
"""

import hashlib
import json
import os
import re
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml

_project_root = Path(__file__).parent
for _p in [str(_project_root / 'utilities'), str(_project_root / 'FORGE')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from spec_helpers import _load_spec


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_exp_group(baseline: str, spec_name: str, spec: dict) -> str:
    ts   = datetime.now().strftime("%Y%m%d%H%M")
    h    = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec_name).strip("-")
    return f"BASELINE_{baseline}_{slug}_{h}__G{ts}"


def _compute_alpha_eo(spec: dict, seed: int, device: str) -> float:
    """Train alpha on real data and return soft EO gap. Uses PCA to match FORGE's feature space."""
    from torch.utils.data import DataLoader, TensorDataset
    from dataset import Dataset
    from agents.ffnn_agent import FFNNAgent
    import reward_helpers as rh

    ds = Dataset(
        spec["dataset_name"],
        minority_id=spec.get("minority_id"),
        majority_id=spec.get("majority_id"),
        pca_components=10,
        seed=seed,
        device=torch.device(device),
        use_pca=True,
    )

    dp_col   = spec.get("dp_protected_col")
    fold_idx = spec.get("fold_idx")
    x_tr, x_val, _, y_tr, y_val, _ = ds.get_data_splits(
        train_size=spec.get("real_data_size"),
        da_pct=spec.get("da_pct"),
        pca_components=10,
        drop_protected=False,
        protected_cols=ds.protected_attributes,
        **({"dp_protected_col": dp_col} if dp_col is not None else {}),
        **({"fold_idx": fold_idx, "n_folds": spec.get("n_folds", 5),
            "fold_rng_seed": spec.get("fold_rng_seed")} if fold_idx is not None else {}),
    )

    ffnn_cfg = spec.get("ffnn", {})
    alpha = FFNNAgent(
        x_tr.shape[1],
        hidden_sizes=ffnn_cfg.get("hidden_sizes", [32, 16]),
        output_size=2, classes=[0, 1], type="classification",
        learning_rate=float(ffnn_cfg.get("learning_rate", ffnn_cfg.get("lr", 0.001))),
        batch_size=int(ffnn_cfg.get("batch_size", 64)),
        epochs=int(ffnn_cfg.get("epochs", 20)),
        device=device, seed=seed,
    )
    alpha.train(DataLoader(TensorDataset(x_tr, y_tr.float()),
                           batch_size=int(ffnn_cfg.get("batch_size", 64)), shuffle=True))

    with torch.no_grad():
        eo = rh.soft_eo_gap(ds.a_val, y_val, rh.p1_from_agent(alpha, x_val))
    return float(eo.item())


def _common_kwargs(spec, seed, device, exp_group, spec_name):
    """Parameters shared by every baseline trainer."""
    return dict(
        exp_group=exp_group,
        spec_name=spec_name,
        dataset_name=spec["dataset_name"],
        seed=seed,
        device=device,
        minority_id=spec.get("minority_id"),
        majority_id=spec.get("majority_id"),
        da_pct=spec.get("da_pct"),
        real_data_size=spec.get("real_data_size", 3000),
        ffnn=spec.get("ffnn"),
        use_pca=spec.get("use_pca", False),
        pca_components=spec.get("pca_components", 10),
        dp_protected_col=spec.get("dp_protected_col"),
    )


def run_baseline_all_seeds(spec_path: str, device: str) -> None:
    spec     = _load_spec(spec_path)
    baseline = spec.get("baseline", "group_dro")
    seeds    = spec.get("seeds", [42])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds!r}")

    eo_guard  = float(spec.get("eo_guard_threshold", 0.0))
    n_needed  = len(seeds)
    fallback  = [s for s in range(200) if s not in set(int(s) for s in seeds)][:20]
    queue     = [int(s) for s in seeds] + fallback

    spec_name = os.path.splitext(os.path.basename(spec_path))[0]
    exp_group = _build_exp_group(baseline, spec_name, spec)

    fold_idx  = spec.get("fold_idx")
    fold_kw   = ({"fold_idx": fold_idx, "n_folds": int(spec.get("n_folds", 5)),
                  "fold_rng_seed": spec.get("fold_rng_seed")}
                 if fold_idx is not None else {})

    print(f"[run_baseline] baseline={baseline}  spec={spec_path}  device={device}")
    print(f"[run_baseline] exp_group={exp_group}  seeds={seeds}")
    if fold_idx is not None:
        print(f"[run_baseline] k-fold: fold_idx={fold_idx}, n_folds={spec.get('n_folds', 5)}")
    if eo_guard > 0.0:
        print(f"[run_baseline] eo_guard_threshold={eo_guard}")

    completed = 0
    for seed in queue:
        if completed >= n_needed:
            break
        seed = int(seed)
        print(f"\n[run_baseline] ---- seed={seed} ----")
        _seed_everything(seed)

        if eo_guard > 0.0:
            eo = _compute_alpha_eo(spec, seed, device)
            if eo < eo_guard:
                print(f"[EO guard] alpha-EO={eo:.4f} < {eo_guard:.4f} — skipping seed {seed}")
                continue
            print(f"[EO guard] alpha-EO={eo:.4f} >= {eo_guard:.4f} — proceeding")

        kw = {**_common_kwargs(spec, seed, device, exp_group, spec_name), **fold_kw}

        if baseline == "group_dro":
            from benchmarks.group_dro import GroupDROTrainer
            trainer = GroupDROTrainer(**kw, group_dro=spec.get("group_dro"))

        elif baseline == "gaussian_ot_repair":
            from benchmarks.gaussian_ot_repair import GaussianOTRepairTrainer
            trainer = GaussianOTRepairTrainer(**kw, ot_repair=spec.get("ot_repair"))

        elif baseline == "ctgan":
            from benchmarks.ctgan_baseline import CTGANBaselineTrainer
            ctgan_cfg = dict(spec.get("ctgan") or {})
            if "n_synthetic" not in ctgan_cfg and spec.get("traj_length"):
                ctgan_cfg["n_synthetic"] = spec["traj_length"]
            trainer = CTGANBaselineTrainer(**kw, ctgan=ctgan_cfg)

        elif baseline == "fairness_loss_balancing":
            from benchmarks.fairness_loss_balancing import FairnessLossBalancingTrainer
            trainer = FairnessLossBalancingTrainer(**kw, flb=spec.get("flb"))

        elif baseline == "smote":
            from benchmarks.smote_baseline import SMOTEBaselineTrainer
            trainer = SMOTEBaselineTrainer(**kw, smote=spec.get("smote"))

        elif baseline == "fairtabddpm":
            from benchmarks.fairtabddpm_baseline import FairTabDDPMTrainer
            fairtabddpm_cfg = dict(spec.get("fairtabddpm") or {})
            if "n_synthetic" not in fairtabddpm_cfg and spec.get("traj_length"):
                fairtabddpm_cfg["n_synthetic"] = spec["traj_length"]
            trainer = FairTabDDPMTrainer(**kw, fairtabddpm=fairtabddpm_cfg)

        else:
            raise ValueError(
                f"Unknown baseline: {baseline!r}. "
                "Supported: group_dro, gaussian_ot_repair, ctgan, "
                "fairness_loss_balancing, smote, fairtabddpm"
            )

        trainer()
        completed += 1

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()

    if completed < n_needed:
        print(f"[run_baseline] WARNING: only {completed}/{n_needed} seeds completed")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run a standalone fairness baseline.")
    p.add_argument("--spec",   required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("[run_baseline] CUDA unavailable, falling back to cpu")
        device = "cpu"

    run_baseline_all_seeds(args.spec, device)


if __name__ == "__main__":
    main()
