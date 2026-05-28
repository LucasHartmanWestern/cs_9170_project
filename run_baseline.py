# run_baseline.py
"""
Entry point for standalone baselines (Group DRO, etc.).

Usage:
    python run_baseline.py --spec experiment_specs/baseline_gdro_credit.yaml --device cuda:0
"""

import hashlib
import json
import os
import random
from datetime import datetime
import yaml

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_spec(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_exp_group(baseline: str, spec_name: str, spec: dict) -> str:
    import re
    ts      = datetime.now().strftime("%Y%m%d%H%M")
    payload = json.dumps(spec, sort_keys=True).encode("utf-8")
    h       = hashlib.sha1(payload).hexdigest()[:8]
    slug    = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec_name).strip("-")
    return f"BASELINE_{baseline}_{slug}_{h}__G{ts}"


def _compute_alpha_eo(spec: dict, seed: int, device: str) -> float:
    """Train alpha on real data for this seed/split and return soft EO gap.
    Uses PCA-transformed features (pca_components=10) to match the RL framework."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from dataset import Dataset
    from agents.ffnn_agent2 import FFNNAgent
    import reward_helpers as rh

    ds = Dataset(
        spec["dataset_name"],
        spec.get("multiclass", False),
        minority_id=spec.get("minority_id"),
        majority_id=spec.get("majority_id"),
        third_id=spec.get("third_id"),
        pca_components=10,
        seed=seed,
        device=torch.device(device),
        use_pca=True,
    )

    dp_col   = spec.get("dp_protected_col")
    fold_idx = spec.get("fold_idx")
    n_folds  = spec.get("n_folds", 5)
    x_tr, x_val, _, y_tr, y_val, _ = ds.get_data_splits(
        train_size=spec.get("real_data_size"),
        bias_pct=spec.get("bias_pct"),
        da_pct=spec.get("da_pct"),
        pca_components=10,
        drop_protected=False,
        protected_cols=ds.protected_attributes,
        bias_val=True,
        win_seconds=float(spec.get("win_seconds", 5.0)),
        step_seconds=float(spec.get("step_seconds", 2.5)),
        **({"dp_protected_col": dp_col} if dp_col is not None else {}),
        **({"acs_states": spec.get("acs_states")} if spec.get("acs_states") is not None else {}),
        **({"fold_idx": fold_idx, "n_folds": n_folds,
            "fold_rng_seed": spec.get("fold_rng_seed")} if fold_idx is not None else {}),
    )

    ffnn_cfg   = spec.get("ffnn", {})
    hidden     = ffnn_cfg.get("hidden_sizes", [32, 16])
    lr         = float(ffnn_cfg.get("learning_rate", ffnn_cfg.get("lr", 0.001)))
    batch_size = int(ffnn_cfg.get("batch_size", 64))
    epochs     = int(ffnn_cfg.get("epochs", 20))

    alpha = FFNNAgent(x_tr.shape[1], hidden_sizes=hidden, output_size=2,
                      classes=[0, 1], type="classification", learning_rate=lr,
                      batch_size=batch_size, epochs=epochs, device=device, seed=seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr.float()), batch_size=batch_size, shuffle=True)
    alpha.train(loader)

    with torch.no_grad():
        p1  = rh.p1_from_agent(alpha, x_val)
        eo  = rh.soft_eo_gap(ds.a_val, y_val, p1)

    return float(eo.item())


def run_baseline_all_seeds(spec_path: str, device: str) -> None:
    spec     = _load_spec(spec_path)
    baseline = spec.get("baseline", "group_dro")
    seeds    = spec.get("seeds", [42])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds!r}")

    eo_guard_threshold = float(spec.get("eo_guard_threshold", 0.0))
    n_seeds_needed     = len(seeds)
    primary_set        = set(int(s) for s in seeds)
    fallback_pool      = [s for s in range(200) if s not in primary_set][:20]
    seed_queue         = [int(s) for s in seeds] + fallback_pool

    spec_name = os.path.splitext(os.path.basename(spec_path))[0]
    exp_group = _build_exp_group(baseline, spec_name, spec)

    print(f"[run_baseline] baseline={baseline}")
    print(f"[run_baseline] spec={spec_path}")
    print(f"[run_baseline] device={device}")
    print(f"[run_baseline] exp_group={exp_group}")
    fold_idx      = spec.get("fold_idx")
    n_folds       = int(spec.get("n_folds", 5))
    fold_rng_seed = spec.get("fold_rng_seed")
    fold_kwargs   = ({"fold_idx": fold_idx, "n_folds": n_folds, "fold_rng_seed": fold_rng_seed}
                     if fold_idx is not None else {})

    print(f"[run_baseline] seeds={seeds}")
    if fold_idx is not None:
        print(f"[run_baseline] k-fold mode: fold_idx={fold_idx}, n_folds={n_folds}")
    if eo_guard_threshold > 0.0:
        print(f"[run_baseline] eo_guard_threshold={eo_guard_threshold} (fallback pool: {fallback_pool[:5]}...)")

    completed = 0
    for seed in seed_queue:
        if completed >= n_seeds_needed:
            break
        seed = int(seed)
        print(f"\n[run_baseline] ---- seed={seed} ----")
        _seed_everything(seed)

        if eo_guard_threshold > 0.0:
            eo = _compute_alpha_eo(spec, seed, device)
            if eo < eo_guard_threshold:
                print(f"[EO guard] alpha-EO={eo:.4f} < threshold={eo_guard_threshold:.4f} — skipping seed {seed}")
                continue
            print(f"[EO guard] alpha-EO={eo:.4f} >= threshold={eo_guard_threshold:.4f} — proceeding")

        win_seconds  = float(spec.get("win_seconds",  5.0))
        step_seconds = float(spec.get("step_seconds", 2.5))

        if baseline == "group_dro":
            from benchmarks.group_dro import GroupDROTrainer
            trainer = GroupDROTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                group_dro=spec.get("group_dro"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                acs_states=spec.get("acs_states"),
                **fold_kwargs,
            )
        elif baseline == "gaussian_ot_repair":
            from benchmarks.gaussian_ot_repair import GaussianOTRepairTrainer
            trainer = GaussianOTRepairTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                ot_repair=spec.get("ot_repair"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                acs_states=spec.get("acs_states"),
                **fold_kwargs,
            )
        elif baseline == "ctgan":
            from benchmarks.ctgan_baseline import CTGANBaselineTrainer
            trainer = CTGANBaselineTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                ctgan=spec.get("ctgan"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                acs_states=spec.get("acs_states"),
                **fold_kwargs,
            )
        elif baseline == "fairness_loss_balancing":
            from benchmarks.fairness_loss_balancing import FairnessLossBalancingTrainer
            trainer = FairnessLossBalancingTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                flb=spec.get("flb"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                acs_states=spec.get("acs_states"),
                **fold_kwargs,
            )
        elif baseline == "smote":
            from benchmarks.smote_baseline import SMOTEBaselineTrainer
            trainer = SMOTEBaselineTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                smote=spec.get("smote"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", True),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                acs_states=spec.get("acs_states"),
                **fold_kwargs,
            )
        elif baseline == "gaussian_augment":
            from benchmarks.gaussian_augment import GaussianAugmentTrainer
            trainer = GaussianAugmentTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                n_synthetic=spec.get("n_synthetic", 2000),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", True),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                **fold_kwargs,
            )
        elif baseline == "fairtabddpm":
            from benchmarks.fairtabddpm_baseline import FairTabDDPMTrainer
            trainer = FairTabDDPMTrainer(
                exp_group=exp_group,
                spec_name=spec_name,
                dataset_name=spec["dataset_name"],
                seed=seed,
                device=device,
                minority_id=spec.get("minority_id"),
                majority_id=spec.get("majority_id"),
                third_id=spec.get("third_id"),
                bias_pct=spec.get("bias_pct"),
                da_pct=spec.get("da_pct"),
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                fairtabddpm=spec.get("fairtabddpm"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                dp_protected_col=spec.get("dp_protected_col", None),
                acs_states=spec.get("acs_states"),
                **fold_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown baseline: {baseline!r}. "
                "Supported: 'group_dro', 'gaussian_ot_repair', 'ctgan', "
                "'fairness_loss_balancing', 'fairtabddpm', 'smote', 'gaussian_augment'"
            )

        trainer()
        completed += 1

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()

    if completed < n_seeds_needed:
        print(f"[run_baseline] WARNING: only {completed}/{n_seeds_needed} seeds completed — fallback pool exhausted")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run a standalone fairness baseline.")
    p.add_argument("--spec",   required=True, help="Path to a YAML baseline spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu)")
    args = p.parse_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print(f"[run_baseline] CUDA unavailable, falling back to cpu")
        device = "cpu"

    run_baseline_all_seeds(args.spec, device)


if __name__ == "__main__":
    main()
