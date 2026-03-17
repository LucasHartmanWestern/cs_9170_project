# run_baseline.py
"""
Entry point for standalone baselines (Group DRO, etc.).

Usage:
    python run_baseline.py --spec experiment_specs/baseline_gdro_credit.json --device cuda:0
"""

import hashlib
import json
import os
import random
from datetime import datetime

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
        return json.load(f)


def _build_exp_group(baseline: str, spec_name: str, spec: dict) -> str:
    import re
    ts      = datetime.now().strftime("%Y%m%d%H%M")
    payload = json.dumps(spec, sort_keys=True).encode("utf-8")
    h       = hashlib.sha1(payload).hexdigest()[:8]
    slug    = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec_name).strip("-")
    return f"BASELINE_{baseline}_{slug}_{h}__G{ts}"


def run_baseline_all_seeds(spec_path: str, device: str) -> None:
    spec     = _load_spec(spec_path)
    baseline = spec.get("baseline", "group_dro")
    seeds    = spec.get("seeds", [42])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds!r}")

    spec_name = os.path.splitext(os.path.basename(spec_path))[0]
    exp_group = _build_exp_group(baseline, spec_name, spec)

    print(f"[run_baseline] baseline={baseline}")
    print(f"[run_baseline] spec={spec_path}")
    print(f"[run_baseline] device={device}")
    print(f"[run_baseline] exp_group={exp_group}")
    print(f"[run_baseline] seeds={seeds}")

    for seed in seeds:
        seed = int(seed)
        print(f"\n[run_baseline] ---- seed={seed} ----")
        _seed_everything(seed)

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
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                group_dro=spec.get("group_dro"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
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
                real_data_size=spec.get("real_data_size", 3000),
                ffnn=spec.get("ffnn"),
                ot_repair=spec.get("ot_repair"),
                multiclass=spec.get("multiclass", False),
                use_pca=spec.get("use_pca", False),
                pca_components=spec.get("pca_components", 10),
            )
        else:
            raise ValueError(
                f"Unknown baseline: {baseline!r}. "
                "Supported: 'group_dro', 'gaussian_ot_repair'"
            )

        trainer()

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run a standalone fairness baseline.")
    p.add_argument("--spec",   required=True, help="Path to a JSON baseline spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu)")
    args = p.parse_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print(f"[run_baseline] CUDA unavailable, falling back to cpu")
        device = "cpu"

    run_baseline_all_seeds(args.spec, device)


if __name__ == "__main__":
    main()
