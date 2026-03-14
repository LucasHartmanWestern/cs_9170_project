# main.py
import os
import json
import hashlib
import random
from datetime import datetime
import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    """Seed all global RNGs before any training objects are created."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")

def _load_spec(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _spec_id(spec_path: str, spec: dict) -> str:

    base = os.path.basename(spec_path)
    base_noext = os.path.splitext(base)[0]
    payload = json.dumps(spec, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:8]
    return f"{_slug(base_noext)}_{h}"

def build_exp_group(spec_path: str, spec: dict) -> str:

    ts = datetime.now().strftime("%Y%m%d%H%M")
    return f"SPEC_{_spec_id(spec_path, spec)}__G{ts}"

def run_spec_all_seeds(spec_path: str, device: str):
    from training import Training

    spec = _load_spec(spec_path)

    # Seeds come from JSON
    seeds = spec.get("seeds", [42])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds}")

    # One exp_group for all seeds
    exp_group = build_exp_group(spec_path, spec)
    spec_base = os.path.basename(spec_path)
    spec_name = os.path.splitext(spec_base)[0]

    print(f"[main] spec={spec_base} device={device}")
    print(f"[main] exp_group={exp_group}")
    print(f"[main] seeds={seeds}")

    # Run all seeds sequentially
    for seed in seeds:
        seed = int(seed)
        print(f"[main] ---- running seed={seed} ----")
        _seed_everything(seed)

        lw = spec.get("local_weights", {})
        rs = spec.get("reward_shaping", {})
        trainer = Training(
            exp_group=exp_group,
            spec_name=spec_name,
            dataset_name=spec["dataset_name"],
            seed=seed,
            device=device,

            multiclass=spec.get("multiclass", False),
            curriculum_learning=spec.get("curriculum_learning", True),

            minority_id=spec.get("minority_id"),
            majority_id=spec.get("majority_id"),
            third_id=spec.get("third_id"),

            bias_pct=spec.get("bias_pct"),
            pca_components=spec["pca_components"],
            traj_length=spec["traj_length"],
            real_data_size=spec["real_data_size"],
            total_episodes=spec["total_episodes"],

            reward_mode=spec["reward_mode"],
            lambda_schedule=tuple(spec["lambda_schedule"]),

            local_squash_k=float(rs.get("local_squash_k", 4.0)),
            local_squash_center=float(rs.get("local_squash_center", 0.5)),
            hard_from_beta=bool(rs.get("hard_from_beta", False)),

            use_delta_actions=spec.get("use_delta_actions", True),
            delta_scale=spec.get("delta_scale", 0.10),
            delta_clip=spec.get("delta_clip", 0.20),
            pca_clip=spec.get("pca_clip", None),
            radius_clip=spec.get("radius_clip", None),

            use_pca=spec.get("use_pca", True),
            bias_val=spec.get("bias_val", True),

            ffnn=spec.get("ffnn"),
            reinforce=spec.get("reinforce"),
            curriculum=spec.get("curriculum"),
            benchmarks=spec.get("benchmarks"),

            gen_both_classes=spec.get("gen_both_classes", False),

            w_anchor=float(lw.get("w_anchor",     0.60)),
            w_hard=float(lw.get("w_hard",         0.30)),
            w_div=float(lw.get("w_div",           0.05)),
            sigma_anchor=float(lw.get("sigma_anchor", 0.85)),
            rho_div=float(lw.get("rho_div",       0.60)),
            hard_margin=float(lw.get("hard_margin", 0.65)),
            use_uncertainty_anchors=bool(lw.get("use_uncertainty_anchors", False)),
            uncertainty_warmup_episodes=int(lw.get("uncertainty_warmup_episodes", 0)),
            sigma_calibration_factor=lw.get("sigma_calibration_factor", None),
            anchor_refresh_interval=int(lw.get("anchor_refresh_interval", 0)),
            anchor_refresh_top_k=int(lw.get("anchor_refresh_top_k", 500)),
            anchor_selection_mode=str(lw.get("anchor_selection_mode", "all")),
            anchor_selection_top_k=int(lw.get("anchor_selection_top_k", 200)),

            global_sigmoid_k=float(rs.get("global_sigmoid_k", 10.0)),
        )

        trainer()

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="Path to a JSON experiment spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu if no CUDA)")
    args = p.parse_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    run_spec_all_seeds(args.spec, device)

if __name__ == "__main__":
    main()
