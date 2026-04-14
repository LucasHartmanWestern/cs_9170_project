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

_KNOWN_TOP_LEVEL_FIELDS = {
    "dataset_name", "multiclass", "bias_pct", "da_pct", "dp_protected_col",
    "reward_mode", "lambda_schedule", "minority_id", "majority_id", "third_id",
    "use_pca", "pca_components", "traj_length", "real_data_size",
    "total_episodes", "phase2_episodes", "curriculum_learning",
    "use_delta_actions", "delta_scale", "delta_clip", "pca_clip", "radius_clip",
    "gen_both_classes", "bias_val", "seeds", "use_cmaes", "cmaes",
    "reward_shaping", "local_weights", "ffnn", "reinforce", "curriculum", "benchmarks",
    "win_seconds", "step_seconds", "eo_guard_threshold",
    "whiten_pca", "beta_reset_interval", "beta_warmstart_from_alpha",
    "pool_pos_fraction",
    "use_ppo", "ppo",
}
_KNOWN_LOCAL_WEIGHTS = {
    "w_ot", "use_dvrl_local", "dvrl_max_bce", "dvrl_scale",
    "w_anchor", "w_hard", "w_div", "sigma_anchor", "rho_div", "hard_margin",
    "use_uncertainty_anchors", "uncertainty_warmup_episodes", "sigma_calibration_factor",
    "anchor_refresh_interval", "anchor_refresh_top_k",
    "anchor_selection_mode", "anchor_selection_top_k",
}
_KNOWN_REWARD_SHAPING = {
    "global_sigmoid_k", "utility_guard_min_factor",
    "local_squash_k", "local_squash_center", "hard_from_beta",
    "roc_eo_lambda",
}
_REQUIRED_TOP_LEVEL = {
    "dataset_name", "reward_mode", "lambda_schedule",
    "pca_components", "traj_length", "real_data_size", "total_episodes",
}

def validate_spec(spec: dict, spec_path: str) -> None:
    """Warn on unknown fields and flag suspicious parameter combinations."""
    warnings = []

    # Unknown top-level fields
    unknown = set(spec.keys()) - _KNOWN_TOP_LEVEL_FIELDS
    for k in sorted(unknown):
        warnings.append(f"Unknown top-level field: '{k}' (typo?)")

    # Unknown nested fields
    for k in sorted(set(spec.get("local_weights", {}).keys()) - _KNOWN_LOCAL_WEIGHTS):
        warnings.append(f"Unknown local_weights field: '{k}'")
    for k in sorted(set(spec.get("reward_shaping", {}).keys()) - _KNOWN_REWARD_SHAPING):
        warnings.append(f"Unknown reward_shaping field: '{k}'")

    # Required fields
    for k in sorted(_REQUIRED_TOP_LEVEL):
        if k not in spec:
            warnings.append(f"Missing required field: '{k}'")

    # Suspicious combos
    if spec.get("use_cmaes") and spec.get("use_delta_actions"):
        warnings.append("use_cmaes=True + use_delta_actions=True: CMA-ES ignores delta constraints — likely unintended")
    if spec.get("use_cmaes") and spec.get("curriculum_learning"):
        warnings.append("use_cmaes=True + curriculum_learning=True: curriculum is a REINFORCE concept; may have no effect with CMA-ES")
    lw = spec.get("local_weights", {})
    if float(lw.get("w_ot", 0.0)) > 0 and spec.get("use_cmaes"):
        warnings.append("w_ot > 0 with use_cmaes=True: OT local reward had negligible effect in v21; confirm this is intentional")
    lambda_sched = spec.get("lambda_schedule", [1.0, 1.0])
    if isinstance(lambda_sched, list) and len(lambda_sched) == 2:
        lam = lambda_sched[0]
        if lam < 1.0 and float(lw.get("w_ot", 0.0)) == 0.0 and not lw.get("use_dvrl_local"):
            warnings.append(f"lambda_schedule={lambda_sched} but all local weights are zero — global-only run with lambda<1.0")
    if spec.get("bias_val") is True:
        warnings.append("bias_val=True: validation set will be biased; use False unless intentionally testing under bias")

    if warnings:
        print(f"[validate_spec] {spec_path}: {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ⚠  {w}")
    else:
        print(f"[validate_spec] {spec_path}: OK")

def run_spec_all_seeds(spec_path: str, device: str, output_dir: str = "training_runs"):
    from training import Training

    spec = _load_spec(spec_path)
    validate_spec(spec, spec_path)

    # Seeds come from JSON
    seeds = spec.get("seeds", [42])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds}")

    eo_guard_threshold = float(spec.get("eo_guard_threshold", 0.0))
    n_seeds_needed = len(seeds)

    # Build fallback pool: first 20 non-primary seeds (used when EO guard skips a seed)
    primary_set = set(int(s) for s in seeds)
    fallback_pool = [s for s in range(200) if s not in primary_set][:20]
    seed_queue = [int(s) for s in seeds] + fallback_pool

    # One exp_group for all seeds
    exp_group = build_exp_group(spec_path, spec)
    spec_base = os.path.basename(spec_path)
    spec_name = os.path.splitext(spec_base)[0]

    print(f"[main] spec={spec_base} device={device}")
    print(f"[main] exp_group={exp_group}")
    print(f"[main] seeds={seeds}")
    if eo_guard_threshold > 0.0:
        print(f"[main] eo_guard_threshold={eo_guard_threshold} (fallback pool: {fallback_pool[:5]}...)")

    # Run seeds sequentially; skip via EO guard and pull from fallback pool as needed
    completed = 0
    for seed in seed_queue:
        if completed >= n_seeds_needed:
            break
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
            da_pct=spec.get("da_pct"),
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
            whiten_pca=spec.get("whiten_pca", False),
            bias_val=spec.get("bias_val", False),
            win_seconds=float(spec.get("win_seconds", 5.0)),
            step_seconds=float(spec.get("step_seconds", 2.5)),

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
            sigma_calibration_factor=float(lw["sigma_calibration_factor"]) if lw.get("sigma_calibration_factor") is not None else None,
            anchor_refresh_interval=int(lw.get("anchor_refresh_interval", 0)),
            anchor_refresh_top_k=int(lw.get("anchor_refresh_top_k", 500)),
            anchor_selection_mode=str(lw.get("anchor_selection_mode", "all")),
            anchor_selection_top_k=int(lw.get("anchor_selection_top_k", 200)),
            use_dvrl_local=bool(lw.get("use_dvrl_local", False)),
            dvrl_max_bce=float(lw.get("dvrl_max_bce", 0.693)),
            dvrl_scale=float(lw.get("dvrl_scale", 1.0)),

            global_sigmoid_k=float(rs.get("global_sigmoid_k", 10.0)),
            utility_guard_min_factor=float(rs.get("utility_guard_min_factor", 1.0)),
            roc_eo_lambda=float(rs.get("roc_eo_lambda", 0.5)),
            phase2_episodes=spec.get("phase2_episodes", None),
            beta_reset_interval=int(spec.get("beta_reset_interval", 1)),
            beta_warmstart_from_alpha=bool(spec.get("beta_warmstart_from_alpha", False)),
            dp_protected_col=spec.get("dp_protected_col", None),
            eo_guard_threshold=eo_guard_threshold,
            w_ot=float(lw.get("w_ot", 0.0)),
            use_cmaes=bool(spec.get("use_cmaes", False)),
            cmaes=spec.get("cmaes", None),
            use_ppo=bool(spec.get("use_ppo", False)),
            ppo=spec.get("ppo", None),
            pool_pos_fraction=spec.get("pool_pos_fraction", None),
        )

        trainer.save_dir = output_dir
        result = trainer()

        if result == "eo_guard_skip":
            print(f"[main] seed={seed} skipped by EO guard — pulling from fallback pool")
            continue

        completed += 1

        del trainer
        if torch.cuda.is_available() and "cuda" in device:
            import gc
            gc.collect()
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()

    if completed < n_seeds_needed:
        print(f"[main] WARNING: only {completed}/{n_seeds_needed} seeds completed — fallback pool exhausted")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="Path to a JSON experiment spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu if no CUDA)")
    p.add_argument("--output_dir", default="training_runs", help="Directory to save results")
    args = p.parse_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    run_spec_all_seeds(args.spec, device, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
