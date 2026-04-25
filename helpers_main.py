import os
import json
import hashlib
from datetime import datetime
import yaml


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")

def _load_spec(path: str) -> dict:
    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    validate_spec(spec, path)
    return spec

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
    "pool_pos_fraction", "permutations", "total_data_size",
    "global_sigmoid_k", "epochs", "max_parallel",
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
# Fields required regardless of format
_REQUIRED_TOP_LEVEL = {
    "dataset_name", "reward_mode", "lambda_schedule", "total_episodes",
}
# Additional fields required for old flat format (no permutations block)
_REQUIRED_FLAT_FORMAT = {
    "pca_components", "traj_length", "real_data_size",
}
# Additional fields required for new permutations format
_REQUIRED_PERMUTATIONS_FORMAT = {
    "permutations", "total_data_size",
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

    # Required fields — base set always checked
    for k in sorted(_REQUIRED_TOP_LEVEL):
        if k not in spec:
            warnings.append(f"Missing required field: '{k}'")
    # Format-specific required fields
    if "permutations" in spec:
        for k in sorted(_REQUIRED_PERMUTATIONS_FORMAT):
            if k not in spec:
                warnings.append(f"Missing required field: '{k}'")
    else:
        for k in sorted(_REQUIRED_FLAT_FORMAT):
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