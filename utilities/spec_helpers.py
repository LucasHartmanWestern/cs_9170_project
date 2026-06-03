import os
import re
import json
import hashlib
from datetime import datetime
import yaml


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def _load_spec(path: str) -> dict:
    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    validate_spec(spec, path)
    return spec


def _spec_id(spec_path: str, spec: dict) -> str:
    base = os.path.splitext(os.path.basename(spec_path))[0]
    h    = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{_slug(base)}_{h}"


def build_exp_group(spec_path: str, spec: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M")
    return f"SPEC_{_spec_id(spec_path, spec)}__G{ts}"


_KNOWN_TOP_LEVEL = {
    # dataset
    "dataset_name", "minority_id", "majority_id", "da_pct", "dp_protected_col",
    # training dimensions
    "pca_components", "traj_length", "real_data_size", "total_episodes",
    "total_data_size", "seeds",
    # reward
    "lambda_schedule", "reward_shaping",
    # env
    "use_pca", "use_delta_actions", "delta_scale", "delta_clip",
    "pca_clip", "radius_clip",
    # agents
    "ffnn", "reinforce",
    # k-fold
    "fold_idx", "n_folds", "fold_rng_seed", "kfold_val_frac",
    # misc
    "beta_reset_interval", "global_sigmoid_k", "epochs",
    "max_parallel", "output_dir", "permutations",
    # baseline runner
    "baseline", "eo_guard_threshold",
    "group_dro", "ot_repair", "ctgan", "flb", "smote", "fairtabddpm",
}

_KNOWN_REWARD_SHAPING = {
    "global_sigmoid_k",
    "utility_guard_min_factor",  # deprecated but accepted; silently ignored
}

_REQUIRED_FLAT = {"dataset_name", "lambda_schedule", "total_episodes",
                  "pca_components", "traj_length", "real_data_size"}
_REQUIRED_PERMUTATIONS = {"dataset_name", "lambda_schedule", "total_episodes",
                           "permutations", "total_data_size"}


def validate_spec(spec: dict, spec_path: str) -> None:
    """Warn on unknown fields and missing required fields."""
    warnings = []

    unknown = set(spec.keys()) - _KNOWN_TOP_LEVEL
    for k in sorted(unknown):
        warnings.append(f"unknown top-level field '{k}' (typo?)")

    for k in sorted(set(spec.get("reward_shaping", {}).keys()) - _KNOWN_REWARD_SHAPING):
        warnings.append(f"unknown reward_shaping field '{k}'")

    required = _REQUIRED_PERMUTATIONS if "permutations" in spec else _REQUIRED_FLAT
    for k in sorted(required - set(spec.keys())):
        warnings.append(f"missing required field '{k}'")

    if warnings:
        print(f"[validate_spec] {spec_path}: {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ⚠  {w}")
    else:
        print(f"[validate_spec] {spec_path}: OK")
