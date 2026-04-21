"""
Generate experiment specs for hyperparameter search.

Supports two modes (can be combined in one config):
  grid   — cartesian product of principal parameter values
  random — N random draws from secondary parameter distributions

Usage:
  python make_search_specs.py search_configs/census_grid.yaml
  python make_search_specs.py search_configs/census_random.yaml
  python make_search_specs.py search_configs/census_full.yaml  # both modes

Search config format (YAML):

  name: census_grid          # output dir: experiment_specs/census_grid/
  base: vanilla_config.json  # base spec to patch (path relative to project root)

  # Fixed patches applied to every generated spec (dot-notation)
  base_patches:
    dataset_name: census_income
    da_pct: 0.01433
    minority_id: 0
    majority_id: 1
    seeds: [0, 1, 42]
    total_episodes: 5000
    reward_mode: wgl

  # Principal grid: cartesian product of all listed values
  grid:
    reward_shaping.global_sigmoid_k: [1, 3, 5, 10]
    pca_components: [5, 10, 15]
    traj_length: [1000, 2000, 3000]
    ffnn.epochs: [10, 20, 40]

  # Secondary random search
  random:
    n_samples: 30
    seed: 0   # RNG seed for reproducibility
    params:
      ffnn.learning_rate:  {dist: log_uniform, low: 1.0e-4, high: 1.0e-2}
      reinforce.lr:        {dist: log_uniform, low: 1.0e-5, high: 1.0e-3}
      delta_scale:         {dist: uniform,     low: 0.05,   high: 0.30}
      ffnn.optimizer:      {dist: choice, values: [adam, adamw, sgd]}
      reinforce.optimizer: {dist: choice, values: [adam, adamw]}

  # SLURM settings (optional — defaults shown)
  slurm:
    account: def-mcapretz
    time: "16:00:00"
    mem: "6G"
    cpus: 2
    gpu: 1

Supported distributions:
  uniform      — uniform(low, high)
  log_uniform  — exp(uniform(log(low), log(high)))
  choice       — uniform draw from values list
"""

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys

import yaml


# ─── helpers ────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def set_nested(d: dict, dotkey: str, value):
    """Set a value using dot-notation key, coercing strings to int/float/bool where possible."""
    parts = dotkey.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = _coerce(value)

def _coerce(value):
    if not isinstance(value, str):
        return value
    for fn in (int, float):
        try:
            return fn(value)
        except ValueError:
            pass
    if value.lower() == "true":  return True
    if value.lower() == "false": return False
    return value

def apply_patches(spec: dict, patches: dict) -> dict:
    s = copy.deepcopy(spec)
    s.pop("_comment", None)
    for key, val in patches.items():
        set_nested(s, key, val)
    return s

def short_label(key: str, val) -> str:
    k = key.split(".")[-1]
    v = str(val).replace(".", "p").replace("-", "m")
    return f"{k}{v}"

def make_slurm(spec_path: str, name: str, out_dir: str, slurm: dict) -> str:
    account = slurm.get("account", "def-mcapretz")
    time    = slurm.get("time",    "16:00:00")
    mem     = slurm.get("mem",     "6G")
    cpus    = slurm.get("cpus",    2)
    gpu     = slurm.get("gpu",     1)
    log_dir = f"{out_dir}/logs"
    job_name = name[:16]  # SLURM limit
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpu}
#SBATCH --output={log_dir}/{name}.out
#SBATCH --error={log_dir}/{name}.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
export OPENBLAS_NUM_THREADS={cpus}
export NUMEXPR_NUM_THREADS={cpus}

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p {log_dir}

python -u main.py --spec {spec_path} --device cuda:0
"""


# ─── sampling ───────────────────────────────────────────────────────────────

def sample_param(cfg: dict, rng: random.Random) -> object:
    dist = cfg["dist"]
    if dist == "uniform":
        return rng.uniform(cfg["low"], cfg["high"])
    elif dist == "log_uniform":
        lo, hi = math.log(cfg["low"]), math.log(cfg["high"])
        return math.exp(rng.uniform(lo, hi))
    elif dist == "choice":
        return rng.choice(cfg["values"])
    else:
        raise ValueError(f"Unknown distribution: {dist}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate hyperparameter search specs")
    ap.add_argument("config", help="Path to YAML search config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name      = cfg["name"]
    base_path = cfg["base"]
    base_spec = load_json(base_path)
    base_patches = cfg.get("base_patches", {})
    slurm     = cfg.get("slurm", {})

    out_dir = os.path.join("experiment_specs", name)
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, "logs"))

    generated = []

    # ── grid search ───────────────────────────────────────────────────────
    if "grid" in cfg:
        grid = cfg["grid"]

        # Separate zipped groups from independent axes.
        # "zip" key holds a dict of keys that move together (same index).
        # All other keys are swept as a cartesian product.
        zip_groups = []   # list of dicts, one dict per step in the zip
        indep_keys = []
        indep_vals = []

        for k, v in grid.items():
            if k == "zip":
                # v is a dict: {param_key: [v0, v1, ...], ...}
                zip_keys = list(v.keys())
                zip_val_lists = [v[zk] for zk in zip_keys]
                lengths = [len(l) for l in zip_val_lists]
                if len(set(lengths)) != 1:
                    raise ValueError(f"zip lists must all be the same length, got {dict(zip(zip_keys, lengths))}")
                zip_groups = [dict(zip(zip_keys, row)) for row in zip(*zip_val_lists)]
            else:
                indep_keys.append(k)
                indep_vals.append(v)

        # If no zip group, treat as a single empty patch so the loop still runs
        if not zip_groups:
            zip_groups = [{}]

        for zip_patch in zip_groups:
            for combo in itertools.product(*indep_vals):
                patches = {**dict(zip(indep_keys, combo)), **zip_patch}
                spec = apply_patches(base_spec, base_patches)
                spec = apply_patches(spec, patches)

                label = "_".join(short_label(k, v) for k, v in patches.items())
                spec_name = f"grid_{label}"
                spec_path = os.path.join(out_dir, f"{spec_name}.json")
                sh_path   = os.path.join(out_dir, f"{spec_name}.sh")

                save_json(spec, spec_path)
                with open(sh_path, "w") as f:
                    f.write(make_slurm(spec_path, spec_name, out_dir, slurm))

                generated.append(spec_name)

    # ── random search ─────────────────────────────────────────────────────
    if "random" in cfg:
        rand_cfg  = cfg["random"]
        n_samples = rand_cfg["n_samples"]
        rng_seed  = rand_cfg.get("seed", 0)
        rng       = random.Random(rng_seed)
        params    = rand_cfg["params"]

        for i in range(n_samples):
            patches = {k: sample_param(v, rng) for k, v in params.items()}
            spec = apply_patches(base_spec, base_patches)
            spec = apply_patches(spec, patches)

            label = f"{i:04d}_" + "_".join(short_label(k, v) for k, v in patches.items())
            spec_name = f"rand_{label}"
            spec_path = os.path.join(out_dir, f"{spec_name}.json")
            sh_path   = os.path.join(out_dir, f"{spec_name}.sh")

            save_json(spec, spec_path)
            with open(sh_path, "w") as f:
                f.write(make_slurm(spec_path, spec_name, out_dir, slurm))

            generated.append(spec_name)

    print(f"Generated {len(generated)} specs in {out_dir}/")
    print(f"  {sum(1 for s in generated if s.startswith('grid_'))} grid specs")
    print(f"  {sum(1 for s in generated if s.startswith('rand_'))} random specs")

    # Write a manifest for easy reference
    manifest_path = os.path.join(out_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"Search: {name}\n")
        f.write(f"Base:   {base_path}\n")
        f.write(f"Count:  {len(generated)}\n\n")
        for s in generated:
            f.write(s + "\n")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
