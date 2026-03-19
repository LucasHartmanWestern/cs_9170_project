#!/usr/bin/env python3
"""
Generates all paper_specs_v1 JSON + SLURM batch files.

Run from the project root: python paper_specs_v1/generate_specs.py

53 specs total:
  Group 1 — Main results          ( 3 specs, census + credit)
  Group 2 — Budget sweep          (36 specs, both datasets × 5 conditions × 2 bias × 2 modes)
  Group 3 — Delta scale sweep     ( 8 specs, both datasets × 2 values × 2 modes)
  Group 4 — FFNN epochs sweep     ( 6 specs, both datasets × 3 values, global-only only)

NOTE: v17a (DVRL) is NOT included in the FFNN epochs sweep by design.
The FFNN epoch count for v17a is fixed at 10 (the default). This choice was
made to limit experiment count; the budget and delta-scale sweeps already
provide paired v18/v17a ablation coverage. If reviewers ask about FFNN
sensitivity for the DVRL variant, extend with additional runs.
"""

import copy
import json
import os
from pathlib import Path

OUT_DIR = Path(__file__).parent
SLURM_ACCOUNT = "def-mcapretz"

# ── Base spec templates ──────────────────────────────────────────────────────

V18_BASE = {
    "dataset_name": "census_income",
    "multiclass": False,
    "bias_pct": 0.10,

    "reward_mode": "fairness",
    "lambda_schedule": [1.0, 1.0],
    "minority_id": 1,
    "majority_id": 0,
    "third_id": None,

    "use_pca": True,
    "pca_components": 10,
    "traj_length": 2000,
    "real_data_size": 3000,
    "total_episodes": 800,
    "phase2_episodes": 200,

    "curriculum_learning": True,
    "use_delta_actions": True,
    "delta_scale": 0.10,
    "delta_clip": 0.20,
    "pca_clip": None,
    "radius_clip": 3.0,

    "gen_both_classes": True,
    "seeds": [42, 0, 1, 2, 3],

    "reward_shaping": {
        "global_sigmoid_k": 10.0,
        "utility_guard_min_factor": 0.0
    },

    "local_weights": {
        "use_dvrl_local": False,
        "dvrl_max_bce": 0.693,
        "w_anchor": 0.0,
        "w_hard": 0.0,
        "w_div": 0.0
    },

    "ffnn": {
        "hidden_sizes": [32, 16],
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10
    },

    "reinforce": {
        "hidden_sizes": [64, 64],
        "lr": 0.0003,
        "gamma": 1.0,
        "entropy_start": 0.02,
        "entropy_end": 0.005
    },

    "curriculum": {
        "start_dim": 10,
        "max_dim_cap": 10,
        "stage_count": 1,
        "schedule": "linear"
    },

    "benchmarks": {
        "run_ctgan": False,
        "run_ctabgan": False
    }
}

# v17a (DVRL) differs only in lambda_schedule and use_dvrl_local
V17A_OVERRIDES = {
    "lambda_schedule": [0.3, 0.5],
    "local_weights": {
        "use_dvrl_local": True,
        "dvrl_max_bce": 0.693,
        "w_anchor": 0.0,
        "w_hard": 0.0,
        "w_div": 0.0
    }
}


def make_v18(overrides: dict) -> dict:
    spec = copy.deepcopy(V18_BASE)
    _apply(spec, overrides)
    return spec


def make_v17a(overrides: dict) -> dict:
    spec = copy.deepcopy(V18_BASE)
    _apply(spec, V17A_OVERRIDES)
    _apply(spec, overrides)
    return spec


def _apply(spec: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(spec.get(k), dict):
            _apply(spec[k], v)
        else:
            spec[k] = v


# ── SLURM time estimation ────────────────────────────────────────────────────
# Baseline observed: 5 seeds × 800 eps × T=2000 ≈ 5h actual.
# We use 8h as a safe default, scaling up for larger budget/epoch runs.

def slurm_time(spec: dict) -> str:
    T      = spec.get("traj_length", 2000)
    real   = spec.get("real_data_size", 3000)
    epochs = spec.get("ffnn", {}).get("epochs", 10)
    seeds  = len(spec.get("seeds", [1]))

    # Approximate relative cost vs baseline (T=2000, real=3000, epochs=10, 5 seeds)
    t_scale     = T / 2000
    real_scale  = (real / 3000) ** 0.5   # real data adds sqrt cost to FFNN training
    epoch_scale = epochs / 10
    seed_scale  = seeds / 5

    cost = t_scale * real_scale * epoch_scale * seed_scale

    base_hours = 5.0   # observed actual for baseline config
    est = base_hours * cost * 1.6   # 60% safety buffer

    if   est <= 5:   return "8:00:00"
    elif est <= 9:   return "12:00:00"
    elif est <= 14:  return "16:00:00"
    else:            return "20:00:00"


# ── File writers ─────────────────────────────────────────────────────────────

def write_spec(name: str, spec: dict) -> None:
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")


def write_sh(name: str, spec: dict) -> None:
    time = slurm_time(spec)
    content = f"""#!/bin/bash
#SBATCH --job-name={name[:24]}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --time={time}
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=paper_specs_v1/logs/{name}.out
#SBATCH --error=paper_specs_v1/logs/{name}.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p paper_specs_v1/logs

python -u main.py --spec paper_specs_v1/{name}.json --device cuda:0
"""
    (OUT_DIR / f"{name}.sh").write_text(content)


def emit(name: str, spec: dict) -> None:
    write_spec(name, spec)
    write_sh(name, spec)
    t = slurm_time(spec)
    print(f"  {name:<52}  time={t}")


# ── Spec definitions ─────────────────────────────────────────────────────────

def main():
    print("\n=== Group 1: Main results (3 specs) ===")

    emit("p1_main_census_bias05_global_5s", make_v18({
        "bias_pct": 0.05
    }))
    emit("p1_main_credit_bias010_global_5s", make_v18({
        "dataset_name": "credit_card"
    }))
    emit("p1_main_credit_bias05_global_5s", make_v18({
        "dataset_name": "credit_card",
        "bias_pct": 0.05
    }))

    # ── Group 2: Budget sweep ────────────────────────────────────────────────
    # Conditions:
    #   curr    : T=2000, real=3000  (current default — only nobias needed)
    #   scale1  : T=3000, real=4500  (same ratio, ×1.5)
    #   scale2  : T=6000, real=9000  (same ratio, ×3)
    #   hisynth : T=6000, real=3000  (2× synth:real ratio)
    #   hireal  : T=2000, real=6000  (0.33× synth:real ratio)
    # Run on both datasets.

    print("\n=== Group 2: Budget sweep (36 specs, both datasets) ===")

    BUDGET_CONDITIONS = {
        "curr":    {"traj_length": 2000, "real_data_size": 3000},
        "scale1":  {"traj_length": 3000, "real_data_size": 4500},
        "scale2":  {"traj_length": 6000, "real_data_size": 9000},
        "hisynth": {"traj_length": 6000, "real_data_size": 3000},
        "hireal":  {"traj_length": 2000, "real_data_size": 6000},
    }

    DATASETS = {
        "census":  "census_income",
        "credit":  "credit_card",
    }

    for ds_tag, ds_name in DATASETS.items():
        for cond_name, budget in BUDGET_CONDITIONS.items():
            bias_settings = [None] if cond_name == "curr" else [0.10, None]
            for bias in bias_settings:
                bias_tag = "bias010" if bias == 0.10 else "nobias"
                bias_override = {"bias_pct": bias}   # None → no bias injection

                emit(
                    f"p1_budget_{ds_tag}_{cond_name}_{bias_tag}_global_5s",
                    make_v18({"dataset_name": ds_name, **budget, **bias_override})
                )
                emit(
                    f"p1_budget_{ds_tag}_{cond_name}_{bias_tag}_dvrl_5s",
                    make_v17a({"dataset_name": ds_name, **budget, **bias_override})
                )

    # ── Group 3: Delta scale sweep ───────────────────────────────────────────
    # Values: 0.05, (0.10 baseline — already run), 0.20
    # delta_clip is set to 2× delta_scale to maintain the same ratio as baseline.
    # Both v18 and v17a, both datasets.

    print("\n=== Group 3: Delta scale sweep (8 specs, both datasets) ===")

    for ds_tag, ds_name in DATASETS.items():
        for delta_scale, delta_clip in [(0.05, 0.10), (0.20, 0.40)]:
            scale_tag = f"ds{int(delta_scale*100):03d}"
            emit(
                f"p1_delta_{ds_tag}_{scale_tag}_bias010_global_5s",
                make_v18({"dataset_name": ds_name,
                          "delta_scale": delta_scale, "delta_clip": delta_clip})
            )
            emit(
                f"p1_delta_{ds_tag}_{scale_tag}_bias010_dvrl_5s",
                make_v17a({"dataset_name": ds_name,
                           "delta_scale": delta_scale, "delta_clip": delta_clip})
            )

    # ── Group 4: FFNN epochs sweep (v18 only, both datasets) ─────────────────
    # v17a is excluded — FFNN epochs fixed at 10 for all v17a runs.
    # See module docstring for rationale.

    print("\n=== Group 4: FFNN epochs sweep (6 specs, global-only, both datasets) ===")

    for ds_tag, ds_name in DATASETS.items():
        for ep in [5, 20, 50]:
            emit(
                f"p1_ffnn_{ds_tag}_ep{ep:02d}_bias010_global_5s",
                make_v18({"dataset_name": ds_name,
                          "ffnn": {"hidden_sizes": [32, 16], "learning_rate": 0.001,
                                   "batch_size": 64, "epochs": ep}})
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    specs = list(OUT_DIR.glob("p1_*.json"))
    print(f"\nTotal specs created: {len(specs)}")
    print(f"Output directory:    {OUT_DIR}")


if __name__ == "__main__":
    main()
