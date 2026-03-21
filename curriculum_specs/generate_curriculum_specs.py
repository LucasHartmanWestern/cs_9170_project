#!/usr/bin/env python3
"""
Generates curriculum learning smoke-test specs.

Run from project root: python3 curriculum_specs/generate_curriculum_specs.py

4 specs — v18 base (global-only, gamma=1.0, census, bias=0.10), 200 episodes, seed=42.
Only curriculum config varies. Designed to run locally before committing to full SLURM runs.

Curriculum configs tested:
  control    : start_dim=10, stage_count=1  → effectively no curriculum (current default)
  gentle     : start_dim=5,  stage_count=2  → single mid-point expansion (10D by ep ~100)
  standard   : start_dim=2,  stage_count=5  → v16-style, expands from 2D→10D in 5 stages
  aggressive : start_dim=1,  stage_count=10 → starts in 1D, expands one dim every ~20 eps

Background: The v16→v17a improvement disabled curriculum but simultaneously changed gamma
(0.99→1.0), so the two effects were never separated. These smokes tests isolate curriculum
at gamma=1.0 to get a clean signal.
"""

import copy, json
from pathlib import Path

OUT_DIR = Path(__file__).parent

V18_SMOKE_BASE = {
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
    "total_episodes": 200,
    "phase2_episodes": 50,

    "curriculum_learning": True,
    "use_delta_actions": True,
    "delta_scale": 0.10,
    "delta_clip": 0.20,
    "pca_clip": None,
    "radius_clip": 3.0,

    "gen_both_classes": True,
    "seeds": [42],

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

CONFIGS = [
    # (name,          start_dim, stage_count, description)
    ("curr_control",    10, 1,  "no curriculum — current default"),
    ("curr_gentle",      5, 2,  "single mid-point expansion: 5D → 10D at ep ~100"),
    ("curr_standard",    2, 5,  "v16-style: 2D → 10D across 5 stages"),
    ("curr_aggressive",  1, 10, "1D start: expands one dim every ~20 eps"),
]


def make_spec(start_dim, stage_count):
    spec = copy.deepcopy(V18_SMOKE_BASE)
    spec["curriculum"]["start_dim"]   = start_dim
    spec["curriculum"]["stage_count"] = stage_count
    return spec


def main():
    print("Generating curriculum smoke-test specs...\n")
    for name, start_dim, stage_count, desc in CONFIGS:
        spec = make_spec(start_dim, stage_count)
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"  {name:<20}  start_dim={start_dim:<2}  stages={stage_count:<2}  # {desc}")

    # ── run_all.sh ─────────────────────────────────────────────────────────
    run_script = OUT_DIR / "run_all.sh"
    lines = ["#!/bin/bash", "# Run all curriculum smoke tests sequentially (local).",
             "# Usage: bash curriculum_specs/run_all.sh", "set -euo pipefail", ""]
    for name, _, _, _ in CONFIGS:
        lines.append(f'echo "=== {name} ==="')
        lines.append(f"python3 -u main.py --spec curriculum_specs/{name}.json --device cuda:0")
        lines.append("")
    run_script.write_text("\n".join(lines))
    run_script.chmod(0o755)

    print(f"\nTotal specs: {len(CONFIGS)}")
    print(f"To run:  bash curriculum_specs/run_all.sh")
    print(f"To analyse: python3 curriculum_specs/analyze_curriculum.py")


if __name__ == "__main__":
    main()
