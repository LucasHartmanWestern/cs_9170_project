#!/usr/bin/env python3
"""
Generates no-PCA smoke-test specs to investigate whether:
  1. The agent can learn at all in raw feature space (no-PCA control)
  2. Curriculum helps/hurts with arbitrary feature ordering (no PCA ordering)
  3. A larger RL network helps in the high-D action space (census ~100D)

Run from project root: python3 curriculum_specs/generate_nopca_specs.py

6 specs — census, bias=0.10, 200 episodes, seed=42.

Background:
  With use_pca=False, pca_components is overridden to the actual feature count at
  runtime (~100 for census after one-hot encoding). The curriculum start_dim=2
  then means the first 2 raw feature columns (age, fnlwgt for census) — an
  arbitrary ordering with no principled importance ranking, unlike PCA which
  orders by explained variance. This is expected to hurt curriculum.

  Contrast with PCA curriculum where start_dim=2 = the two most informative
  directions in the data. The hypothesis: curriculum benefit disappears without
  PCA's principled feature ordering.
"""

import copy, json
from pathlib import Path

OUT_DIR = Path(__file__).parent

NOPCA_SMOKE_BASE = {
    "dataset_name": "census_income",
    "multiclass": False,
    "bias_pct": 0.10,

    "reward_mode": "fairness",
    "lambda_schedule": [1.0, 1.0],
    "minority_id": 1,
    "majority_id": 0,
    "third_id": None,

    "use_pca": False,
    "pca_components": 1,           # placeholder — overridden at runtime to ~100

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
        "start_dim": 1,            # overridden per spec
        "max_dim_cap": 999,        # effectively no cap — uses actual feature dim
        "stage_count": 1,
        "schedule": "linear"
    },
    "benchmarks": {
        "run_ctgan": False,
        "run_ctabgan": False
    }
}

CONFIGS = [
    # (name,                    start_dim, stage_count, hidden_sizes,  description)
    # --- no-curriculum baseline in raw feature space ---
    ("nopca_control",           999, 1,  [64, 64],   "no curriculum, default network [64,64]"),
    # --- curriculum with arbitrary (raw feature column) ordering ---
    ("nopca_curr_standard",     2,   5,  [64, 64],   "curriculum start_dim=2 (arbitrary ordering), [64,64]"),
    ("nopca_curr_gentle",       5,   2,  [64, 64],   "curriculum start_dim=5 (arbitrary ordering), [64,64]"),
    # --- larger RL network to handle ~100D action space ---
    ("nopca_large_net_control", 999, 1,  [256, 256], "no curriculum, larger network [256,256]"),
    ("nopca_large_net_curr",    2,   5,  [256, 256], "curriculum start_dim=2, larger network [256,256]"),
    # --- credit card: ~29D, smaller problem, check if no-pca works well here ---
    ("nopca_credit_control",    999, 1,  [64, 64],   "credit, no curriculum, [64,64] (~29D action space)"),
]


def make_spec(start_dim, stage_count, hidden_sizes, dataset="census_income"):
    spec = copy.deepcopy(NOPCA_SMOKE_BASE)
    # start_dim=999 is our sentinel for "no curriculum" — set to max_dim_cap
    spec["curriculum"]["start_dim"]   = start_dim
    spec["curriculum"]["stage_count"] = stage_count
    spec["reinforce"]["hidden_sizes"] = hidden_sizes
    spec["dataset_name"] = dataset
    return spec


def main():
    print("Generating no-PCA smoke-test specs...\n")
    for name, start_dim, stage_count, hidden, desc in CONFIGS:
        dataset = "credit_card" if "credit" in name else "census_income"
        spec = make_spec(start_dim, stage_count, hidden, dataset)
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"  {name:<30}  net={str(hidden):<12}  start_dim={start_dim:<4}  stages={stage_count}  # {desc}")

    # run script
    run_script = OUT_DIR / "run_nopca.sh"
    lines = [
        "#!/bin/bash",
        "# Run all no-PCA smoke tests. Census runs on cuda:0, credit on cuda:1.",
        "# Usage: bash curriculum_specs/run_nopca.sh",
        "set -euo pipefail",
        "source ~/envs/rl/bin/activate",
        "",
    ]
    cuda0 = [n for n, *_ in CONFIGS if "credit" not in n]
    cuda1 = [n for n, *_ in CONFIGS if "credit" in n]

    lines.append("# Census specs on cuda:0")
    for name in cuda0:
        lines.append(f'echo "=== {name} ===" && python3 -u main.py --spec curriculum_specs/{name}.json --device cuda:0 \\'
                     f'\n    > curriculum_specs/logs/{name}.log 2>&1 &')
    lines.append("")
    lines.append("# Credit spec on cuda:1")
    for name in cuda1:
        lines.append(f'echo "=== {name} ===" && python3 -u main.py --spec curriculum_specs/{name}.json --device cuda:1 \\'
                     f'\n    > curriculum_specs/logs/{name}.log 2>&1 &')
    lines += ["", "wait", 'echo "All done"']
    run_script.write_text("\n".join(lines))
    run_script.chmod(0o755)

    print(f"\nTotal specs: {len(CONFIGS)}")
    print(f"To run:     bash curriculum_specs/run_nopca.sh")
    print(f"To analyse: python3 curriculum_specs/analyze_nopca.py")


if __name__ == "__main__":
    main()
