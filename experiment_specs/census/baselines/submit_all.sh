#!/bin/bash
cd "$(dirname "$0")/../../.."
sbatch experiment_specs/census/baselines/group_dro.sh
sbatch experiment_specs/census/baselines/gaussian_ot_repair.sh
sbatch experiment_specs/census/baselines/ctgan.sh
sbatch experiment_specs/census/baselines/fairness_loss_balancing.sh
sbatch experiment_specs/census/baselines/smote.sh
sbatch experiment_specs/census/baselines/fairtabddpm.sh
