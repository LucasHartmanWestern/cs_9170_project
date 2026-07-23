#!/bin/bash
cd "$(dirname "$0")/../../.."
sbatch experiment_specs/capture24/baselines/group_dro_fold0.sh
sbatch experiment_specs/capture24/baselines/group_dro_fold1.sh
sbatch experiment_specs/capture24/baselines/group_dro_fold2.sh
sbatch experiment_specs/capture24/baselines/group_dro_fold3.sh
sbatch experiment_specs/capture24/baselines/group_dro_fold4.sh
sbatch experiment_specs/capture24/baselines/gaussian_ot_repair_fold0.sh
sbatch experiment_specs/capture24/baselines/gaussian_ot_repair_fold1.sh
sbatch experiment_specs/capture24/baselines/gaussian_ot_repair_fold2.sh
sbatch experiment_specs/capture24/baselines/gaussian_ot_repair_fold3.sh
sbatch experiment_specs/capture24/baselines/gaussian_ot_repair_fold4.sh
sbatch experiment_specs/capture24/baselines/fairness_loss_balancing_fold0.sh
sbatch experiment_specs/capture24/baselines/fairness_loss_balancing_fold1.sh
sbatch experiment_specs/capture24/baselines/fairness_loss_balancing_fold2.sh
sbatch experiment_specs/capture24/baselines/fairness_loss_balancing_fold3.sh
sbatch experiment_specs/capture24/baselines/fairness_loss_balancing_fold4.sh
sbatch experiment_specs/capture24/baselines/fairtabddpm_fold0.sh
sbatch experiment_specs/capture24/baselines/fairtabddpm_fold1.sh
sbatch experiment_specs/capture24/baselines/fairtabddpm_fold2.sh
sbatch experiment_specs/capture24/baselines/fairtabddpm_fold3.sh
sbatch experiment_specs/capture24/baselines/fairtabddpm_fold4.sh
