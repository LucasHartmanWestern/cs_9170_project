#!/bin/bash
# EXP-046 DRAC grid — 15 FORGE jobs
# Run from ~/cs_9170_project on DRAC
cd ~/cs_9170_project

sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep10_fold0.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep10_fold1.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep10_fold2.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep10_fold3.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep10_fold4.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep20_fold0.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep20_fold1.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep20_fold2.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep20_fold3.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k3_ep20_fold4.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k5_ep10_fold3.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k5_ep10_fold4.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k5_ep20_fold2.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k5_ep20_fold3.sh
sbatch experiment_specs/c24_grid_search/drac/forge_k5_ep20_fold4.sh
