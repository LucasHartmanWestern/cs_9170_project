#!/bin/bash
# Submit all SLURM jobs for this search
cd "$(dirname "$0")/../.."
sbatch experiment_specs/census_stage2/grid_global_sigmoid_k10_pca_components10_epochs30_traj_length4500.sh
sbatch experiment_specs/census_stage2/grid_global_sigmoid_k5_pca_components15_epochs30_traj_length4500.sh
sbatch experiment_specs/census_stage2/grid_global_sigmoid_k10_pca_components10_epochs30_traj_length2000.sh
