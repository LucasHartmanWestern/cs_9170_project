#!/bin/bash
# Submit all SLURM jobs for this search
cd "$(dirname "$0")/../.."
sbatch experiment_specs/cal_census_solo/grid_pca_components10_epochs30.sh
