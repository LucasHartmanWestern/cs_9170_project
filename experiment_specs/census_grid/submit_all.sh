#!/bin/bash
# Submit all SLURM jobs for this search
cd "$(dirname "$0")/../.."
sbatch experiment_specs/census_grid/bundle_00.sh
sbatch experiment_specs/census_grid/bundle_01.sh
sbatch experiment_specs/census_grid/bundle_02.sh
sbatch experiment_specs/census_grid/bundle_03.sh
sbatch experiment_specs/census_grid/bundle_04.sh
sbatch experiment_specs/census_grid/bundle_05.sh
sbatch experiment_specs/census_grid/bundle_06.sh
sbatch experiment_specs/census_grid/bundle_07.sh
sbatch experiment_specs/census_grid/bundle_08.sh
sbatch experiment_specs/census_grid/bundle_09.sh
sbatch experiment_specs/census_grid/bundle_10.sh
sbatch experiment_specs/census_grid/bundle_11.sh
sbatch experiment_specs/census_grid/bundle_12.sh
sbatch experiment_specs/census_grid/bundle_13.sh
sbatch experiment_specs/census_grid/bundle_14.sh
sbatch experiment_specs/census_grid/bundle_15.sh
sbatch experiment_specs/census_grid/bundle_16.sh
sbatch experiment_specs/census_grid/bundle_17.sh
sbatch experiment_specs/census_grid/bundle_18.sh
sbatch experiment_specs/census_grid/bundle_19.sh
sbatch experiment_specs/census_grid/bundle_20.sh
sbatch experiment_specs/census_grid/bundle_21.sh
sbatch experiment_specs/census_grid/bundle_22.sh
sbatch experiment_specs/census_grid/bundle_23.sh
sbatch experiment_specs/census_grid/bundle_24.sh
sbatch experiment_specs/census_grid/bundle_25.sh
sbatch experiment_specs/census_grid/bundle_26.sh
