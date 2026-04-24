#!/bin/bash
# Submit all SLURM jobs for this search
cd "$(dirname "$0")/../.."
sbatch experiment_specs/census_grid/k0_bundle_1.sh
sbatch experiment_specs/census_grid/k0_bundle_2.sh
sbatch experiment_specs/census_grid/k0_bundle_3.sh
sbatch experiment_specs/census_grid/k0_bundle_4.sh
sbatch experiment_specs/census_grid/k0_bundle_5.sh
sbatch experiment_specs/census_grid/k0_bundle_6.sh
sbatch experiment_specs/census_grid/k0_bundle_7.sh
sbatch experiment_specs/census_grid/k3_bundle_1.sh
sbatch experiment_specs/census_grid/k3_bundle_2.sh
sbatch experiment_specs/census_grid/k3_bundle_3.sh
sbatch experiment_specs/census_grid/k3_bundle_4.sh
sbatch experiment_specs/census_grid/k3_bundle_5.sh
sbatch experiment_specs/census_grid/k3_bundle_6.sh
sbatch experiment_specs/census_grid/k3_bundle_7.sh
sbatch experiment_specs/census_grid/k5_bundle_1.sh
sbatch experiment_specs/census_grid/k5_bundle_2.sh
sbatch experiment_specs/census_grid/k5_bundle_3.sh
sbatch experiment_specs/census_grid/k5_bundle_4.sh
sbatch experiment_specs/census_grid/k5_bundle_5.sh
sbatch experiment_specs/census_grid/k5_bundle_6.sh
sbatch experiment_specs/census_grid/k5_bundle_7.sh
sbatch experiment_specs/census_grid/k10_bundle_1.sh
sbatch experiment_specs/census_grid/k10_bundle_2.sh
sbatch experiment_specs/census_grid/k10_bundle_3.sh
sbatch experiment_specs/census_grid/k10_bundle_4.sh
sbatch experiment_specs/census_grid/k10_bundle_5.sh
sbatch experiment_specs/census_grid/k10_bundle_6.sh
sbatch experiment_specs/census_grid/k10_bundle_7.sh
