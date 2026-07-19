#!/bin/bash
# Submit all SLURM jobs for this search
cd "$(dirname "$0")/../.."
sbatch experiment_specs/cal_census/bundle_0.sh
