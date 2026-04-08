#!/bin/bash
#SBATCH --job-name=c24_sex_gdro
#SBATCH --account=def-mcapretz
#SBATCH --time=03:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --output=experiment_specs/capture24_sex_feature/logs/c24_sex_gdro.out
#SBATCH --error=experiment_specs/capture24_sex_feature/logs/c24_sex_gdro.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/capture24_sex_feature/logs

python -u run_baseline.py --spec experiment_specs/capture24_sex_feature/c24_sex_gdro.json --device cpu --output_dir paper_results_v4/training_runs
