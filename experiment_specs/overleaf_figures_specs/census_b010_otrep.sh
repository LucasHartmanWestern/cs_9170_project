#!/bin/bash
#SBATCH --job-name=census_b010_otrep
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/paper_results_v2/logs/census_b010_otrep.out
#SBATCH --error=experiment_specs/paper_results_v2/logs/census_b010_otrep.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/paper_results_v2/logs

python -u run_baseline.py --spec experiment_specs/paper_results_v2/census_b010_otrep.json --device cpu
