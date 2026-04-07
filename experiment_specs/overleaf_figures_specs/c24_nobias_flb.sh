#!/bin/bash
#SBATCH --job-name=c24_nobias_flb
#SBATCH --account=def-mcapretz
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiment_specs/paper_results_v2/logs/c24_nobias_flb.out
#SBATCH --error=experiment_specs/paper_results_v2/logs/c24_nobias_flb.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/paper_results_v2/logs

python -u run_baseline.py --spec experiment_specs/paper_results_v2/c24_nobias_flb.json --device cpu
