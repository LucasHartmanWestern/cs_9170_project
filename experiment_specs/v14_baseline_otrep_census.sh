#!/bin/bash
#SBATCH --job-name=v14_otrep_census
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/logs/v14_baseline_otrep_census.out
#SBATCH --error=experiment_specs/logs/v14_baseline_otrep_census.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

mkdir -p experiment_specs/logs

python -u run_baseline.py --spec experiment_specs/v14_baseline_otrep_census.json --device cpu
