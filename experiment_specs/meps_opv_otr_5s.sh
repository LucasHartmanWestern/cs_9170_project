#!/bin/bash
#SBATCH --job-name=meps_opv_otr
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/logs/meps_opv_otr_5s.out
#SBATCH --error=experiment_specs/logs/meps_opv_otr_5s.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u run_baseline.py --spec experiment_specs/meps_opv_otr_5s.json --device cpu
