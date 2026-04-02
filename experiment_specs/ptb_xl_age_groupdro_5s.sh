#!/bin/bash
#SBATCH --job-name=ptbxl_gdro
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --output=experiment_specs/logs/ptb_xl_age_groupdro_5s.out
#SBATCH --error=experiment_specs/logs/ptb_xl_age_groupdro_5s.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u run_baseline.py --spec experiment_specs/ptb_xl_age_groupdro_5s.json --device cpu
