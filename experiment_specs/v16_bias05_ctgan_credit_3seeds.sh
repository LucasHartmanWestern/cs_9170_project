#!/bin/bash
#SBATCH --job-name=v16_ctgan_credit_005
#SBATCH --account=def-mcapretz
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiment_specs/logs/v16_bias05_ctgan_credit_3seeds.out
#SBATCH --error=experiment_specs/logs/v16_bias05_ctgan_credit_3seeds.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u run_baseline.py --spec experiment_specs/v16_bias05_ctgan_credit_3seeds.json --device cpu
