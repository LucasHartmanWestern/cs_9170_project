#!/bin/bash
#SBATCH --job-name=v3_credit_budget_very_high_ratio
#SBATCH --account=def-mcapretz
#SBATCH --time=36:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/v3_credit_budget_very_high_ratio.out
#SBATCH --error=experiment_specs/logs/v3_credit_budget_very_high_ratio.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

mkdir -p experiment_specs/logs

python -u main.py --spec experiment_specs/v3_credit_budget_very_high_ratio.json --device cuda:0
