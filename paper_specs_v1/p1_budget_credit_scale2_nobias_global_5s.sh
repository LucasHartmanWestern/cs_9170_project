#!/bin/bash
#SBATCH --job-name=p1_budget_credit_scale2_
#SBATCH --account=def-mcapretz
#SBATCH --time=20:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=paper_specs_v1/logs/p1_budget_credit_scale2_nobias_global_5s.out
#SBATCH --error=paper_specs_v1/logs/p1_budget_credit_scale2_nobias_global_5s.err

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
mkdir -p paper_specs_v1/logs

python -u main.py --spec paper_specs_v1/p1_budget_credit_scale2_nobias_global_5s.json --device cuda:0
