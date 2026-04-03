#!/bin/bash
#SBATCH --job-name=acs_rl1500
#SBATCH --account=def-mcapretz
#SBATCH --time=14:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/acs_income_rl_1500ep_5s.out
#SBATCH --error=experiment_specs/logs/acs_income_rl_1500ep_5s.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u main.py --spec experiment_specs/acs_income_rl_1500ep_5s.json --device cuda:0
