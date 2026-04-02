#!/bin/bash
#SBATCH --job-name=ptbxl_rl15
#SBATCH --account=rrg-kgroling
#SBATCH --time=20:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/ptb_xl_age_rl_1500ep_5s.out
#SBATCH --error=experiment_specs/logs/ptb_xl_age_rl_1500ep_5s.err

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

python -u main.py --spec experiment_specs/ptb_xl_age_rl_1500ep_5s.json --device cuda:0
