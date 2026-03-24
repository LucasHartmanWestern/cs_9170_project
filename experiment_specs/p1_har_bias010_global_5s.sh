#!/bin/bash
#SBATCH --job-name=har_b010_rl
#SBATCH --account=def-mcapretz
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/p1_har_bias010_global_5s.out
#SBATCH --error=experiment_specs/logs/p1_har_bias010_global_5s.err

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

python -u main.py --spec experiment_specs/p1_har_bias010_global_5s.json --device cuda:0
