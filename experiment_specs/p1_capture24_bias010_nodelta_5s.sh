#!/bin/bash
#SBATCH --job-name=c24_b010_nodelta
#SBATCH --account=def-mcapretz
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/p1_capture24_bias010_nodelta_5s.out
#SBATCH --error=experiment_specs/logs/p1_capture24_bias010_nodelta_5s.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u main.py --spec experiment_specs/p1_capture24_bias010_nodelta_5s.json --device cuda:0
