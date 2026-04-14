#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --account=def-mcapretz
#SBATCH --time=00:05:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/April_13_Experiments/logs/smoke_test_5min.out
#SBATCH --error=experiment_specs/April_13_Experiments/logs/smoke_test_5min.err

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
mkdir -p experiment_specs/April_13_Experiments/logs

python -u main.py --spec experiment_specs/April_13_Experiments/smoke_test_5min.json --device cuda:0
