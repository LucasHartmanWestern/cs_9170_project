#!/bin/bash
#SBATCH --job-name=census_nosigmoid
#SBATCH --account=def-mcapretz
#SBATCH --time=20:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/April_7_Experiments/logs/census_nosigmoid_3s.out
#SBATCH --error=experiment_specs/April_7_Experiments/logs/census_nosigmoid_3s.err

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
mkdir -p experiment_specs/April_7_Experiments/logs

python -u main.py --spec experiment_specs/April_7_Experiments/census_nosigmoid_3s.json --device cuda:0
