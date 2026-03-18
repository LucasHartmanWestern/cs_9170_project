#!/bin/bash
#SBATCH --job-name=v17c_b010_ds020
#SBATCH --account=def-mcapretz
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/v17c_bias010_ds020_rl_census_3seeds.out
#SBATCH --error=experiment_specs/logs/v17c_bias010_ds020_rl_census_3seeds.err

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

python -u main.py --spec experiment_specs/v17c_bias010_ds020_rl_census_3seeds.json --device cuda:0
