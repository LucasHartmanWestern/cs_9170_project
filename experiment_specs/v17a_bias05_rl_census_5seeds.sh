#!/bin/bash
#SBATCH --job-name=v17a_b05_5s
#SBATCH --account=def-mcapretz
#SBATCH --time=05:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/v17a_bias05_rl_census_5seeds.out
#SBATCH --error=experiment_specs/logs/v17a_bias05_rl_census_5seeds.err

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

python -u main.py --spec experiment_specs/v17a_bias05_rl_census_5seeds.json --device cuda:0
