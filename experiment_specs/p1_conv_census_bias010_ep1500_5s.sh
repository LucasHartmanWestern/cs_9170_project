#!/bin/bash
#SBATCH --job-name=conv_census_ep1500
#SBATCH --account=def-mcapretz
#SBATCH --time=10:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/p1_conv_census_bias010_ep1500_5s.out
#SBATCH --error=experiment_specs/logs/p1_conv_census_bias010_ep1500_5s.err

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

python -u main.py --spec experiment_specs/p1_conv_census_bias010_ep1500_5s.json --device cuda:0
