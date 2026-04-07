#!/bin/bash
#SBATCH --job-name=census_ep1500_nophase2
#SBATCH --account=def-mcapretz
#SBATCH --time=12:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/April_7_Experiments/logs/census_ep1500_nophase2.out
#SBATCH --error=experiment_specs/April_7_Experiments/logs/census_ep1500_nophase2.err

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

python -u main.py --spec experiment_specs/April_7_Experiments/census_ep1500_nophase2.json --device cuda:0
