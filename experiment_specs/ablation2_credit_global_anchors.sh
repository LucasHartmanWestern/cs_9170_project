#!/bin/bash
#SBATCH --job-name=ablation2_credit_global_anchors
#SBATCH --account=def-mcapretz
#SBATCH --time=18:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/ablation2_credit_global_anchors.out
#SBATCH --error=experiment_specs/logs/ablation2_credit_global_anchors.err

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

python -u main.py --spec experiment_specs/ablation2_credit_global_anchors.json --device cuda:0
