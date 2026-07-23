#!/bin/bash
#SBATCH --job-name=rand_0004_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24_random/logs/rand_0004_learning_rate0p00041715277661223254_lr0p0002881797740567003_delta_scale0p27470957199199836_optimizersgd_optimizeradamw_fold2.out
#SBATCH --error=experiment_specs/capture24_random/logs/rand_0004_learning_rate0p00041715277661223254_lr0p0002881797740567003_delta_scale0p27470957199199836_optimizersgd_optimizeradamw_fold2.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/capture24_random/logs

python -u main.py --spec experiment_specs/capture24_random/rand_0004_learning_rate0p00041715277661223254_lr0p0002881797740567003_delta_scale0p27470957199199836_optimizersgd_optimizeradamw_fold2.yaml --device cpu
