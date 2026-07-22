#!/bin/bash
#SBATCH --job-name=rand_0008_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=30:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_random/logs/rand_0008_learning_rate0p0026333394617191514_lr0p0003716169539861266_delta_scale0p2063164573184081_optimizersgd_optimizeradamw.out
#SBATCH --error=experiment_specs/census_random/logs/rand_0008_learning_rate0p0026333394617191514_lr0p0003716169539861266_delta_scale0p2063164573184081_optimizersgd_optimizeradamw.err

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
mkdir -p experiment_specs/census_random/logs

python -u main.py --spec experiment_specs/census_random/rand_0008_learning_rate0p0026333394617191514_lr0p0003716169539861266_delta_scale0p2063164573184081_optimizersgd_optimizeradamw.yaml --device cpu
