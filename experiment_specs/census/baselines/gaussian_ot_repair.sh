#!/bin/bash
#SBATCH --job-name=bl_gaussian_ot_
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census/baselines/logs/gaussian_ot_repair.out
#SBATCH --error=experiment_specs/census/baselines/logs/gaussian_ot_repair.err

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
mkdir -p experiment_specs/census/baselines/logs
python -u run_baseline.py --spec experiment_specs/census/baselines/gaussian_ot_repair.yaml --device cpu
