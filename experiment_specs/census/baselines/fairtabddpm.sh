#!/bin/bash
#SBATCH --job-name=bl_fairtabddpm
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census/baselines/logs/fairtabddpm.out
#SBATCH --error=experiment_specs/census/baselines/logs/fairtabddpm.err

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
python -u run_baseline.py --spec experiment_specs/census/baselines/fairtabddpm.yaml --device cpu
