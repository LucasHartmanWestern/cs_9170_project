#!/bin/bash
#SBATCH --job-name=rand0014_confirm
#SBATCH --account=def-mcapretz
#SBATCH --time=48:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_random_confirm/logs/rand0014.out
#SBATCH --error=experiment_specs/census_random_confirm/logs/rand0014.err
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
mkdir -p experiment_specs/census_random_confirm/logs
python -u main.py --spec experiment_specs/census_random_confirm/rand0014.yaml --device cpu
