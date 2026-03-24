#!/bin/bash
#SBATCH --job-name=har_b010_otrep
#SBATCH --account=def-mcapretz
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/logs/p1_har_bias010_otrep_5s.out
#SBATCH --error=experiment_specs/logs/p1_har_bias010_otrep_5s.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

mkdir -p experiment_specs/logs

python -u run_baseline.py \
    --spec experiment_specs/p1_har_bias010_otrep_5s.json \
    --device cpu
