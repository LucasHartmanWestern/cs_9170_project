#!/bin/bash
#SBATCH --job-name=v16_gdro_bias05
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/logs/v16_bias05_gdro_census_3seeds.out
#SBATCH --error=experiment_specs/logs/v16_bias05_gdro_census_3seeds.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

mkdir -p experiment_specs/logs

python -u run_baseline.py --spec experiment_specs/v16_bias05_gdro_census_3seeds.json --device cpu
