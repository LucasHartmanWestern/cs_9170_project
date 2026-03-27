#!/bin/bash
#SBATCH --job-name=ptbxl_b005_fairtabddpm
#SBATCH --account=def-mcapretz
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiment_specs/logs/p1_ptbxl_bias004_fairtabddpm_5s.out
#SBATCH --error=experiment_specs/logs/p1_ptbxl_bias004_fairtabddpm_5s.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u run_baseline.py --spec experiment_specs/p1_ptbxl_bias004_fairtabddpm_5s.json --device cpu
