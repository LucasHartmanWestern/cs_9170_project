#!/bin/bash
#SBATCH --job-name=v2_cens_ftddpm
#SBATCH --account=def-mcapretz
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=paper_specs_v2/logs/v2_census_fairtabddpm_5s.out
#SBATCH --error=paper_specs_v2/logs/v2_census_fairtabddpm_5s.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p paper_specs_v2/logs

python -u run_baseline.py --spec paper_specs_v2/v2_census_fairtabddpm_5s.json --device cpu
