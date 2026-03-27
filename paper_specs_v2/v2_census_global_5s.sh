#!/bin/bash
#SBATCH --job-name=v2_cens_global
#SBATCH --account=def-mcapretz
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=paper_specs_v2/logs/v2_census_global_5s.out
#SBATCH --error=paper_specs_v2/logs/v2_census_global_5s.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p paper_specs_v2/logs

python -u main.py --spec paper_specs_v2/v2_census_global_5s.json --device cpu
