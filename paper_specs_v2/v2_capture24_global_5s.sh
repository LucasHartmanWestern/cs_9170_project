#!/bin/bash
#SBATCH --job-name=v2_c24_global
#SBATCH --account=def-mcapretz
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=paper_specs_v2/logs/v2_capture24_global_5s.out
#SBATCH --error=paper_specs_v2/logs/v2_capture24_global_5s.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p paper_specs_v2/logs

python -u main.py --spec paper_specs_v2/v2_capture24_global_5s.json --device cpu
