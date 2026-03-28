#!/bin/bash
#SBATCH --job-name=v2_comp_smote
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=paper_specs_v2/logs/v2_compas_smote_5s.out
#SBATCH --error=paper_specs_v2/logs/v2_compas_smote_5s.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p paper_specs_v2/logs

python -u run_baseline.py --spec paper_specs_v2/v2_compas_smote_5s.json --device cpu
