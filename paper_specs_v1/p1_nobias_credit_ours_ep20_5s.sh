#!/bin/bash
#SBATCH --job-name=p1_nobias_credit_ep20
#SBATCH --account=def-mcapretz
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=paper_specs_v1/logs/p1_nobias_credit_ours_ep20_5s.out
#SBATCH --error=paper_specs_v1/logs/p1_nobias_credit_ours_ep20_5s.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

mkdir -p paper_specs_v1/logs

python -u main.py \
    --spec paper_specs_v1/p1_nobias_credit_ours_ep20_5s.json \
    --device cpu
