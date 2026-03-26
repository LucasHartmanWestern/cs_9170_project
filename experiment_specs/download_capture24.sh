#!/bin/bash
#SBATCH --job-name=dl_capture24
#SBATCH --account=def-mcapretz
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/logs/download_capture24.out
#SBATCH --error=experiment_specs/logs/download_capture24.err

set -euo pipefail
module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/logs

python -u scripts/download_capture24.py \
    --data-dir datasets/capture24 \
    --n-female 20 \
    --n-male 20 \
    --seed 42
