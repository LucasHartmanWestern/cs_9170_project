#!/bin/bash
#SBATCH --job-name=adapt_rl
#SBATCH --account=def-mcapretz
#SBATCH --time=52:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                 
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

set -euo pipefail

mkdir -p logs

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

module purge

# ---- load modules ----
module load python/3.12.4 cuda cudnn

# ---- Activate environment----
source ~/envs/rl/bin/activate

python -V
nvidia-smi || true

# Run your main program (whatever file contains the __main__ block above)
# Using srun is recommended on DRAC
srun python -u training.py
