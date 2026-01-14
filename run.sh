#!/bin/bash
#SBATCH --job-name=adapt_rl_fair
#SBATCH --account=def-mcapretz
#SBATCH --time=18:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

SPEC_DIR="experiment_specs"
SPECS=(
  "${SPEC_DIR}/creditLambda06.json"
  "${SPEC_DIR}/creditLambda08.json"
  "${SPEC_DIR}/creditLambda04085.json"
)

mkdir -p "${SPEC_DIR}/logs"

launch () {
  local SPEC="$1"
  local NAME
  NAME="$(basename "$SPEC" .json)"
  srun --overlap -N1 -n1 -c2 \
    --output="${SPEC_DIR}/logs/${NAME}.out" \
    --error="${SPEC_DIR}/logs/${NAME}.err" \
    python -u main.py --spec "$SPEC" --device cuda:0 &
  echo $!
}

PIDS=()
for SPEC in "${SPECS[@]}"; do
  PIDS+=("$(launch "$SPEC")")
done

wait "${PIDS[@]}"
echo "Done."
