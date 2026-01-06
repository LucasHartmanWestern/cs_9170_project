#!/bin/bash
#SBATCH --job-name=adapt_rl_pack
#SBATCH --account=def-mcapretz
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Prevent CPU oversubscription when running multiple Python processes
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

python -V
nvidia-smi || true

# --------------------------------------------------------------------
# CONCURRENT RUNS ON A SINGLE GPU
# Each main.py call runs: 1 spec file × all seeds in that JSON
# Logs go to experiment_specs/ and are named after the spec
# --------------------------------------------------------------------

SPEC_DIR="experiment_specs"

SPEC_A="${SPEC_DIR}/test_spec_A.json"
SPEC_B="${SPEC_DIR}/test_spec_B.json"
SPEC_C="${SPEC_DIR}/test_spec_C.json"

mkdir -p "${SPEC_DIR}/logs"

launch () {
  local SPEC="$1"
  local NAME
  NAME="$(basename "$SPEC" .json)"

  local OUT="experiment_specs/logs/${NAME}.out"
  local ERR="experiment_specs/logs/${NAME}.err"

  echo "[batch] Launching ${SPEC} -> ${OUT} / ${ERR}"

  srun --exclusive -N1 -n1 -c2 \
    python -u main.py --spec "$SPEC" --device cuda:0 \
    1> "$OUT" 2> "$ERR" &
  echo $!
}

PID1=$(launch "$SPEC_A")
PID2=$(launch "$SPEC_B")
PID3=$(launch "$SPEC_C")

wait $PID1 $PID2 $PID3

echo "Job finished: $(date)"
