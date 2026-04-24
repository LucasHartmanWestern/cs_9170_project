#!/bin/bash
#SBATCH --job-name=bundle_test
#SBATCH --account=rrg-kgroling
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/bundle_test/logs/bundle_test.out
#SBATCH --error=experiment_specs/bundle_test/logs/bundle_test.err

set -uo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/bundle_test/logs

SPECS=(
  experiment_specs/bundle_test/test_k1.yaml\1
  experiment_specs/bundle_test/test_k3.yaml\1
  experiment_specs/bundle_test/test_k5.yaml\1
  experiment_specs/bundle_test/test_k10.yaml\1
)

PIDS=()
for spec in "${SPECS[@]}"; do
  name=$(basename "$spec" .yaml\1)
  python -u main.py --spec "$spec" --device cuda:0 \
    > experiment_specs/bundle_test/logs/${name}.out \
    2> experiment_specs/bundle_test/logs/${name}.err &
  PIDS+=($!)
  echo "[bundle_test] Launched $name (PID $!)"
done

echo "[bundle_test] Waiting for all 4 specs..."
PASS=0
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  code=$?
  name=$(basename "${SPECS[$i]}" .yaml\1)
  if [ $code -eq 0 ]; then
    echo "[bundle_test] PASS: $name (exit 0)"
    PASS=$((PASS + 1))
  else
    echo "[bundle_test] FAIL: $name (exit $code)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=== Bundle test result: $PASS/4 passed, $FAIL/4 failed ==="
[ $FAIL -eq 0 ] && exit 0 || exit 1
