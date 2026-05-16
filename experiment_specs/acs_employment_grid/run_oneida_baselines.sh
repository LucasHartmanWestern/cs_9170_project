#!/bin/bash
# Oneida — EXP-028: ACS Employment baselines (natural scarcity)
# Run AFTER EXP-027 (run_oneida.sh) completes, or in parallel if both GPUs are available.
# GPU 0: gdro, flb, ot_repair
# GPU 1: smote, ctgan, fairtabddpm
# Both GPU streams run sequentially within each GPU to avoid memory contention.
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/acs_employment_grid/baselines
LOGS=experiment_specs/acs_employment_grid/logs
mkdir -p "$LOGS"

echo "[$(date)] === EXP-028: ACS Employment baselines ==="

# GPU 0 stream
run_gpu0() {
    for method in gdro flb ot_repair; do
        echo "[$(date)] GPU 0: Starting acs_$method"
        python -u run_baseline.py \
            --spec $SPECS/acs_emp_${method}.yaml \
            --device cuda:0 \
            2>&1 | tee -a "$LOGS/acs_${method}_gpu0.log"
        echo "[$(date)] GPU 0: Finished acs_$method"
    done
}

# GPU 1 stream
run_gpu1() {
    for method in smote ctgan fairtabddpm; do
        echo "[$(date)] GPU 1: Starting acs_$method"
        python -u run_baseline.py \
            --spec $SPECS/acs_emp_${method}.yaml \
            --device cuda:1 \
            2>&1 | tee -a "$LOGS/acs_${method}_gpu1.log"
        echo "[$(date)] GPU 1: Finished acs_$method"
    done
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait $PID0
echo "[$(date)] GPU 0 stream complete."
wait $PID1
echo "[$(date)] GPU 1 stream complete."

echo "[$(date)] EXP-028 complete. Run check_run.py / plot_results.ipynb."
