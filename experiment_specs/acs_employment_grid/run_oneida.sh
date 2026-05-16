#!/bin/bash
# Oneida — EXP-027: ACS Employment (natural scarcity, disability protected attribute)
# Run AFTER Experiment3 GPU runs complete.
# GPU 0: k=5, ep=30 (census best config)   → 3 seeds × ~5000 ep
# GPU 1: k=3, ep=10 (capture24 best config) → 3 seeds × ~5000 ep
# Start both GPUs in parallel; script blocks until both finish.
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/acs_employment_grid
LOGS=$SPECS/logs
mkdir -p "$LOGS"

echo "[$(date)] === EXP-027: ACS Employment natural scarcity ==="
echo "[$(date)] Launching GPU 0: k=5, ep=30"
nohup python -u main.py \
    --spec $SPECS/acs_emp_k5_ep30_natural.yaml \
    --device cuda:0 \
    > "$LOGS/acs_k5_ep30_natural_gpu0.log" 2>&1 &
PID_GPU0=$!
echo "[$(date)] GPU 0 PID: $PID_GPU0"

echo "[$(date)] Launching GPU 1: k=3, ep=10"
nohup python -u main.py \
    --spec $SPECS/acs_emp_k3_ep10_natural.yaml \
    --device cuda:1 \
    > "$LOGS/acs_k3_ep10_natural_gpu1.log" 2>&1 &
PID_GPU1=$!
echo "[$(date)] GPU 1 PID: $PID_GPU1"

echo "[$(date)] Both configs running. Waiting..."
wait $PID_GPU0
echo "[$(date)] GPU 0 (k=5 ep=30) done."
wait $PID_GPU1
echo "[$(date)] GPU 1 (k=3 ep=10) done."

echo "[$(date)] EXP-027 complete. Run check_run.py on each output directory."
