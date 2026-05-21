#!/bin/bash
# FORGE on ACS Employment at da_pct=0.01433 (DA+=43 injected scarcity).
# Queued behind natural-scarcity k=5 runs (PIDs 113761 / 114020).
# Pass --nowait to launch immediately.
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

FORGE_GPU0_PID=113761
FORGE_GPU1_PID=114020
LOGS=experiment_specs/Experiment3/logs
mkdir -p "$LOGS"

if [[ "$1" != "--nowait" ]]; then
    echo "[$(date)] Waiting for natural-scarcity GPU0 run (PID $FORGE_GPU0_PID)..."
    tail --pid=$FORGE_GPU0_PID -f /dev/null 2>/dev/null || true
    echo "[$(date)] Waiting for natural-scarcity GPU1 run (PID $FORGE_GPU1_PID)..."
    tail --pid=$FORGE_GPU1_PID -f /dev/null 2>/dev/null || true
    echo "[$(date)] Natural-scarcity runs complete. Launching da43 FORGE."
fi

nohup python -u main.py \
    --spec experiment_specs/Experiment3/acs_forge_gpu0_da43.yaml \
    --device cuda:0 --parallel \
    > "$LOGS/acs_forge_gpu0_da43.log" 2>&1 &
echo "[$(date)] GPU0 da43 PID=$!"

nohup python -u main.py \
    --spec experiment_specs/Experiment3/acs_forge_gpu1_da43.yaml \
    --device cuda:1 --parallel \
    > "$LOGS/acs_forge_gpu1_da43.log" 2>&1 &
echo "[$(date)] GPU1 da43 PID=$!"

wait
echo "[$(date)] All da43 FORGE runs complete."
