#!/bin/bash
# Pipeline per GPU:
#   GPU0: census seed_2 (running) -> ACS seeds 0,1,42 -> capture24 seed_2
#   GPU1: census seed_3 (running) -> ACS seeds 2,3   -> capture24 seed_3
# Census PIDs: seed_2=53033 (cuda:0), seed_3=53034 (cuda:1)
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/Experiment3
LOGS=$SPECS/logs
mkdir -p "$LOGS"

# --- GPU 0 pipeline ---
gpu0_pipeline() {
    echo "[$(date)] [GPU0] Waiting for census seed_2 (PID 53033)..."
    tail --pid=53033 -f /dev/null 2>/dev/null || true

    echo "[$(date)] [GPU0] Launching ACS Employment seeds 0,1,42 on cuda:0"
    python -u main.py \
        --spec $SPECS/acs_forge_gpu0.yaml \
        --device cuda:0 \
        2>&1 | tee "$LOGS/acs_forge_gpu0.log"

    echo "[$(date)] [GPU0] ACS done. Launching capture24 seed_2 on cuda:0"
    python -u main.py \
        --spec $SPECS/capture24_forge_seed2.yaml \
        --device cuda:0 \
        2>&1 | tee "$LOGS/capture24_forge_seed2.log"

    echo "[$(date)] [GPU0] Pipeline complete."
}

# --- GPU 1 pipeline ---
gpu1_pipeline() {
    echo "[$(date)] [GPU1] Waiting for census seed_3 (PID 53034)..."
    tail --pid=53034 -f /dev/null 2>/dev/null || true

    echo "[$(date)] [GPU1] Launching ACS Employment seeds 2,3 on cuda:1"
    python -u main.py \
        --spec $SPECS/acs_forge_gpu1.yaml \
        --device cuda:1 \
        2>&1 | tee "$LOGS/acs_forge_gpu1.log"

    echo "[$(date)] [GPU1] ACS done. Launching capture24 seed_3 on cuda:1"
    python -u main.py \
        --spec $SPECS/capture24_forge_seed3.yaml \
        --device cuda:1 \
        2>&1 | tee "$LOGS/capture24_forge_seed3.log"

    echo "[$(date)] [GPU1] Pipeline complete."
}

gpu0_pipeline &
gpu1_pipeline &
wait
echo "[$(date)] All pipelines complete."
