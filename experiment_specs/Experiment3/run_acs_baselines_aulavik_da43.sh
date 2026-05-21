#!/bin/bash
# ACS Employment baselines at da_pct=0.01433 (DA+=43 injected scarcity) in PCA-10 space.
# Tests whether baselines struggle under the paper's target scarcity regime.
# Usage: bash experiment_specs/Experiment3/run_acs_baselines_aulavik_da43.sh
# Pass --nowait to skip waiting for the PCA run (PID_PCA) to finish first.
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/Experiment3/baselines
LOGS=$SPECS/logs
mkdir -p "$LOGS"

PCA_MASTER_PID=121564

if [[ "$1" != "--nowait" ]]; then
    echo "[$(date)] Waiting for PCA baseline run (PID $PCA_MASTER_PID) to finish..."
    tail --pid=$PCA_MASTER_PID -f /dev/null 2>/dev/null || true
    echo "[$(date)] PCA run complete. Launching da43 baselines."
fi

run_baseline() {
    local spec=$1
    local device=$2
    local name=$(basename $spec .yaml)
    echo "[$(date)] Starting $name on $device"
    python -u run_baseline.py --spec $spec --device $device \
        2>&1 | tee "$LOGS/${name}_aulavik.log"
    echo "[$(date)] Done: $name"
}

run_baseline $SPECS/acs_gdro_da43.yaml      cuda:1 &
run_baseline $SPECS/acs_flb_da43.yaml       cuda:1 &
run_baseline $SPECS/acs_smote_da43.yaml     cuda:1 &
run_baseline $SPECS/acs_ot_repair_da43.yaml cuda:1 &
wait
echo "[$(date)] Light da43 baselines complete. Starting generative baselines."

run_baseline $SPECS/acs_ctgan_da43.yaml       cuda:1
run_baseline $SPECS/acs_fairtabddpm_da43.yaml cuda:1

echo "[$(date)] All ACS da43 baselines complete (Aulavik GPU1)."
