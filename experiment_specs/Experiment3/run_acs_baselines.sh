#!/bin/bash
# Run all 6 ACS Employment baselines after the FORGE runs finish.
# Usage: bash experiment_specs/Experiment3/run_acs_baselines.sh
# Pass --nowait to skip waiting for FORGE PIDs and launch immediately.
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/Experiment3/baselines
LOGS=$SPECS/logs
mkdir -p "$LOGS"

FORGE_PID_GPU0=56219
FORGE_PID_GPU1=56221

if [[ "$1" != "--nowait" ]]; then
    echo "[$(date)] Waiting for FORGE GPU0 (PID $FORGE_PID_GPU0)..."
    tail --pid=$FORGE_PID_GPU0 -f /dev/null 2>/dev/null || true
    echo "[$(date)] Waiting for FORGE GPU1 (PID $FORGE_PID_GPU1)..."
    tail --pid=$FORGE_PID_GPU1 -f /dev/null 2>/dev/null || true
    echo "[$(date)] FORGE runs complete. Launching baselines."
fi

run_baseline() {
    local spec=$1
    local device=$2
    local name=$(basename $spec .yaml)
    echo "[$(date)] Starting $name on $device"
    python -u run_baseline.py --spec $spec --device $device \
        2>&1 | tee "$LOGS/${name}.log"
    echo "[$(date)] Done: $name"
}

# Light baselines on GPU0, heavy generative ones on GPU1
run_baseline $SPECS/acs_gdro.yaml      cuda:0 &
run_baseline $SPECS/acs_flb.yaml       cuda:0 &
run_baseline $SPECS/acs_smote.yaml     cuda:0 &
run_baseline $SPECS/acs_ot_repair.yaml cuda:0 &
run_baseline $SPECS/acs_ctgan.yaml         cuda:1 &
run_baseline $SPECS/acs_fairtabddpm.yaml   cuda:1 &

wait
echo "[$(date)] All ACS baselines complete."
