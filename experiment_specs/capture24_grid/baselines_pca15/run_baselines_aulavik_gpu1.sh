#!/bin/bash
# capture24 baselines matched to best FORGE config: pca=15, real=4000, traj=1000, da_pct=0.015
# Run on Aulavik GPU1 (cuda:1). Light methods parallel, heavy sequential.
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/capture24_grid/baselines_pca15
LOGS=$SPECS/logs
mkdir -p "$LOGS"

echo "[$(date)] Starting light baselines (gdro, flb, smote, ot_repair) in parallel on cuda:1..."

nohup python -u run_baseline.py \
    --spec $SPECS/capture24_gdro_pca15_r4000.yaml \
    --device cuda:1 \
    > "$LOGS/gdro.log" 2>&1 &
PID_GDRO=$!

nohup python -u run_baseline.py \
    --spec $SPECS/capture24_flb_pca15_r4000.yaml \
    --device cuda:1 \
    > "$LOGS/flb.log" 2>&1 &
PID_FLB=$!

nohup python -u run_baseline.py \
    --spec $SPECS/capture24_smote_pca15_r4000.yaml \
    --device cuda:1 \
    > "$LOGS/smote.log" 2>&1 &
PID_SMOTE=$!

nohup python -u run_baseline.py \
    --spec $SPECS/capture24_ot_repair_pca15_r4000.yaml \
    --device cuda:1 \
    > "$LOGS/ot_repair.log" 2>&1 &
PID_OT=$!

echo "[$(date)] PIDs: gdro=$PID_GDRO flb=$PID_FLB smote=$PID_SMOTE ot_repair=$PID_OT"
wait $PID_GDRO $PID_FLB $PID_SMOTE $PID_OT
echo "[$(date)] Light baselines complete. Starting CTGAN..."

python -u run_baseline.py \
    --spec $SPECS/capture24_ctgan_pca15_r4000.yaml \
    --device cuda:1 \
    > "$LOGS/ctgan.log" 2>&1
echo "[$(date)] CTGAN complete. Starting FairTabDDPM..."

python -u run_baseline.py \
    --spec $SPECS/capture24_fairtabddpm_pca15_r4000.yaml \
    --device cuda:1 \
    > "$LOGS/fairtabddpm.log" 2>&1
echo "[$(date)] All baselines complete."
