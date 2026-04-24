#!/bin/bash
set -e
source ~/envs/rl/bin/activate

SPECS=experiment_specs/April_10_Experiments
LOGS=$SPECS/logs

echo "=== Census (k=10, EP3000) ==="
python3 main.py --spec $SPECS/census_v18_ep3000_gpu0.yaml\1 --device cuda:0 > $LOGS/census_gpu0.out 2>&1 &
python3 main.py --spec $SPECS/census_v18_ep3000_gpu1.yaml\1 --device cuda:1 > $LOGS/census_gpu1.out 2>&1 &
wait
echo "Census done."

echo "=== Capture-24 (k=3, EP3000) ==="
python3 main.py --spec $SPECS/capture24_v18k3_ep3000_gpu0.yaml\1 --device cuda:0 > $LOGS/capture24_gpu0.out 2>&1 &
python3 main.py --spec $SPECS/capture24_v18k3_ep3000_gpu1.yaml\1 --device cuda:1 > $LOGS/capture24_gpu1.out 2>&1 &
wait
echo "Capture-24 done."

echo "=== COMPAS (k=3, EP3000) ==="
python3 main.py --spec $SPECS/compas_v18k3_ep3000_gpu0.yaml\1 --device cuda:0 > $LOGS/compas_gpu0.out 2>&1 &
python3 main.py --spec $SPECS/compas_v18k3_ep3000_gpu1.yaml\1 --device cuda:1 > $LOGS/compas_gpu1.out 2>&1 &
wait
echo "COMPAS done."

echo "All experiments complete."
