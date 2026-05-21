#!/bin/bash
# Run k=0 capture24 gap-fills on Aulavik GPU0 (cuda:0) in sequence.
# 54 runs total: ep10/pca15 (9) -> ep20/allpca (27) -> ep30/pca1015 (18)
# Estimated wall time: ~10 days. Started from ~/cs_9170_project.

set -e
LOGDIR=experiment_specs/capture24_grid/logs
mkdir -p "$LOGDIR"

source ~/envs/rl/bin/activate

echo "[$(date)] Starting capture24_k0_aulavik_ep10pca15 on cuda:0"
python -u main.py \
  --spec experiment_specs/capture24_grid/capture24_k0_aulavik_ep10pca15.yaml \
  --device cuda:0 --parallel \
  2>&1 | tee -a "$LOGDIR/k0_aulavik_gpu0.log"
echo "[$(date)] Finished capture24_k0_aulavik_ep10pca15"

echo "[$(date)] Starting capture24_k0_aulavik_ep20 on cuda:0"
python -u main.py \
  --spec experiment_specs/capture24_grid/capture24_k0_aulavik_ep20.yaml \
  --device cuda:0 --parallel \
  2>&1 | tee -a "$LOGDIR/k0_aulavik_gpu0.log"
echo "[$(date)] Finished capture24_k0_aulavik_ep20"

echo "[$(date)] Starting capture24_k0_aulavik_ep30pca1015 on cuda:0"
python -u main.py \
  --spec experiment_specs/capture24_grid/capture24_k0_aulavik_ep30pca1015.yaml \
  --device cuda:0 --parallel \
  2>&1 | tee -a "$LOGDIR/k0_aulavik_gpu0.log"
echo "[$(date)] Finished capture24_k0_aulavik_ep30pca1015"

echo "[$(date)] All k=0 Aulavik GPU0 runs complete."
