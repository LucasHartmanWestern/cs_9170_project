#!/bin/bash
# EXP-022 census random search — Lambda (specs 14-19)
set -eo pipefail
cd ~/cs_9170_project
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/census_random/logs
for spec in \
  experiment_specs/census_random/rand_0014_learning_rate0p0015580224927221986_lr5p8857565140163605em05_delta_scale0p19391275354122212_optimizeradamw_optimizeradam.yaml \
  experiment_specs/census_random/rand_0015_learning_rate0p0002392118319988556_lr2p362974617171558em05_delta_scale0p20319329496715166_optimizersgd_optimizeradamw.yaml \
  experiment_specs/census_random/rand_0016_learning_rate0p0008975568877999452_lr1p5123375045656258em05_delta_scale0p23940098049160918_optimizeradam_optimizeradam.yaml \
  experiment_specs/census_random/rand_0017_learning_rate0p004840836853954369_lr0p0006256713125146971_delta_scale0p2807706099550442_optimizersgd_optimizeradamw.yaml \
  experiment_specs/census_random/rand_0018_learning_rate0p004739477462834176_lr0p00011198191194252835_delta_scale0p18044752518018464_optimizeradam_optimizeradam.yaml \
  experiment_specs/census_random/rand_0019_learning_rate0p006167056607692355_lr0p000151217609067703_delta_scale0p28744121830803016_optimizersgd_optimizeradamw.yaml; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name"
  python -u main.py --spec "$spec" --device cuda:1 \
    >> experiment_specs/census_random/logs/lambda_${name}.out \
    2>> experiment_specs/census_random/logs/lambda_${name}.err
  echo "[$(date)] Done $name"
done
echo "Lambda batch complete"
