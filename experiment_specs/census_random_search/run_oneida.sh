#!/bin/bash
# EXP-022 census random search — Oneida (specs 0-6)
set -eo pipefail
cd ~/cs_9170_project
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/census_random_search/logs
for spec in \
  experiment_specs/census_random_search/rand_0000_learning_rate0p004884765313390349_lr0p0003280264061679065_delta_scale0p15514289520771124_optimizeradamw_optimizeradamw.yaml \
  experiment_specs/census_random_search/rand_0001_learning_rate0p0006454584264762864_lr0p00036948531142849293_delta_scale0p12582818151973185_optimizeradamw_optimizeradamw.yaml \
  experiment_specs/census_random_search/rand_0002_learning_rate0p0014681285318933388_lr0p0006549765787793988_delta_scale0p17617171395434755_optimizeradamw_optimizeradam.yaml \
  experiment_specs/census_random_search/rand_0003_learning_rate0p003247943069996102_lr0p0001724797020759017_delta_scale0p11262658534061014_optimizersgd_optimizeradam.yaml \
  experiment_specs/census_random_search/rand_0004_learning_rate0p00041715277661223254_lr0p0002881797740567003_delta_scale0p27470957199199836_optimizersgd_optimizeradamw.yaml \
  experiment_specs/census_random_search/rand_0005_learning_rate0p0008796004257632532_lr1p590019374041111em05_delta_scale0p15854295886344594_optimizersgd_optimizeradam.yaml \
  experiment_specs/census_random_search/rand_0006_learning_rate0p008574577609317105_lr8p995380802416628em05_delta_scale0p26632748194291_optimizeradamw_optimizeradam.yaml; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name"
  python -u main.py --spec "$spec" --device cuda:0 --parallel \
    >> experiment_specs/census_random_search/logs/oneida_${name}.out \
    2>> experiment_specs/census_random_search/logs/oneida_${name}.err
  echo "[$(date)] Done $name"
done
echo "Oneida batch complete"
