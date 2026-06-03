#!/bin/bash
# EXP-022 census random search — Aulavik (specs 7-13)
set -eo pipefail
cd ~/cs_9170_project
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/census_random_search/logs
for spec in \
  experiment_specs/census_random_search/rand_0007_learning_rate0p004074324861703429_lr0p00012514070798404372_delta_scale0p05351042504100474_optimizersgd_optimizeradamw.yaml \
  experiment_specs/census_random_search/rand_0008_learning_rate0p0026333394617191514_lr0p0003716169539861266_delta_scale0p2063164573184081_optimizersgd_optimizeradamw.yaml \
  experiment_specs/census_random_search/rand_0009_learning_rate0p004530295770311979_lr4p6373542652131554em05_delta_scale0p23256964408830622_optimizersgd_optimizeradam.yaml \
  experiment_specs/census_random_search/rand_0010_learning_rate0p00024106501268706255_lr0p00013646506337961615_delta_scale0p10965398215380505_optimizeradam_optimizeradamw.yaml \
  experiment_specs/census_random_search/rand_0011_learning_rate0p00015220604727359953_lr0p0009692662864435307_delta_scale0p2687718218340364_optimizeradamw_optimizeradam.yaml \
  experiment_specs/census_random_search/rand_0012_learning_rate0p0004007685047508204_lr3p821037896892759em05_delta_scale0p08120267865702678_optimizeradamw_optimizeradam.yaml \
  experiment_specs/census_random_search/rand_0013_learning_rate0p008465977149720158_lr0p00016083155314132244_delta_scale0p19690426604385908_optimizeradamw_optimizeradam.yaml; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name"
  python -u main.py --spec "$spec" --device cuda:0 \
    >> experiment_specs/census_random_search/logs/aulavik_${name}.out \
    2>> experiment_specs/census_random_search/logs/aulavik_${name}.err
  echo "[$(date)] Done $name"
done
echo "Aulavik batch complete"
