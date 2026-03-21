#!/bin/bash
set -euo pipefail
source ~/envs/rl/bin/activate
mkdir -p paper_specs_v1/logs
python -u run_baseline.py \
    --spec paper_specs_v1/p1_nobias_census_gdro_5s.json \
    --device cpu \
    2>&1 | tee paper_specs_v1/logs/p1_nobias_census_gdro_5s.out
