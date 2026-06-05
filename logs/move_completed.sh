#!/bin/bash
RUNS_DIR=/home/epigou/cs_9170_project/training_runs
CENSUS_DST=/storage_1/epigou_storage/FORGE/experiment3/census_baselines
C24_DST=/storage_1/epigou_storage/FORGE/experiment3/capture24_baselines

moved=0
pending=0

for dir in "$RUNS_DIR"/BASELINE_*G20260604152* "$RUNS_DIR"/BASELINE_*G20260604153* "$RUNS_DIR"/BASELINE_*G20260604154* "$RUNS_DIR"/BASELINE_*G20260604155*; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  if [ -f "$dir/final_test_metrics.csv" ]; then
    if [[ "$name" == *fold* ]]; then
      mv "$dir" "$C24_DST/" && echo "moved c24: $name"
    else
      mv "$dir" "$CENSUS_DST/" && echo "moved census: $name"
    fi
    moved=$((moved + 1))
  else
    pending=$((pending + 1))
  fi
done
echo "Summary: moved=$moved still_running=$pending"
