#!/bin/bash
RUNS_DIR=/home/epigou/cs_9170_project/training_runs
CENSUS_DST=/storage_1/epigou_storage/FORGE/experiment3/census_baselines
C24_DST=/storage_1/epigou_storage/FORGE/experiment3/capture24_baselines

echo "[$(date '+%H:%M:%S')] watcher started"

is_census_complete() {
  local dir=$1
  # need header + 5 seed rows = 6 lines
  local n=$(wc -l < "$dir/final_test_metrics.csv" 2>/dev/null || echo 0)
  [ "$n" -ge 6 ]
}

is_c24_complete() {
  local dir=$1
  # need header + 1 seed row = 2 lines
  local n=$(wc -l < "$dir/final_test_metrics.csv" 2>/dev/null || echo 0)
  [ "$n" -ge 2 ]
}

while true; do
  pending=0
  for dir in "$RUNS_DIR"/BASELINE_*G20260604152* "$RUNS_DIR"/BASELINE_*G20260604153* "$RUNS_DIR"/BASELINE_*G20260604154* "$RUNS_DIR"/BASELINE_*G20260604155* "$RUNS_DIR"/BASELINE_*G20260604160* "$RUNS_DIR"/BASELINE_*G20260604161* "$RUNS_DIR"/BASELINE_*G20260604162* "$RUNS_DIR"/BASELINE_*G20260604163* "$RUNS_DIR"/BASELINE_*G20260604164* "$RUNS_DIR"/BASELINE_*G20260604165* "$RUNS_DIR"/BASELINE_*G20260604166* "$RUNS_DIR"/BASELINE_*G20260604167* "$RUNS_DIR"/BASELINE_*G20260604168* "$RUNS_DIR"/BASELINE_*G20260604169*; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    if [[ "$name" == *fold* ]]; then
      if is_c24_complete "$dir"; then
        rsync -a --remove-source-files "$dir/" "$C24_DST/$name/" && \
          rmdir "$dir" 2>/dev/null && \
          echo "[$(date '+%H:%M:%S')] moved c24: $name"
      else
        pending=$((pending + 1))
      fi
    else
      if is_census_complete "$dir"; then
        rsync -a --remove-source-files "$dir/" "$CENSUS_DST/$name/" && \
          rmdir "$dir" 2>/dev/null && \
          echo "[$(date '+%H:%M:%S')] moved census: $name"
      else
        pending=$((pending + 1))
      fi
    fi
  done
  [ $pending -eq 0 ] && echo "[$(date '+%H:%M:%S')] all done — exiting" && exit 0
  sleep 30
done
