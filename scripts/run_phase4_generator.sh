#!/usr/bin/env bash
# Phase 4 step 2: full-scale generator validation, two separate runs (G8).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/phase4/phase4_generator_full
LOG=results/phase4/logs
for run in 1 2; do
  echo "$(date -u +%FT%TZ) START run $run (load $(cut -d' ' -f1-3 /proc/loadavg))"
  ./build/cpp/fse_filter_conditions configs/phase4/generator_full.yaml \
    > "$LOG/generator_full_run$run.log" 2>&1
  echo "$(date -u +%FT%TZ) END run $run exit=$?"
  rm -rf "${OUT}_run$run"
  cp -r "$OUT" "${OUT}_run$run"
done
echo "$(date -u +%FT%TZ) ALL DONE"
