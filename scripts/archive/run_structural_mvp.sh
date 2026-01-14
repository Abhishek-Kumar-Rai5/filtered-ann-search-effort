#!/usr/bin/env bash
# Structural validation experiment (docs/structural_design.md §5).
#   scripts/run_structural_mvp.sh
# Requires the conditions manifest (fse_fragment_conditions). Pre-/post-filter
# sweeps run one condition at a time (16 OpenMP threads); ACORN-γ and ACORN-1
# sweeps run as single-threaded processes in parallel (process-global
# counters); then fse_reach per method. Logs START/OK/FAIL with load.
# Run detached: nohup setsid scripts/run_structural_mvp.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
MAN=results/structural/mvp/conditions/manifest.yaml
LOG=results/structural/mvp/logs
BIN=./build/cpp
mkdir -p "$LOG"
mapfile -t CONDS < <(grep -o 'name: [^,]*' "$MAN" | cut -d' ' -f2)
fail=0

step() {  # step <label> <cmd...>
  local label=$1
  shift
  echo "$(date -u +%FT%TZ) START $label (load $(cut -d' ' -f1 /proc/loadavg))"
  if "$@" > "$LOG/${label// /_}.log" 2>&1; then
    echo "$(date -u +%FT%TZ) OK    $label"
  else
    echo "$(date -u +%FT%TZ) FAIL  $label"
    return 1
  fi
}

for c in "${CONDS[@]}"; do
  step "prefilter $c" $BIN/fse_matrix configs/structural/matrix_mvp.yaml sweep prefilter "$c" || fail=1
done
for c in "${CONDS[@]}"; do
  step "postfilter $c" $BIN/fse_matrix configs/structural/matrix_mvp.yaml sweep postfilter "$c" || fail=1
done
pids=()
for c in "${CONDS[@]}"; do
  step "acorn_gamma $c" $BIN/fse_matrix configs/structural/matrix_mvp.yaml sweep acorn "$c" &
  pids+=($!)
  step "acorn_1 $c" $BIN/fse_matrix configs/structural/matrix_mvp_acorn1.yaml sweep acorn "$c" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
pids=()
for m in acorn_gamma acorn_1 postfilter; do
  step "reach $m" $BIN/fse_reach configs/structural/reach_mvp.yaml "$m" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "$(date -u +%FT%TZ) ALL DONE fail=$fail"
