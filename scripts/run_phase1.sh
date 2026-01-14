#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/cpp/fse_acorn_repro
CFG=configs/phase1
LOG=results/phase1/logs
RUNS=(acorn_sift1m_buildA_drawA acorn_sift1m_buildB_drawA acorn_sift1m_buildA_drawB)
METHODS=(acorn_gamma acorn_1)
mkdir -p "$LOG"

stage() {
  local run=$1 st=$2 m=${3:-}
  local log="$LOG/${run}_${st}${m:+_$m}.log"
  echo "$(date -u +%FT%TZ) START $run $st $m"
  if "$BIN" "$CFG/$run.yaml" "$st" ${m:+"$m"} >"$log" 2>&1; then
    echo "$(date -u +%FT%TZ) OK    $run $st $m"
  else
    echo "$(date -u +%FT%TZ) FAIL  $run $st $m (see $log)"
    return 1
  fi
}

stage acorn_sift1m_buildA_drawA gt_checks
stage acorn_sift1m_buildA_drawB gt_checks
for run in "${RUNS[@]}"; do
  for m in "${METHODS[@]}"; do
    stage "$run" build "$m"
  done
done

pids=()
for run in "${RUNS[@]}"; do
  for m in "${METHODS[@]}"; do
    stage "$run" sweep "$m" &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail -eq 0 ]] || { echo "a sweep failed; stopping"; exit 1; }

for run in "${RUNS[@]}"; do
  for m in "${METHODS[@]}"; do
    stage "$run" verify "$m"
    stage "$run" timing "$m"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE"
