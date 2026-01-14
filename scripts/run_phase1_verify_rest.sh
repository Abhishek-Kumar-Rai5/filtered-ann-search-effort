#!/usr/bin/env bash
# Phase 1, after the user's 2026-10-03 decision (deviation D3): wait for the
# in-flight primary ACORN-1 timing stage, then run only the remaining verify
# stages (correctness). QPS timing for the follow-up runs is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/cpp/fse_acorn_repro
CFG=configs/phase1
LOG=results/phase1/logs
wait_pid=${1:-}
if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do sleep 10; done
  echo "$(date -u +%FT%TZ) in-flight pid $wait_pid exited"
fi
verify() {  # verify <run> <method>
  local run=$1 m=$2 log="$LOG/${1}_verify_$2.log"
  echo "$(date -u +%FT%TZ) START $run verify $m"
  if "$BIN" "$CFG/$run.yaml" verify "$m" >"$log" 2>&1; then
    echo "$(date -u +%FT%TZ) OK    $run verify $m"
  else
    echo "$(date -u +%FT%TZ) FAIL  $run verify $m (see $log)"
  fi
}
# Two at a time once the machine is free: the determinism pass is single
# threaded; each process briefly holds a 10 GB nq x N filter map, so not 4.
for run in acorn_sift1m_buildB_drawA acorn_sift1m_buildA_drawB; do
  verify "$run" acorn_gamma &
  verify "$run" acorn_1 &
  wait
done
echo "$(date -u +%FT%TZ) ALL DONE"
