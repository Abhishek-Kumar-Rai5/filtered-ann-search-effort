#!/usr/bin/env bash
# Phase 1 clean QPS re-time (deviation D3 follow-up, user-approved
# 2026-10-03): primary run only, both methods, exactly the pre-registered
# timing stage, on the idle VM. Load average is sampled every 30 s so the
# idleness of the machine is on record.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/cpp/fse_acorn_repro
CFG=configs/phase1/acorn_sift1m_buildA_drawA.yaml
LOG=results/phase1/logs
( while true; do echo "$(date -u +%FT%TZ) $(cat /proc/loadavg)"; sleep 30; done ) \
  > "$LOG/retime_loadavg.log" &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
for m in acorn_gamma acorn_1; do
  echo "$(date -u +%FT%TZ) START retime $m (loadavg: $(cat /proc/loadavg))"
  if "$BIN" "$CFG" timing "$m" > "$LOG/retime_timing_$m.log" 2>&1; then
    echo "$(date -u +%FT%TZ) OK    retime $m"
  else
    echo "$(date -u +%FT%TZ) FAIL  retime $m"
  fi
done
echo "$(date -u +%FT%TZ) ALL DONE"
