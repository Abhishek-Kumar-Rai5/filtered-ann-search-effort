#!/usr/bin/env bash
# Controlled ACORN-only timing audit (docs/phase4_matrix.md §13).
#   scripts/run_phase4_timing_audit.sh [config]
# Refuses to start unless the VM is idle (1-min load < 1.0) and no other
# fse_matrix process runs; logs load and the top CPU processes every 30 s.
# Run detached: nohup setsid scripts/run_phase4_timing_audit.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/phase4/timing_acorn_audit.yaml}
OUT=results/phase4/timing_audit
LOG=results/phase4/logs/timing_audit
mkdir -p "$OUT" "$LOG"

load1=$(cut -d' ' -f1 /proc/loadavg)
if pgrep -x fse_matrix > /dev/null; then
  echo "$(date -u +%FT%TZ) ABORT another fse_matrix is running"
  exit 1
fi
if ! awk -v l="$load1" 'BEGIN { exit !(l < 1.0) }'; then
  echo "$(date -u +%FT%TZ) ABORT VM not idle (load $load1)"
  exit 1
fi

(
  while true; do
    echo "$(date -u +%FT%TZ) load $(cat /proc/loadavg)"
    ps -eo pcpu,pid,comm --sort=-pcpu | sed -n '2,6p' | sed 's/^/    /'
    sleep 30
  done
) > "$LOG/load.log" 2>&1 &
logger=$!

echo "$(date -u +%FT%TZ) START timing audit (load $load1)"
./build/cpp/fse_matrix "$CFG" timing > "$LOG/fse_matrix.log" 2>&1
rc=$?
kill "$logger" 2> /dev/null
echo "$(date -u +%FT%TZ) DONE rc=$rc (load $(cut -d' ' -f1 /proc/loadavg))"
exit $rc
