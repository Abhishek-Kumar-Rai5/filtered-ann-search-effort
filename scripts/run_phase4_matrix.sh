#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=$1
LOG=$2
shift 2
CONDS=("$@")
ACORN_PROCS=${ACORN_PROCS:-12}
BIN=./build/cpp/fse_matrix
mkdir -p "$LOG"
fail=0

sweep() {
  local m=$1 c=$2
  echo "$(date -u +%FT%TZ) START $m $c (load $(cut -d' ' -f1 /proc/loadavg))"
  if "$BIN" "$CFG" sweep "$m" "$c" > "$LOG/${m}_${c}.log" 2>&1; then
    echo "$(date -u +%FT%TZ) OK    $m $c"
  else
    echo "$(date -u +%FT%TZ) FAIL  $m $c (see $LOG/${m}_${c}.log)"
    return 1
  fi
}

for c in "${CONDS[@]}"; do sweep prefilter "$c" || fail=1; done
for c in "${CONDS[@]}"; do sweep postfilter "$c" || fail=1; done
running=0
pids=()
for c in "${CONDS[@]}"; do
  sweep acorn "$c" &
  pids+=($!)
  running=$((running + 1))
  if [[ $running -ge $ACORN_PROCS ]]; then
    wait -n || fail=1
    running=$((running - 1))
  fi
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
wait || true
echo "$(date -u +%FT%TZ) ALL DONE fail=$fail"
