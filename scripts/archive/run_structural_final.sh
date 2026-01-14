#!/usr/bin/env bash
# Final structural study (docs/structural_design.md §4): s x C x 5
# realizations; PRE, POST, ACORN-γ, ACORN-1. One selectivity block at a time;
# a block whose decomposition validation fails stops the run.
# Run detached: nohup setsid scripts/run_structural_final.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
F=results/structural/final
L=$F/logs
BIN=./build/cpp
ts() { date -u +%FT%TZ; }
load() { cut -d' ' -f1 /proc/loadavg; }
mkdir -p "$L"
if [ ! -f "$F/conditions/manifest.yaml" ]; then
  echo "$(ts) generate (load $(load))"
  $BIN/fse_fragment_conditions configs/structural/conditions_final.yaml > $L/conditions.log 2>&1 \
    || { echo "$(ts) FAIL generate"; exit 1; }
fi
.venv/bin/python - <<'PY'
import yaml
m = yaml.safe_load(open('results/structural/final/conditions/manifest.yaml'))
for s in sorted({c['s'] for c in m['conditions']}):
    b = dict(m, conditions=[c for c in m['conditions'] if c['s'] == s])
    yaml.safe_dump(b, open(f'results/structural/final/conditions/manifest_s{s:.4f}.yaml', 'w'), sort_keys=False)
PY
for S in 0.0100 0.0625 0.2500; do
  BM=$F/conditions/manifest_s$S.yaml
  mapfile -t CONDS < <(grep -o 'name: [^,}]*' "$BM" | cut -d' ' -f2)
  echo "$(ts) BLOCK s=$S START (${#CONDS[@]} conditions, load $(load))"
  for c in "${CONDS[@]}"; do
    $BIN/fse_matrix configs/structural/matrix_final.yaml sweep prefilter "$c" > $L/pre_$c.log 2>&1 \
      || echo "$(ts) FAIL prefilter $c"
  done
  echo "$(ts) s=$S PRE done"
  for c in "${CONDS[@]}"; do
    echo "configs/structural/matrix_final.yaml $c gamma"
    echo "configs/structural/matrix_final_acorn1.yaml $c acorn1"
  done | xargs -P 16 -L 1 bash -c "$BIN/fse_matrix \$0 sweep acorn \$1 > $L/\$2_\$1.log 2>&1 || echo \"\$(date -u +%FT%TZ) FAIL acorn \$2 \$1\""
  echo "$(ts) s=$S ACORN done"
  for c in "${CONDS[@]}"; do
    $BIN/fse_matrix configs/structural/matrix_final.yaml sweep postfilter "$c" > $L/post_$c.log 2>&1 \
      || echo "$(ts) FAIL postfilter $c"
    echo "$(ts) s=$S POST $c done"
  done
  RC=$F/reach_config_s$S.yaml
  sed -e "s#BLOCK_MANIFEST#$BM#" -e "s#BLOCK_REACH#$F/reach_s$S#" configs/structural/reach_final_template.yaml > $RC
  for m in acorn_gamma acorn_1 postfilter; do
    $BIN/fse_reach $RC $m > $L/reach_${m}_s$S.log 2>&1 || echo "$(ts) FAIL reach $m s=$S" &
  done
  wait
  echo "$(ts) s=$S reach done"
  DC=$F/decomposition_config_s$S.yaml
  sed -e "s#BLOCK_MANIFEST#$BM#" -e "s#BLOCK_REACH#$F/reach_s$S#" -e "s#BLOCK_OUT#$F/analysis_s$S#" \
    configs/structural/decomposition_final_template.yaml > $DC
  .venv/bin/python python/analysis/decomposition.py $DC > $L/decomposition_s$S.log 2>&1
  if grep -q "all_pass: True" $L/decomposition_s$S.log; then
    echo "$(ts) BLOCK s=$S DONE validation all_pass"
  else
    echo "$(ts) BLOCK s=$S VALIDATION FAILED -- stopping"
    exit 1
  fi
done
echo "$(ts) ALL BLOCKS DONE"
