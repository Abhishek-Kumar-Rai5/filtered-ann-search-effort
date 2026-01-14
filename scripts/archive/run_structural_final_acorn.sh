#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
F=results/structural/final
L=$F/logs
BIN=./build/cpp
ts() { date -u +%FT%TZ; }
while pgrep -x fse_fragment_co > /dev/null; do sleep 30; done
[ -f $F/conditions/manifest.yaml ] || { echo "$(ts) FAIL no manifest"; exit 1; }
.venv/bin/python - <<'PY'
import yaml
m = yaml.safe_load(open('results/structural/final/conditions/manifest.yaml'))
b = dict(m, conditions=[c for c in m['conditions'] if abs(c['s'] - 0.01) < 1e-12])
yaml.safe_dump(b, open('results/structural/final/conditions/manifest_s0.0100.yaml', 'w'), sort_keys=False)
print(len(b['conditions']), 'conditions at s = 0.01')
PY
mapfile -t CONDS < <(grep -o 'name: [^,}]*' $F/conditions/manifest_s0.0100.yaml | cut -d' ' -f2)
echo "$(ts) ACORN sweeps (${#CONDS[@]} conditions x 2)"
for c in "${CONDS[@]}"; do
  echo "configs/structural/matrix_final.yaml $c gamma"
  echo "configs/structural/matrix_final_acorn1.yaml $c acorn1"
done | xargs -P 16 -L 1 bash -c "$BIN/fse_matrix \$0 sweep acorn \$1 > $L/\$2_\$1.log 2>&1 || echo \"\$(date -u +%FT%TZ) FAIL acorn \$2 \$1\""
echo "$(ts) reach"
$BIN/fse_reach configs/structural/reach_final_acorn.yaml acorn_gamma > $L/reach_acorn_gamma.log 2>&1 || echo "$(ts) FAIL reach gamma" &
$BIN/fse_reach configs/structural/reach_final_acorn.yaml acorn_1 > $L/reach_acorn_1.log 2>&1 || echo "$(ts) FAIL reach acorn_1" &
$BIN/fse_reach configs/structural/baseline_reach.yaml postfilter > $L/reach_baseline_post.log 2>&1 || echo "$(ts) FAIL reach baseline post" &
wait
echo "$(ts) decomposition"
.venv/bin/python python/analysis/decomposition.py configs/structural/decomposition_final_acorn.yaml > $L/decomposition_acorn.log 2>&1
.venv/bin/python python/analysis/decomposition.py configs/structural/baseline_decomposition.yaml > $L/decomposition_baseline.log 2>&1
grep -H "all_pass" $L/decomposition_*.log
echo "$(ts) DONE"
