#!/usr/bin/env bash
# Reruns the reported structural results end to end. It needs SIFT1M, the
# ACORN patch applied, and the Phase 1 / Phase 4 indexes and s = 0.01 sweeps
# already built (see the README).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/cpp
PY=.venv/bin/python
ts() { date -u +%FT%TZ; }

echo "$(ts) 1/6 conditions (C100/C1000/C10000/random x 5 realizations, s = 0.01)"
$BIN/fse_fragment_conditions configs/structural/conditions_pilot.yaml
$PY python/analysis/structural_manifests.py decision results/structural/pilot/conditions/manifest.yaml
$PY python/analysis/structural_manifests.py slice results/structural/pilot/decision/manifest.yaml

echo "$(ts) 2/6 vertical slice: ACORN-1 sweeps (ACORN-γ = Phase 4 sweeps), reachability, decomposition"
for c in random_s0.0100 clustered_s0.0100; do
  $BIN/fse_matrix configs/structural/slice_acorn1.yaml sweep acorn "$c" &
done
wait
$BIN/fse_reach configs/structural/slice_reach.yaml acorn_gamma
$BIN/fse_reach configs/structural/slice_reach.yaml acorn_1
$PY python/analysis/decomposition.py configs/structural/slice_decomposition.yaml

echo "$(ts) 3/6 decision experiment: realization-0 sweeps"
$BIN/fse_matrix configs/structural/decision_gamma.yaml sweep acorn C100_r0_s0.0100 &
$BIN/fse_matrix configs/structural/decision_acorn1.yaml sweep acorn C100_r0_s0.0100 &
wait
D=results/structural/pilot/decision
mkdir -p $D/matrix_gamma/acorn $D/matrix_acorn1/acorn
ln -sfn ../../../../../phase4/matrix/acorn/clustered_s0.0100 $D/matrix_gamma/acorn/C1000_r0_s0.0100
ln -sfn ../../../../../phase4/matrix/acorn/random_s0.0100 $D/matrix_gamma/acorn/random_r0_s0.0100
ln -sfn ../../../../slice/matrix_acorn1/acorn/clustered_s0.0100 $D/matrix_acorn1/acorn/C1000_r0_s0.0100
ln -sfn ../../../../slice/matrix_acorn1/acorn/random_s0.0100 $D/matrix_acorn1/acorn/random_r0_s0.0100

echo "$(ts) 4/6 decision experiment: reachability (all 12 conditions), decomposition (r0)"
$BIN/fse_reach configs/structural/decision_reach.yaml acorn_gamma
$BIN/fse_reach configs/structural/decision_reach.yaml acorn_1
$PY python/analysis/decomposition.py configs/structural/decision_decomposition.yaml

echo "$(ts) 5/6 PRE/POST baselines (Phase 4 sweeps; POST reachability on the unfiltered graph)"
$BIN/fse_reach configs/structural/baseline_reach.yaml postfilter
$PY python/analysis/decomposition.py configs/structural/baseline_decomposition.yaml

echo "$(ts) 6/6 summary and final report"
$PY python/analysis/structural_decision_summary.py
$PY python/analysis/structural_report.py
echo "$(ts) done -> results/structural/report/"
