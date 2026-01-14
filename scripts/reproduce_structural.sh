#!/usr/bin/env bash
# Reproduce the reported structural results (docs/structural_results.md):
# vertical slice, decision experiment and PRE/POST baselines at s = 0.01,
# then the summary and the final report.
#
# Prerequisites (earlier phases, see README "Reproducing"):
#   - SIFT1M in data/sift/ (scripts/download_sift1m.sh)
#   - ACORN instrumentation applied (patches/acorn_instrumentation.patch), build/
#   - Phase 1 ACORN-1 index  data/cache/phase1/acorn_acorn_1_buildA.index
#   - Phase 4 indexes, attributes, ground truth (data/cache/phase4/) and the
#     Phase 4 matrix sweeps at s = 0.01 (results/phase4/matrix/{prefilter,
#     postfilter,acorn}/{random,clustered}_s0.0100) -- configs/phase4/
# Index hashes are verified by every driver.
#
# Runtime on the reference VM (16 vCPU AMD EPYC-Milan, 58 GB RAM): condition
# generation ~1.5 h (k-means per realization), ACORN sweeps 10-20 min per
# condition (single-threaded, run in parallel), reachability ~1 min per
# condition and method, analysis a few minutes.
#
# Usage: scripts/reproduce_structural.sh     (run detached: nohup setsid ...)
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
# C100 r0: new sweeps. C1000 r0 and random r0 are the Phase 4 / slice
# conditions (identical masks): their sweeps are reused via relative links.
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
