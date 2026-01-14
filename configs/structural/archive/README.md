# Archived structural configs — NOT used for any reported result

Configs of structural runs that were stopped by user decision before
completion (docs/structural_design.md §7):

- `*_mvp*` — 8-condition validation grid (stopped during POST, C100).
- `pilot_*` — 5-realization fragmentation pilot (stopped during condition
  generation; superseded by the decision experiment).
- `*_final*` — full 60-condition study and the reduced 20-condition run
  (stopped; partial outputs are not used).

The configs that produced the reported results are in the parent directory:
`conditions_pilot.yaml`, `slice_*`, `decision_*`, `baseline_*`
(see `scripts/reproduce_structural.sh`). Paths inside these archived files
refer to their original locations and are not maintained.
