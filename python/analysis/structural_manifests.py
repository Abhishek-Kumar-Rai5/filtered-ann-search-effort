"""Build the condition manifests of the reported structural runs.

During the original runs these manifests were assembled ad hoc; this script
reproduces them from generator output (condition ids and mask hashes are
copied, never recomputed, and every downstream driver re-verifies them).

  decision  <pilot manifest> -> results/structural/pilot/decision/manifest.yaml
            (+ manifest_r0.yaml): C100 r0-r4, C1000 r0-r4, random r0-r1 at
            s = 0.01, selected from the output of
            fse_fragment_conditions configs/structural/conditions_pilot.yaml
  slice     <decision manifest> -> results/structural/slice/manifest.yaml:
            C1000 r0 and random r0 under their Phase 4 names
            (clustered_s0.0100, random_s0.0100) with the Phase 4 attribute and
            ground-truth files (identical masks: same seeds, verified hashes).

The committed decision manifest points random r0/r1 at the earlier MVP
cache (data/cache/structural); a rebuilt one points at the pilot cache. The
attribute hashes, condition ids and mask hashes are identical.

Usage:
  .venv/bin/python python/analysis/structural_manifests.py decision \
      results/structural/pilot/conditions/manifest.yaml
  .venv/bin/python python/analysis/structural_manifests.py slice \
      results/structural/pilot/decision/manifest.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

DECISION = {"C100": range(5), "C1000": range(5), "random": range(2)}
PHASE4 = {"C1000": ("clustered_s0.0100", "clustered"), "random": ("random_s0.0100", "random")}


def decision(src: Path) -> None:
    m = yaml.safe_load(src.read_text())
    keep = [c for c in m["conditions"]
            if abs(c["s"] - 0.01) < 1e-12 and c["realization"] in DECISION.get(c["level"], ())]
    if len(keep) != sum(len(r) for r in DECISION.values()):
        raise SystemExit(f"expected 12 decision conditions, found {len(keep)}")
    out = Path("results/structural/pilot/decision")
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(dict(m, conditions=keep), (out / "manifest.yaml").open("w"), sort_keys=False)
    r0 = [c for c in keep if c["realization"] == 0]
    yaml.safe_dump(dict(m, conditions=r0), (out / "manifest_r0.yaml").open("w"), sort_keys=False)
    print(f"wrote {out}/manifest.yaml ({len(keep)}) and manifest_r0.yaml ({len(r0)})")


def slice_(src: Path) -> None:
    m = yaml.safe_load(src.read_text())
    out = []
    for c in m["conditions"]:
        if c["realization"] != 0 or c["level"] not in PHASE4:
            continue
        name, corr = PHASE4[c["level"]]
        out.append(dict(c, name=name,
                        attribute=f"data/cache/phase4/attr_{corr}_{c['attribute_hash']}.bin",
                        gt=f"data/cache/phase4/{name}_{c['condition_id']}.gt"))
    dst = Path("results/structural/slice")
    dst.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(dict(m, conditions=out), (dst / "manifest.yaml").open("w"), sort_keys=False)
    print(f"wrote {dst}/manifest.yaml ({len(out)})")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("decision", "slice"):
        raise SystemExit(__doc__)
    (decision if sys.argv[1] == "decision" else slice_)(Path(sys.argv[2]))
