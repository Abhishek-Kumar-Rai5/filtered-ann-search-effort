"""Tests for the pre-registered Phase 1 summary procedures."""

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "phase1_acorn_summary",
    Path(__file__).resolve().parents[1] / "analysis" / "phase1_acorn_summary.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_crossing_interpolates_linearly_in_recall():
    c = summary.recall_crossing([10, 20, 30], [0.70, 0.78, 0.86],
                                [400, 500, 700])
    assert c["status"] == "ok"
    assert (c["efs_lo"], c["efs_hi"]) == (20, 30)
    # 0.8 is 1/4 of the way from 0.78 to 0.86 -> 500 + 0.25 * 200.
    assert c["ndis_interp"] == pytest.approx(550.0)
    assert c["ndis_at_hi"] == 700


def test_crossing_uses_first_upward_crossing_and_sorts_by_efs():
    c = summary.recall_crossing([30, 10, 20, 40], [0.81, 0.79, 0.80, 0.79],
                                [3, 1, 2, 4])
    assert (c["efs_lo"], c["efs_hi"]) == (10, 20)
    assert c["ndis_interp"] == pytest.approx(2.0)  # exactly at target


def test_crossing_edge_cases():
    above = summary.recall_crossing([10, 20], [0.85, 0.9], [1, 2])
    assert above["status"] == "above_target_at_min_efs"
    assert above["ndis_interp"] is None
    never = summary.recall_crossing([10, 20], [0.5, 0.6], [1, 2])
    assert never["status"] == "never_reaches_target"
    with pytest.raises(ValueError):
        summary.recall_crossing([], [], [])


def test_qps_at_recall_interpolates_in_log_space():
    q = summary.qps_at_recall([0.8, 1.0], [1e5, 1e3], 0.9)
    assert q == pytest.approx(1e4)
    assert summary.qps_at_recall([0.8, 0.85], [1, 2], 0.9) is None


def test_rel_gap():
    assert summary.rel_gap(110.0, 100.0) == pytest.approx(0.1)
    assert summary.rel_gap(None, 100.0) is None
    assert math.isclose(summary.rel_gap(90.0, 100.0), -0.1)
