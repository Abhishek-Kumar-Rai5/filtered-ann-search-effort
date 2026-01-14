import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "decomposition",
    Path(__file__).resolve().parents[1] / "analysis" / "decomposition.py")
d = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(d)


def test_decompose_partitions_misses_into_unreachable_and_navigational():
    targets = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    found = np.array([[1, 9, -1, -1], [5, 6, 7, 8]])
    reach = np.array([[True, True, False, False], [True, True, True, True]])
    r = d.decompose(targets, found, reach)
    assert r["L"].tolist() == [0.75, 0.0]
    assert r["U"].tolist() == [0.5, 0.0]
    assert r["N"].tolist() == [0.25, 0.0]
    assert np.allclose(r["L"], r["U"] + r["N"])
    assert r["found_unreachable"].tolist() == [0, 0]
    assert r["recall"].tolist() == [0.25, 1.0]


def test_found_target_outside_reach_is_flagged():
    r = d.decompose(np.array([[1, 2]]), np.array([[2, -1]]),
                    np.array([[True, False]]))
    assert r["found_unreachable"].tolist() == [1]


def test_bits_to_matrix_and_padding_never_matches():
    m = d.bits_to_matrix(np.array([0b101, 0]), 3)
    assert m.tolist() == [[True, False, True], [False, False, False]]
    r = d.decompose(np.array([[-1, 3]]), np.array([[-1, -1]]),
                    np.array([[True, True]]))
    assert r["L"].tolist() == [1.0]


def test_gt_reader_checks_identity(tmp_path):
    ids = np.arange(6, dtype=np.int64).reshape(2, 3)
    header = np.array([0xAB, 0xCD, 0xEF, 2, 3], dtype=np.uint64)
    p = tmp_path / "x.gt"
    p.write_bytes(b"FSEGTv1\0" + header.tobytes() + ids.tobytes()
                  + np.zeros(6, np.float32).tobytes())
    got, qh = d.read_gt(p, f"{0xAB:016x}", f"{0xCD:016x}")
    assert np.array_equal(got, ids) and qh == f"{0xEF:016x}"
    with pytest.raises(ValueError):
        d.read_gt(p, f"{0xAC:016x}", f"{0xCD:016x}")


def test_bootstrap_mean_ci_brackets_mean():
    x = np.r_[np.zeros(500), np.ones(500)]
    mu, lo, hi = d.bootstrap_mean_ci(x, np.random.default_rng(0), 400, 0.95)
    assert mu == 0.5 and lo < 0.5 < hi and hi - lo < 0.1
