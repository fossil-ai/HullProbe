"""Readouts of hullprobe.audit on hand-built cubes (no data, CPU)."""
import numpy as np

from hullprobe import audit

Z = np.arange(64)


def bit(p):
    return (Z >> p) & 1


def test_single_cell_classifier():
    # correct iff cell 0 is kept: only cell 0 matters, purely first-order, one swap flips it
    A = (bit(0) == 0).astype(np.int8)[None].repeat(3, 0)
    assert np.allclose(audit.influence(A), [1, 0, 0, 0, 0, 0])
    assert np.isclose(audit.walsh_degrees(A)[0], 1.0)
    assert audit.semantic_l0(A) == 1


def test_pairwise_and_never_flip():
    # correct unless cells 1 and 4 are both swapped: a pure degree-2 AND
    A = (~((bit(1) == 1) & (bit(4) == 1))).astype(np.int8)[None]
    inf = audit.influence(A)
    assert np.allclose(inf[[0, 2, 3, 5]], 0) and np.allclose(inf[[1, 4]], 0.5)
    assert audit.semantic_l0(A) == 2
    assert audit.semantic_l0(np.ones((2, 64), np.int8)) == 7      # never flips within six cells


def test_summarize_hull_share():
    # corrected influence 0.3 on each hull cell and 0.1 on each superstructure cell -> share 0.75
    res = [{"inf_cross": np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]),
            "inf_same": np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]),
            "W_by_degree": np.array([0.6, 0.2, 0.1, 0.1, 0, 0]), "r": 1}] * 5
    s = audit.summarize(res, n_boot=50)
    assert np.isclose(s["hull_share"]["value"], 0.75)
    assert np.isclose(s["W_le2"]["value"], 0.8) and s["frac_r1"]["value"] == 1.0
