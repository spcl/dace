# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the DaCe ports of the SC26 reference kernels through the brute-force layout sweep.

Each kernel enumerates its global layout candidates (dimension permutations of the array whose
orientation is the layout decision), and the sweep compiles/runs/verifies every candidate against
the numpy oracle. The invariant asserted here is CORRECTNESS: every transparent candidate reproduces
the oracle and ``best()`` returns a correct one. Timing is not asserted (noisy on a shared host)."""
import numpy

from dace.transformation.layout.brute_force import sweep, best
from tests.transformations.layout.kernels import k04_mvt, k15_colstore


def _sdfg_candidates(program, candidate_dict):
    """Turn ``{name: apply}`` into ``{name: make_sdfg}`` -- build a fresh SDFG and apply the layout."""

    def make_for(apply):

        def make():
            sdfg = program.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    return {name: make_for(apply) for name, apply in candidate_dict.items()}


def test_k04_mvt_layout_sweep_all_verify():
    """mvt reads A in both orientations; every global permutation of A reproduces x1/x2."""
    n = 24
    inputs = k04_mvt.make_inputs(n)
    reference = k04_mvt.oracle(inputs["A"], inputs["y1"], inputs["y2"])
    candidates = _sdfg_candidates(k04_mvt.mvt, k04_mvt.candidates())

    results = sweep(candidates, k04_mvt.run_closure(inputs, n), reference, do_time=False)
    assert set(candidates) == {"permute_A_01", "permute_A_10"}
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


def test_k15_colstore_layout_sweep_all_verify():
    """Row-store vs column-store of the table T: both column-sum candidates reproduce the oracle."""
    n, c = 32, 8
    inputs = k15_colstore.make_inputs(n, c)
    reference = k15_colstore.oracle(inputs["T"])
    candidates = _sdfg_candidates(k15_colstore.colsum, k15_colstore.candidates())

    results = sweep(candidates, k15_colstore.run_closure(inputs, n, c), reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


if __name__ == "__main__":
    test_k04_mvt_layout_sweep_all_verify()
    test_k15_colstore_layout_sweep_all_verify()
    print("kernel sweep tests PASS")
