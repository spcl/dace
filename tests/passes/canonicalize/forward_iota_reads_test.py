# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.libraries.sort.nodes import ScatterConflictCheck
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, W = dace.symbol("N"), dace.symbol("W")


@dace.program
def scatter_through_arange(sx: dace.float64[N, W], c: dace.int64[N], x: dace.float64[N]):
    rows = np.arange(N)
    for i in range(N):
        sx[rows[i], c[i]] = x[i]


@dace.program
def scatter_through_rewritten_arange(sx: dace.float64[N, W], c: dace.int64[N], x: dace.float64[N]):
    rows = np.arange(N)
    rows[0] = 1
    for i in range(N):
        sx[rows[i], c[i]] = x[i]


def conflict_checks(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, ScatterConflictCheck) for n, _ in sdfg.all_nodes_recursive())


def test_a_row_scatter_through_arange_is_proven_without_a_conflict_check():
    """``sx[rows[i], c[i]]`` with ``rows = arange(N)`` writes row ``i`` only: no two iterations collide."""
    sdfg = scatter_through_arange.to_sdfg(simplify=False)
    canonicalize(sdfg, target="cpu", nonnegative_symbols=["N", "W"])
    assert conflict_checks(sdfg) == 0
    assert not any(isinstance(n, dace.sdfg.state.LoopRegion) for n in sdfg.all_control_flow_regions())
    n, w = 64, 5
    rng = np.random.default_rng(0)
    c, x = rng.integers(0, w, n), rng.random(n)
    sx = np.zeros((n, w))
    sdfg(sx=sx, c=c, x=x, N=n, W=w)
    ref = np.zeros((n, w))
    ref[np.arange(n), c] = x
    assert np.array_equal(sx, ref)


def test_an_index_array_written_after_its_fill_keeps_the_conflict_check():
    sdfg = scatter_through_rewritten_arange.to_sdfg(simplify=False)
    canonicalize(sdfg, target="cpu", nonnegative_symbols=["N", "W"])
    assert conflict_checks(sdfg) == 1


if __name__ == "__main__":
    test_a_row_scatter_through_arange_is_proven_without_a_conflict_check()
    test_an_index_array_written_after_its_fill_keeps_the_conflict_check()
