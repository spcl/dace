# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` and transients that live only inside the loop body.

``apply`` privatizes every such transient into the loop-body NestedSDFG, so ``can_be_applied``
exempts them from its dependence analysis. That exemption is sound only while each iteration writes
the transient before it reads it. A partially-guarded write (``if c: t = x`` followed by an
unconditional ``use(t)``) leaves the previous iteration's value visible, and privatizing it silently
dropped the carry -- the loop lifted to a Map and produced zeros where the carry should have been.

The first test pins that refusal. The remaining three pin the shapes that must keep parallelizing,
because the cheap fix (never privatize) would cost the pipeline most of its parallelism: a plain
per-iteration scratch value, an exhaustive if/else that writes on every branch, and the CloudSC
shape where one inner loop fills a scratch row and the next reads it back.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol('N')
M = dace.symbol('M')


def n_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


@dace.program
def guarded_carry(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        if i % 2 == 0:
            t = A[i]
        B[i] = t


@dace.program
def private_scratch(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        t = A[i] * 2.0
        B[i] = t + 1.0


@dace.program
def exhaustive_branches(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        if A[i] > 0.5:
            t = A[i]
        else:
            t = -A[i]
        B[i] = t


@dace.program
def scratch_row_refilled(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in range(N):
        row = np.ndarray((M, ), dtype=np.float64)
        for j in range(M):
            row[j] = A[i, j] * 3.0
        for j in range(M):
            B[i, j] = row[j] + 1.0


def test_guarded_write_carry_is_refused():
    """The guarded write leaves ``t`` carrying the previous iteration, so the loop is not DOALL."""
    sdfg = guarded_carry.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0
    sdfg.validate()

    rng = np.random.default_rng(0)
    a = rng.random(8)
    b = np.zeros(8)
    sdfg(A=a, B=b, N=8)

    ref = np.zeros(8)
    carried = 0.0
    for i in range(8):
        if i % 2 == 0:
            carried = a[i]
        ref[i] = carried
    assert np.array_equal(b, ref)


def test_private_scratch_still_maps():
    sdfg = private_scratch.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) > 0
    sdfg.validate()
    assert n_maps(sdfg) > 0

    rng = np.random.default_rng(1)
    a = rng.random(8)
    b = np.zeros(8)
    sdfg(A=a, B=b, N=8)
    assert np.allclose(b, a * 2.0 + 1.0)


def test_exhaustive_branches_still_map():
    """Every branch of an if/else writes ``t``, so no iteration can observe an older value."""
    sdfg = exhaustive_branches.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) > 0
    sdfg.validate()
    assert n_maps(sdfg) > 0

    rng = np.random.default_rng(2)
    a = rng.random(8)
    b = np.zeros(8)
    sdfg(A=a, B=b, N=8)
    assert np.allclose(b, np.where(a > 0.5, a, -a))


def test_scratch_filled_by_inner_loop_still_maps():
    """A nested loop's write counts as written: the CloudSC fill-then-read-back scratch row."""
    sdfg = scratch_row_refilled.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) > 0
    sdfg.validate()
    assert n_maps(sdfg) > 0

    rng = np.random.default_rng(3)
    a = rng.random((4, 5))
    b = np.zeros((4, 5))
    sdfg(A=a, B=b, N=4, M=5)
    assert np.allclose(b, a * 3.0 + 1.0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__]))
