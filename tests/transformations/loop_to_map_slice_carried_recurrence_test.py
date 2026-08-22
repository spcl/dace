# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopToMap must refuse a slice-indexed loop-carried recurrence.

``for j in range(N-2, 0, -1): u[1:N-1, j] = p[1:N-1, j] * u[1:N-1, j+1] + q[1:N-1, j]`` -- adi's
and deriche's backward sweep -- reads at ``j+1`` what the next iteration writes at ``j``. That is a
genuine dependence, yet LoopToMap parallelized it and both benchmarks produced wrong numbers.

The cause was symbol IDENTITY, not the negative step (the forward twin below miscompiled too). A
DaCe symbol folds its dtype into its sympy identity, so the frontend's subset carried ``j:int64``
while ``LoopToMap`` reparsed the loop variable into ``j:int32``. ``_affine_coeffs`` differentiates
w.r.t. the latter, saw no occurrence of it, and answered ``a == 0`` -- i.e. "this index is a
loop-INVARIANT constant". ``j+1`` and ``j`` then read as the constants ``1`` and ``0``, whose
difference is nonzero, so ``_dim_provably_disjoint`` certified the recurrence as provably disjoint.

The 1D form ``a[i] = a[i+1]`` was refused all along: its subsets are built from one symbol
instance, so nothing mismatched. The 2D slice is what routes the index through a second instance.
"""
import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.interstate.loop_to_map import _affine_coeffs, _dim_provably_disjoint

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def backward_slice_recurrence(u: dace.float64[N, N], p: dace.float64[N, N], q: dace.float64[N, N]):
    for j in range(N - 2, 0, -1):
        u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


@dace.program
def forward_slice_recurrence(u: dace.float64[N, N], p: dace.float64[N, N], q: dace.float64[N, N]):
    for j in range(1, N - 1):
        u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j - 1] + q[1:N - 1, j]


@dace.program
def backward_slice_parallel(u: dace.float64[N, N], p: dace.float64[N, N], q: dace.float64[N, N]):
    for j in range(N - 2, 0, -1):
        u[1:N - 1, j] = p[1:N - 1, j] * q[1:N - 1, j] + 1.0


def _loops(sdfg):
    return [b for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion)]


def _maps_over(sdfg, itervar):
    return [
        n for state in sdfg.all_states() for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and itervar in n.map.params
    ]


def test_affine_coeffs_sees_through_a_dtype_tagged_iteration_symbol():
    """The root cause at unit scale: ``j:int64`` in the index, ``j:int32`` as the iteration symbol.

    Before the fix this returned ``(0, 1)`` -- the index misread as the constant ``1``.
    """
    j64 = symbolic.symbol('j', dace.int64)
    itersym = symbolic.pystr_to_symbolic('j')
    assert j64 is not itersym and j64 != itersym, 'the test needs two distinct instances of j'
    assert _affine_coeffs(j64 + 1, itersym) == (1, 1)
    assert _affine_coeffs(j64, itersym) == (1, 0)


def test_a_unit_distance_recurrence_is_not_provably_disjoint():
    """``u[..., j+1]`` read against ``u[..., j]`` written, over ``j = N-2 .. 1`` step ``-1``."""
    j64 = symbolic.symbol('j', dace.int64)
    itersym = symbolic.pystr_to_symbolic('j')
    step = symbolic.pystr_to_symbolic('-1')
    start = symbolic.pystr_to_symbolic('N - 2')
    assert not _dim_provably_disjoint(j64 + 1, j64, itersym, step, start)
    # The strided case the test is NOT allowed to break: step 2 writes evens, reads odds.
    assert _dim_provably_disjoint(j64 + 1, j64, itersym, symbolic.pystr_to_symbolic('2'),
                                  symbolic.pystr_to_symbolic('0'))


@pytest.mark.parametrize('program', [backward_slice_recurrence, forward_slice_recurrence])
def test_loop_to_map_refuses_a_slice_carried_recurrence(program):
    sdfg = program.to_sdfg(simplify=True)
    before = sdfg.to_json()

    applied = sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    assert applied == 0, 'LoopToMap parallelized a loop-carried slice recurrence'
    assert len(_loops(sdfg)) == 1, 'the sequential loop must survive as a LoopRegion'
    assert not _maps_over(sdfg, 'j'), 'no map may range over the carrying iteration variable'
    assert sdfg.to_json() == before, 'a transformation that does not apply must not mutate the SDFG'


def test_loop_to_map_still_lifts_a_parallel_backward_loop():
    """Positive control: same backward shape, no carried read -- must still become a map."""
    sdfg = backward_slice_parallel.to_sdfg(simplify=True)

    applied = sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    assert applied == 1, 'LoopToMap must still parallelize a backward loop without a carried read'
    assert not _loops(sdfg), 'the loop must be gone'
    assert _maps_over(sdfg, 'j'), 'a map over j must have replaced it'


def test_backward_slice_recurrence_computes_the_sequential_result():
    n = 24
    rng = np.random.default_rng(0)
    p = rng.random((n, n))
    q = rng.random((n, n))
    u = rng.random((n, n))
    ref = u.copy()
    for j in range(n - 2, 0, -1):
        ref[1:n - 1, j] = p[1:n - 1, j] * ref[1:n - 1, j + 1] + q[1:n - 1, j]

    backward_slice_recurrence(u=u, p=p, q=q, N=n)

    assert np.allclose(u, ref)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
