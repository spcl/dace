# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` and the loop-INDEPENDENT vs loop-CARRIED distinction.

Aliasing is not dependence. Two accesses that address the same element at *some* point still
leave a loop DOALL when every such point lies inside a single iteration -- the classic
dependence-distance criterion (Banerjee / Allen-Kennedy): only a dependence carried at the
loop's own level, i.e. one connecting two DISTINCT iterations, forbids parallelizing it.

The kernel here is TSVC ``s114``::

    for i:
        for j:
            aa[i, j] = aa[j, i] + bb[i, j]

For a fixed ``i`` the ``j`` loop writes ``aa[i, j]`` and reads ``aa[j, i]``; the two address the
same element only when ``j == i``, i.e. within ONE iteration of that loop, where the map body
keeps the read-before-write order verbatim. So ``j`` is parallel. The ``i`` loop is not: read
``aa[j_p, p]`` of iteration ``p`` meets write ``aa[q, j_q]`` of iteration ``q`` whenever
``j_p == q`` and ``p == j_q``, which has solutions with ``p != q`` -- the transpose
anti-dependence, genuinely carried.

Structurally the two look identical (the loop variable lands in different dimensions of the read
and the write either way). What separates them is that ``j`` VARIES inside the ``i`` loop's body,
so it cannot be treated as one shared parameter of a cross-iteration comparison, while ``i`` is
fixed for the whole execution of the ``j`` loop and can. That is the ``varying`` guard inside
``_collision_forces_same_iteration``; the negative tests below are what pins it down.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')
C = dace.symbol('C')
NB = dace.symbol('NB')
V = dace.symbol('V')


def maps_with_param(sdfg: dace.SDFG, param: str):
    """MapEntry nodes (recursively) whose map iterates over ``param``."""
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and param in n.map.params]


def loops_with_var(sdfg: dace.SDFG, var: str):
    """LoopRegions whose loop variable is ``var``, across every (nested) SDFG."""
    return [
        r for nested in sdfg.all_sdfgs_recursive() for r in nested.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable == var
    ]


# ---------------------------------------------------------------------------
# Positive: the alias is confined to distance 0, so the axis is DOALL.
# ---------------------------------------------------------------------------


@dace.program
def transpose_add(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            aa[i, j] = aa[j, i] + bb[i, j]


def transpose_add_ref(aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
    out = aa.copy()
    n = aa.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = out[j, i] + bb[i, j]
    return out


def test_inner_axis_alias_at_distance_zero_maps():
    """``aa[i,j] = aa[j,i] + ...``: the inner ``j`` axis parallelizes, the outer ``i`` does not.

    The inner axis' read and write DO alias -- at ``j == i`` iteration ``j`` reads and then writes
    ``aa[i, i]`` -- but only there, so the dependence is loop-independent and cannot be observed by
    reordering iterations. Before the collision certificate was consulted for read/write pairs,
    ``LoopToMap`` refused this loop outright and s114 emitted zero parallel loops.
    """
    sdfg = transpose_add.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert maps_with_param(sdfg, 'j'), "inner axis j aliases only at distance 0 and must become a Map"
    assert not loops_with_var(sdfg, 'j'), "inner axis j should no longer be a sequential loop"

    n = 9
    rng = np.random.default_rng(0)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    ref = transpose_add_ref(aa, bb)
    sdfg(aa=aa, bb=bb, N=n)
    assert np.allclose(aa, ref)


@dace.program
def transpose_add_scaled_triangular(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for i in range(NB):
        for j in range(i * V):
            aa[i, j] = aa[j, i] + bb[i, j]


def transpose_add_scaled_triangular_ref(aa: np.ndarray, bb: np.ndarray, nb: int, v: int) -> np.ndarray:
    out = aa.copy()
    for i in range(nb):
        for j in range(i * v):
            out[i, j] = out[j, i] + bb[i, j]
    return out


def test_scaled_triangular_inner_axis_maps():
    """TSVC ``s114``'s own shape: the inner bound is ``i * V``, not ``i``.

    The scaling matters. With a plain ``range(i)`` bound, ``j < i`` puts every write strictly below
    the diagonal and every read strictly above it, and the propagated read/write slabs are already
    disjoint -- the loop parallelizes without any dependence reasoning. With ``range(i * V)`` the
    slabs overlap again and only the distance-0 argument saves it, which is why s114 in particular
    emitted no parallel loop at all before this change.
    """
    sdfg = transpose_add_scaled_triangular.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert maps_with_param(sdfg, 'j'), "s114's inner axis j must become a Map"
    assert not loops_with_var(sdfg, 'j')

    n, nb, v = 16, 4, 4
    rng = np.random.default_rng(1)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    ref = transpose_add_scaled_triangular_ref(aa, bb, nb, v)
    sdfg(aa=aa, bb=bb, N=n, NB=nb, V=v)
    assert np.allclose(aa, ref)


# ---------------------------------------------------------------------------
# Negative: a dependence carried at distance != 0 must STILL refuse.
# ---------------------------------------------------------------------------


def test_outer_transpose_axis_stays_sequential():
    """The outer ``i`` axis of the SAME kernel carries the transpose anti-dependence.

    Iteration ``p`` reads ``aa[j, p]`` for every inner ``j``, iteration ``q`` writes ``aa[q, j]``
    for every inner ``j``; they meet at ``j == q`` / ``j == p`` with ``p != q``, so the dependence
    is carried and the axis must stay a sequential loop. The read and the write have the very same
    SHAPE as in the accepted inner case -- what differs is that ``j`` varies inside this loop's
    body, so it may not be shared between the reading and the writing iteration. Dropping that
    guard makes this loop parallelize and silently miscompile.
    """
    sdfg = transpose_add.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert not maps_with_param(sdfg, 'i'), "carried outer transpose axis i must NOT become a Map"
    assert loops_with_var(sdfg, 'i'), "carried outer transpose axis i must remain a sequential LoopRegion"

    n = 9
    rng = np.random.default_rng(2)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    ref = transpose_add_ref(aa, bb)
    sdfg(aa=aa, bb=bb, N=n)
    assert np.allclose(aa, ref)


def test_outer_scaled_triangular_transpose_axis_stays_sequential():
    """Same refusal for TSVC ``s114``'s own outer axis."""
    sdfg = transpose_add_scaled_triangular.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert not maps_with_param(sdfg, 'i'), "s114's outer axis i must NOT become a Map"
    assert loops_with_var(sdfg, 'i')

    n, nb, v = 16, 4, 4
    rng = np.random.default_rng(3)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    ref = transpose_add_scaled_triangular_ref(aa, bb, nb, v)
    sdfg(aa=aa, bb=bb, N=n, NB=nb, V=v)
    assert np.allclose(aa, ref)


@dace.program
def transpose_shifted(A: dace.float64[N, N]):
    for i in range(1, N):
        A[i, C] = A[C, i - 1] + 1.0


def transpose_shifted_ref(a: np.ndarray, c: int) -> np.ndarray:
    out = a.copy()
    for i in range(1, a.shape[0]):
        out[i, c] = out[c, i - 1] + 1.0
    return out


def test_shifted_transpose_carried_at_distance_one_stays_sequential():
    """``A[i, C] = A[C, i-1]``: a single loop, no inner varying symbol, alias at ``p == q + 1``.

    The collision system is the same transpose shape that certifies distance 0, shifted by one:
    the read hits the write's element exactly one iteration later. The certificate's constant term
    is what rules it out, so this pins the arithmetic rather than the ``varying`` guard. A
    dependence at distance 1 is carried and the loop must stay sequential.
    """
    sdfg = transpose_shifted.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert not maps_with_param(sdfg, 'i'), "distance-1 carried dependence must NOT become a Map"
    assert loops_with_var(sdfg, 'i'), "distance-1 carried dependence must remain a sequential LoopRegion"

    n, c = 8, 3
    a = np.random.default_rng(4).random((n, n))
    ref = transpose_shifted_ref(a, c)
    sdfg(A=a, N=n, C=c)
    assert np.allclose(a, ref)


if __name__ == '__main__':
    test_inner_axis_alias_at_distance_zero_maps()
    test_scaled_triangular_inner_axis_maps()
    test_outer_transpose_axis_stays_sequential()
    test_outer_scaled_triangular_transpose_axis_stays_sequential()
    test_shifted_transpose_carried_at_distance_one_stays_sequential()
