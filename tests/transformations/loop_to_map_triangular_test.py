# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` on triangular loop nests.

A triangular nest is one whose inner loop bounds depend on an outer loop index, e.g.
``for i: for j in range(i, N): ...``. The DOALL (parallel) axis of such a nest is
parallelizable: distinct iterations of that axis touch disjoint slabs of every array
(they share the same injective index -- typically ``i`` -- on one dimension), so lifting
it to a Map preserves semantics.

The tests below cover both directions:

* the parallel axis of a genuine triangular DOALL nest is accepted (becomes a Map), and

* an axis carrying a real loop-carried dependency -- even when the nest *looks* triangular
  -- stays a sequential loop. ``LoopToMap`` must never over-parallelize a recurrence.

The accept-side recognizer is ``_read_write_same_iteration`` in ``loop_to_map.py``: when a
read and a write to the same container share the same injective affine index of the
iteration variable on some dimension, a collision forces the reading and writing
iterations to coincide, so the overlap is confined to a single iteration (program order in
the map body preserves it) and never becomes a cross-iteration dependency.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')
M = dace.symbol('M')


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
# Accept side: triangular DOALL axes must become Maps.
# ---------------------------------------------------------------------------


@dace.program
def tri_doall(C: dace.float64[N, N], A: dace.float64[N, N]):
    for i in range(N):
        for j in range(i, N):
            C[i, j] = A[i, j] * 2.0


def test_triangular_doall_maps_both_axes():
    """``for i: for j in range(i, N)`` with no carried dependency: both axes parallelize."""
    sdfg = tri_doall.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert maps_with_param(sdfg, 'i'), "outer triangular DOALL axis i should become a Map"
    assert not loops_with_var(sdfg, 'i'), "outer axis i should no longer be a sequential loop"

    n = 7
    a = np.random.default_rng(0).random((n, n))
    c = np.zeros((n, n))
    ref = c.copy()
    for i in range(n):
        for j in range(i, n):
            ref[i, j] = a[i, j] * 2.0
    sdfg(C=c, A=a, N=n)
    assert np.allclose(c, ref)


@dace.program
def tri_prefix(B: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(i):
            B[i, i] = B[i, i] + B[i, j]


def test_triangular_read_write_slab_maps_outer():
    """Row ``i`` reads its prefix ``B[i, 0:i]`` and writes ``B[i, i]``.

    The read slab and the write differ (so the ``read == write`` fast path does not fire),
    yet both share the injective outer index ``i`` on dimension 0. Distinct rows never
    interfere, so the outer axis is DOALL. This is the case that specifically exercises
    ``_read_write_same_iteration``: without it, ``LoopToMap`` refuses with a spurious
    read-after-write conflict.
    """
    sdfg = tri_prefix.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert maps_with_param(sdfg, 'i'), "outer axis i should become a Map (triangular read/write slab)"
    assert not loops_with_var(sdfg, 'i')

    n = 9
    b = np.random.default_rng(1).random((n, n))
    ref = b.copy()
    for i in range(1, n):
        for j in range(i):
            ref[i, i] = ref[i, i] + ref[i, j]
    sdfg(B=b, N=n)
    assert np.allclose(b, ref)


@dace.program
def inner_carry_outer_doall(A: dace.float64[N, N]):
    for i in range(N):
        for j in range(1, N):
            A[i, j] = A[i, j - 1] + 1.0


def test_outer_doall_inner_recurrence_splits():
    """Outer ``i`` is DOALL; inner ``j`` carries a dependency within each row.

    The correct outcome parallelizes ONLY the outer axis and keeps the inner recurrence a
    sequential loop -- exactly the "parallelize the genuinely-parallel axis, keep the
    recurrence sequential" contract.
    """
    sdfg = inner_carry_outer_doall.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert maps_with_param(sdfg, 'i'), "outer DOALL axis i should become a Map"
    assert not maps_with_param(sdfg, 'j'), "inner recurrence axis j must stay a sequential loop"
    assert loops_with_var(sdfg, 'j'), "inner recurrence axis j must remain a LoopRegion"

    n = 6
    a = np.random.default_rng(2).random((n, n))
    ref = a.copy()
    for i in range(n):
        for j in range(1, n):
            ref[i, j] = ref[i, j - 1] + 1.0
    sdfg(A=a, N=n)
    assert np.allclose(a, ref)


# ---------------------------------------------------------------------------
# Reject side: a carried dependency on the axis must keep it sequential.
# ---------------------------------------------------------------------------


@dace.program
def row_recurrence(A: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(i, N):
            A[i, j] = A[i - 1, j] + 1.0


def test_triangular_looking_row_recurrence_stays_sequential():
    """Looks triangular (``range(i, N)``) but row ``i`` reads row ``i-1``: a genuine
    carried dependency across the outer axis. The outer ``i`` axis must stay a loop.

    The inner ``j`` axis IS parallel (each column independent for fixed ``i``) and may
    become a Map -- that is correct and does not violate the recurrence.
    """
    sdfg = row_recurrence.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert not maps_with_param(sdfg, 'i'), "carried outer axis i must NOT become a Map"
    assert loops_with_var(sdfg, 'i'), "carried outer axis i must remain a sequential LoopRegion"

    n = 6
    a = np.random.default_rng(3).random((n, n))
    ref = a.copy()
    for i in range(1, n):
        for j in range(i, n):
            ref[i, j] = ref[i - 1, j] + 1.0
    sdfg(A=a, N=n)
    assert np.allclose(a, ref)


@dace.program
def prefix_scan(B: dace.float64[N]):
    for i in range(1, N):
        B[i] = B[i] + B[i - 1]


def test_prefix_scan_stays_sequential():
    """Classic prefix-scan recurrence ``B[i] = B[i] + B[i-1]`` must not parallelize."""
    sdfg = prefix_scan.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert not maps_with_param(sdfg, 'i'), "prefix scan axis i must NOT become a Map"
    assert loops_with_var(sdfg, 'i'), "prefix scan axis i must remain a sequential LoopRegion"

    n = 12
    b = np.random.default_rng(4).random(n)
    ref = b.copy()
    for i in range(1, n):
        ref[i] = ref[i] + ref[i - 1]
    sdfg(B=b, N=n)
    assert np.allclose(b, ref)


if __name__ == '__main__':
    test_triangular_doall_maps_both_axes()
    test_triangular_read_write_slab_maps_outer()
    test_outer_doall_inner_recurrence_splits()
    test_triangular_looking_row_recurrence_stays_sequential()
    test_prefix_scan_stays_sequential()
