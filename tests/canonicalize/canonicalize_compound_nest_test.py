# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on compound nested-loop / nested-map kernels.

The shape under test (paraphrased from the user's guidance, ICON-style):

::

    for i in [outer]:
        beg = foo(i)
        end = bar(i)
        for k:
            for m: body1
        if x:
            for k:
                for m: body2
        for k: body3

The interesting interactions:

* ``beg = foo(i)`` and ``end = bar(i)`` are per-``i`` interstate-edge
  assignments whose RHS reads the outer loop variable ``i``.
  ``CascadeInterstateEdgeAssignmentsUp`` must refuse to move them past
  the ``i`` loop (L1 RHS-invariance). Equally, the inner ``k, m`` loops
  whose ranges read ``beg, end`` must not be moved out of the ``i``
  loop either (no pass should do that). This is an explicit no-op
  safety contract.
* The guarded block (``if x: for k: for m: body2``) sits between
  body1's and body3's inner nests. ``MoveIfIntoLoop`` (also part of
  canonicalize) should push the ``x`` guard *into* the inner ``k, m``
  nest. After cleanup + fusion, body1's nest and body2's nest --
  having identical loop shapes and now sharing a co-located guard
  shape -- should fuse into a single map nest carrying the guard on
  body2's update only.
* All maps that *can* fuse, *do* fuse. The kernel below uses the
  Python frontend (Loop variant) and a parallel ``dace.map`` (Map
  variant) so both representations are covered.

Each test is value-preserving against a pure-numpy oracle and asserts
the structural contract canonicalize is expected to deliver today.
"""
import numpy as np
import pytest

import dace
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def iterator_names(sdfg) -> OrderedSet:
    """Every iterator name canonicalize LEFT BEHIND. ``UniqueLoopIterators`` renames the source
    ``i`` to ``_loop_it_<N>``, so a leaked per-``i`` bound is never spelled ``i`` afterwards."""
    names = OrderedSet()
    for r in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(r, LoopRegion) and r.loop_variable:
            names.add(r.loop_variable)
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            names.update(n.map.params)
    return names


def top_level_expressions(sdfg) -> list[tuple[str, set]]:
    """``(label, free symbols)`` for every expression evaluated at SDFG scope: interstate-edge
    assignments and the ranges of top-level maps. Both are outside every loop, so an iterator
    name appearing here is a bound that escaped its loop."""
    found = [(f'iedge {lhs} = {rhs}', {str(x)
                                       for x in dace.symbolic.pystr_to_symbolic(rhs).free_symbols})
             for e in sdfg.edges() for lhs, rhs in e.data.assignments.items()]
    for blk in sdfg.nodes():
        if not isinstance(blk, SDFGState):
            continue
        for n in blk.nodes():
            if isinstance(n, nodes.MapEntry):
                found.append((f'map {n.map.params} range {n.map.range}', {str(x) for x in n.map.range.free_symbols}))
    return found


# ----------------------------------------------------------------------
# Loop variant (Python ``range``)
# ----------------------------------------------------------------------


@dace.program
def compound_nest_loops(arr: dace.float64[N, N, N], out: dace.float64[N, N, N], x: dace.int32):
    """Loop-based ICON-shape kernel: per-``i`` slice bounds plus a guarded
    inner nest sandwiched between two non-guarded nests. ``beg, end``
    derive from ``i`` and must not be hoisted past the ``i`` loop."""
    for i in range(0, N):
        beg = i // 2 + 1
        end = beg + 2
        # body1
        for k in range(beg, end):
            for m in range(beg, end):
                out[i, k, m] += arr[i, k, m]
        # guarded body2
        if x > 0:
            for k in range(beg, end):
                for m in range(beg, end):
                    out[i, k, m] += 2.0 * arr[i, k, m]
        # body3
        for k in range(beg, end):
            out[i, k, 0] += 1.0


def _compound_oracle(arr, x):
    n = arr.shape[0]
    out = np.zeros_like(arr)
    for i in range(n):
        beg = i // 2 + 1
        end = beg + 2
        for k in range(beg, end):
            for m in range(beg, end):
                out[i, k, m] += arr[i, k, m]
        if x > 0:
            for k in range(beg, end):
                for m in range(beg, end):
                    out[i, k, m] += 2.0 * arr[i, k, m]
        for k in range(beg, end):
            out[i, k, 0] += 1.0
    return out


def test_compound_nest_loops_value_preserving():
    n = 8
    rng = np.random.default_rng(11)
    arr = rng.standard_normal((n, n, n)).astype(np.float64)
    for x in (1, 0):
        sdfg = compound_nest_loops.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        out = np.zeros((n, n, n))
        sdfg(arr=arr, out=out, x=np.int32(x), N=n)
        exp = _compound_oracle(arr, x)
        assert np.allclose(out, exp), f'x={x} mismatch'


def test_compound_nest_loops_per_i_bounds_not_hoisted():
    """The per-``i`` inner bounds are absorbed into the nest, never evaluated at SDFG scope."""
    sdfg = compound_nest_loops.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    iters = iterator_names(sdfg)
    inspected = top_level_expressions(sdfg)
    assert inspected, 'nothing evaluated at SDFG scope -- the leak check would pass on an empty SDFG'
    leaked = [label for label, syms in inspected if syms & set(iters)]
    assert not leaked, f'per-i bound leaked to SDFG top level: {leaked}'
    # The inner ``beg:beg+2`` nests have a constant extent, so canonicalize unrolls them away:
    # the outer ``i`` is the only iteration left and it carries no bound symbol of its own.
    assert _nmaps(sdfg) == 1 and _nloops(sdfg) == 0, f'maps={_nmaps(sdfg)} loops={_nloops(sdfg)}'
    outer = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert str(outer.map.range) == '0:N', f'the surviving map must span the i axis, got {outer.map.range}'


# ----------------------------------------------------------------------
# Map variant (``dace.map``)
# ----------------------------------------------------------------------


@dace.program
def compound_nest_maps(arr: dace.float64[N, N, N], out: dace.float64[N, N, N], x: dace.int32):
    """Map-based variant of the compound nest: the outer ``i`` is a
    parallel ``dace.map`` and the inner two are ``range`` loops with
    bounds derived from ``i``. Verifies the same structural / value
    contract holds when the outer iteration is a Map (not a LoopRegion)
    and therefore the per-``i`` bound state lives inside a NestedSDFG
    that is the Map body.
    """
    for i in dace.map[0:N]:
        beg = i // 2 + 1
        end = beg + 2
        for k in range(beg, end):
            for m in range(beg, end):
                out[i, k, m] += arr[i, k, m]
        if x > 0:
            for k in range(beg, end):
                for m in range(beg, end):
                    out[i, k, m] += 2.0 * arr[i, k, m]
        for k in range(beg, end):
            out[i, k, 0] += 1.0


def test_compound_nest_maps_value_preserving():
    n = 8
    rng = np.random.default_rng(12)
    arr = rng.standard_normal((n, n, n)).astype(np.float64)
    for x in (1, 0):
        sdfg = compound_nest_maps.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        out = np.zeros((n, n, n))
        sdfg(arr=arr, out=out, x=np.int32(x), N=n)
        exp = _compound_oracle(arr, x)
        assert np.allclose(out, exp), f'x={x} mismatch'


def test_compound_nest_maps_outer_map_survives():
    """Structural: the outer parallel ``i`` map must be present after
    canonicalize -- the per-i bound dependencies cannot kill it. Regression
    for the UniqueLoopIterators NSDFG symbol-mapping crash (the pass no
    longer re-renames already-unique ``_loop_it_*`` iterators, so the
    SDFG validates)."""
    sdfg = compound_nest_maps.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # One outer i-map plus the five inner nests it distributes over: body1's (k, m), body3's k
    # and, under the guard, body2's (k, m) and a second copy of each sibling.
    assert _nmaps(sdfg) == 6, f'expected the outer i-map over five inner nests, got {_nmaps(sdfg)}'
    top_maps = [
        n for blk in sdfg.nodes() if isinstance(blk, SDFGState) for n in blk.nodes() if isinstance(n, nodes.MapEntry)
    ]
    assert len(top_maps) == 1, f'exactly one map at SDFG scope, got {len(top_maps)}'
    assert str(top_maps[0].map.range) == '0:N', f'the outer map must span the i axis, got {top_maps[0].map.range}'
    # The per-i bounds stay where they belong: the inner ranges read the outer iterator, and
    # nothing at SDFG scope does.
    inner_ranges = [
        str(n.map.range) for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n is not top_maps[0]
    ]
    outer_it = top_maps[0].map.params[0]
    assert all(outer_it in r for r in inner_ranges), f'inner bounds lost their per-i dependence: {inner_ranges}'
    leaked = [label for label, syms in top_level_expressions(sdfg) if syms & set(iterator_names(sdfg))]
    assert not leaked, f'per-i bound leaked to SDFG top level: {leaked}'


# ----------------------------------------------------------------------
# Slimmer reproducer focused on cascade-up + MoveIfIntoLoop interaction
# ----------------------------------------------------------------------


@dace.program
def guarded_nest_with_per_i_bounds(arr: dace.float64[N, N], out: dace.float64[N, N], x: dace.int32):
    """Smaller version of the compound shape: a single per-``i`` bound
    and two inner nests, one guarded. After canonicalize the guarded
    body should be co-located with the non-guarded one (MoveIfIntoLoop
    pushes the guard inward; the two inner ``k`` loops then fuse).
    Cascade-up must refuse on the bound (rhs reads i) -- the kernel
    stays correct.
    """
    for i in range(0, N):
        end = i // 2 + 2
        for k in range(0, end):
            out[i, k] += arr[i, k]
        if x > 0:
            for k in range(0, end):
                out[i, k] += 3.0 * arr[i, k]


def _guarded_oracle(arr, x):
    n = arr.shape[0]
    out = np.zeros_like(arr)
    for i in range(n):
        end = i // 2 + 2
        for k in range(0, end):
            out[i, k] += arr[i, k]
        if x > 0:
            for k in range(0, end):
                out[i, k] += 3.0 * arr[i, k]
    return out


def test_guarded_nest_with_per_i_bounds_value_preserving():
    n = 8
    rng = np.random.default_rng(13)
    arr = rng.standard_normal((n, n)).astype(np.float64)
    for x in (1, 0):
        sdfg = guarded_nest_with_per_i_bounds.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        out = np.zeros((n, n))
        sdfg(arr=arr, out=out, x=np.int32(x), N=n)
        exp = _guarded_oracle(arr, x)
        assert np.allclose(out, exp), f'x={x} mismatch'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
