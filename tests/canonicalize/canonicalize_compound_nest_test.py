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
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


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
    """Structural: the ``beg, end`` iedge assignments (their post-
    promotion symbol forms) must NOT appear on iedges at the SDFG top
    level -- they depend on ``i`` and must stay inside the ``i`` loop."""
    sdfg = compound_nest_loops.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # Top-level iedge assignment RHSes must not reference 'i' (because
    # if they did, they would be invalid at SDFG scope where 'i' is
    # undeclared). Equivalently: no top-level iedge should carry a per-
    # iteration bound expression.
    for e in sdfg.edges():
        for lhs, rhs in e.data.assignments.items():
            assert 'i' not in {str(s) for s in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}, \
                f'per-i bound {lhs} = {rhs} leaked to SDFG top level'


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
    assert _nmaps(sdfg) >= 1, 'outer parallel i-map was lost during canonicalize'


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
