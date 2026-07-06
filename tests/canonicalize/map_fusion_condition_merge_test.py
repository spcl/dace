# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Map fusion with guarded map bodies, and merging same-condition in-map guards.

Two separate loops guarded by the SAME condition

    for i in range(N):
        if cond: b[i] = ...
    for i in range(N):
        if cond: c[i] = ...

become two parallel maps whose bodies each carry an ``if cond`` guard. This
file pins two behaviors:

1. ``MapFusion`` fuses the two if-bodied maps into ONE map (it is not blocked by
   the guards in the map bodies) -- ``test_mapfusion_fuses_two_if_bodied_maps``.

2. The two same-condition guards, now co-located inside that one map, SHOULD
   merge into a single ``if cond: {b; c}`` -- the map analogue of
   ``ConditionFusion``'s CFG-level ``if c: s1; if c: s2 -> if c: {s1; s2}``.
   This does NOT happen today: after fusion the guards live in two separate
   nested SDFGs under the map, so ConditionFusion's pattern (two consecutive
   ``ConditionalBlock``s in one CFG) never matches. ``test_*_merge_within_map``
   is marked ``xfail(strict=True)`` to track the gap; it flips to a hard pass
   the moment an in-map condition merge is wired in.

Numerical correctness holds in every case (the un-merged double-guard form is
still correct, just not maximally canonicalized).
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes import canonicalize

N = dace.symbol('N')


# --------------------------------------------------------------------------
# Kernels: two separate loops guarded by the same condition
# --------------------------------------------------------------------------
@dace.program
def two_guarded_loops(cond: dace.int32, a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Map-invariant guard: after fusion + merge + hoist the ideal form is
    ``if cond: { map: {b; c} }``."""
    for i in range(N):
        if cond > 0:
            b[i] = a[i] + 1.0
    for i in range(N):
        if cond > 0:
            c[i] = a[i] * 2.0


@dace.program
def two_guarded_loops_idx(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Index-dependent guard: cannot hoist out, so the ideal form keeps the
    merged guard inside the fused map: ``map[i]: { if i%2==0: {b; c} }``."""
    for i in range(N):
        if i % 2 == 0:
            b[i] = a[i] + 1.0
    for i in range(N):
        if i % 2 == 0:
            c[i] = a[i] * 2.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _top_maps(sdfg: dace.SDFG):
    return [
        n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None
    ]


def _num_condblocks(sdfg: dace.SDFG) -> int:
    return sum(1 for cb in sdfg.all_control_flow_regions(recursive=True) if isinstance(cb, ConditionalBlock))


# --------------------------------------------------------------------------
# (1) MapFusion fuses two if-bodied maps -- WORKS
# --------------------------------------------------------------------------
def test_mapfusion_fuses_two_if_bodied_maps():
    n = 16
    sdfg = two_guarded_loops.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)

    assert len(_top_maps(sdfg)) == 1, ("MapFusion must fuse two same-shape maps into one even though their "
                                       "bodies carry an if-guard")

    rng = np.random.default_rng(0)
    a = rng.random(n)
    for cond in (1, 0):
        b, c = np.zeros(n), np.zeros(n)
        sdfg(cond=np.int32(cond), a=a.copy(), b=b, c=c, N=n)
        assert np.allclose(b, np.where(cond > 0, a + 1.0, 0.0)), f"b mismatch cond={cond}"
        assert np.allclose(c, np.where(cond > 0, a * 2.0, 0.0)), f"c mismatch cond={cond}"


def test_mapfusion_fuses_two_if_bodied_maps_index_dependent():
    n = 16
    sdfg = two_guarded_loops_idx.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)

    assert len(_top_maps(sdfg)) == 1, "MapFusion must fuse the two index-guarded maps into one"

    rng = np.random.default_rng(1)
    a = rng.random(n)
    b, c = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    even = np.arange(n) % 2 == 0
    assert np.allclose(b, np.where(even, a + 1.0, 0.0)), "b mismatch"
    assert np.allclose(c, np.where(even, a * 2.0, 0.0)), "c mismatch"


# --------------------------------------------------------------------------
# (2) Merge same-condition guards WITHIN the fused map -- KNOWN GAP
# --------------------------------------------------------------------------
@pytest.mark.xfail(reason="in-map condition merge not implemented: after MapFusion co-locates two same-condition "
                   "guards inside one map, they sit in two separate nested SDFGs, so ConditionFusion's "
                   "two-consecutive-ConditionalBlocks pattern never matches and the guards are not merged "
                   "into a single `if cond: {b; c}`.",
                   strict=True)
def test_two_guards_merge_within_map():
    sdfg = two_guarded_loops_idx.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    # Desired canonical form: one fused map carrying ONE merged guard.
    assert len(_top_maps(sdfg)) == 1
    assert _num_condblocks(sdfg) == 1, ("the two same-condition guards inside the fused map should merge into one "
                                        "(map[i]: { if cond: {b; c} })")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
