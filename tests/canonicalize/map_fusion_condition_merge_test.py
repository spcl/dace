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
   the guards in the map bodies).

2. The two same-condition guards, now co-located inside that one map, merge into
   a single ``if cond: {b; c}`` -- the map analogue of ``ConditionFusion``'s
   CFG-level ``if c: s1; if c: s2 -> if c: {s1; s2}``. After fusion the guards
   live in two separate nested SDFGs under the map, so ``NormalizeMapBody``
   first sequences those siblings into one nested SDFG (making the two
   ``ConditionalBlock``s consecutive), and ``ConditionFusion`` then folds them.
   A map-invariant merged guard hoists out (``if cond: { map: {b; c} }``); an
   index-dependent one stays merged inside the map (``map[i]: { if c: {b; c} }``).
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


def _guard_over_a_map(sdfg: dace.SDFG) -> bool:
    """A ConditionalBlock whose branch body contains a top-level map (guard hoisted out)."""
    for cb in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(cb, ConditionalBlock):
            continue
        for _, branch in cb.branches:
            for st in branch.all_states():
                if any(isinstance(n, nodes.MapEntry) and st.entry_node(n) is None for n in st.nodes()):
                    return True
    return False


def _guard_inside_a_map(sdfg: dace.SDFG) -> bool:
    """A ConditionalBlock inside a NestedSDFG that sits under a map scope (guard in-map)."""
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.NestedSDFG) and st.entry_node(n) is not None:
                if any(isinstance(b, ConditionalBlock) for b in n.sdfg.all_control_flow_regions(recursive=True)):
                    return True
    return False


# --------------------------------------------------------------------------
# (1) MapFusion fuses two if-bodied maps; NormalizeMapBody merges the guards
# --------------------------------------------------------------------------
def test_mapfusion_fuses_then_merges_invariant_guard():
    """Map-invariant guard: MapFusion fuses the two if-bodied maps, then
    NormalizeMapBody + ConditionFusion merge the two same-condition guards into
    one, which (being map-invariant) hoists OUT: ``if cond: { map: {b; c} }``."""
    n = 16
    sdfg = two_guarded_loops.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)

    assert _num_condblocks(sdfg) == 1, "the two same-condition guards must merge into one"
    assert _guard_over_a_map(sdfg), "the merged map-invariant guard must hoist out of the map"

    rng = np.random.default_rng(0)
    a = rng.random(n)
    for cond in (1, 0):
        b, c = np.zeros(n), np.zeros(n)
        sdfg(cond=np.int32(cond), a=a.copy(), b=b, c=c, N=n)
        assert np.allclose(b, np.where(cond > 0, a + 1.0, 0.0)), f"b mismatch cond={cond}"
        assert np.allclose(c, np.where(cond > 0, a * 2.0, 0.0)), f"c mismatch cond={cond}"


def test_mapfusion_fuses_then_merges_index_dependent_guard():
    """Index-dependent guard: cannot hoist, so the two same-condition guards
    merge INSIDE the single fused map: ``map[i]: { if i%2==0: {b; c} }``."""
    n = 16
    sdfg = two_guarded_loops_idx.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)

    assert len(_top_maps(sdfg)) == 1, "the two index-guarded maps must fuse into one"
    assert _num_condblocks(sdfg) == 1, "the two same-condition in-map guards must merge into one"
    assert _guard_inside_a_map(sdfg), "the merged index-dependent guard must stay inside the map"

    rng = np.random.default_rng(1)
    a = rng.random(n)
    b, c = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    even = np.arange(n) % 2 == 0
    assert np.allclose(b, np.where(even, a + 1.0, 0.0)), "b mismatch"
    assert np.allclose(c, np.where(even, a * 2.0, 0.0)), "c mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
