# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization round-trip for a guarded multi-statement loop body.

A loop whose body is a single guard wrapping several statements

    for i in range(N):
        if cond:
            st1
            st2

must canonicalize through the full split -> fuse -> merge -> hoist chain:

* ``SplitStatements`` replicates the guard once per independent output so the
  statements can fission (and snapshots a forward-read WAR so a
  data-*dependent*-but-splittable body fissions too);
* ``MapFusion`` recombines the now-independent per-statement maps;
* ``ConditionFusion`` merges the replicated ``if cond: st1`` / ``if cond: st2``
  back into one ``if cond: {st1; st2}``;
* ``MoveMapInvariantIfUp`` / ``MoveLoopInvariantIfUp`` hoist a map-invariant
  guard out of the map (``if cond: { map: {st1; st2} }``); an index-dependent
  guard that cannot hoist stays merged *inside* the map
  (``map[i]: { if i%2==0: {st1; st2} }``).

The canonical form is therefore always ONE parallel map carrying BOTH
statements under ONE conditional block -- never two split maps or two
un-merged guards. Each test asserts that structure and numerical equality to a
numpy oracle (guard taken and not taken).
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes import canonicalize

N = dace.symbol('N')


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@dace.program
def guarded_independent(cond: dace.int32, a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Two data-independent outputs under a map-invariant guard."""
    for i in range(N):
        if cond > 0:
            b[i] = a[i] + 1.0
            c[i] = a[i] * 2.0


@dace.program
def guarded_index_dependent(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Guard depends on the map index, so it cannot hoist out of the map: the
    two statements must merge into a single guard *inside* the map body."""
    for i in range(N):
        if i % 2 == 0:
            b[i] = a[i] + 1.0
            c[i] = a[i] * 2.0


@dace.program
def guarded_forward_read_war(cond: dace.int32, a: dace.float64[N], d: dace.float64[N]):
    """Data-dependent but splittable: ``d`` reads ``a[i+1]`` ahead of the
    ``a[i]`` write (a forward-read WAR, TSVC s1244 shape) under a guard.
    SplitStatements snapshots the read-ahead so the body still fissions."""
    for i in range(N - 1):
        if cond > 0:
            d[i] = a[i] + a[i + 1]
            a[i] = a[i] * 2.0


# --------------------------------------------------------------------------
# Structural helpers
# --------------------------------------------------------------------------
def _top_map_entries(sdfg: dace.SDFG):
    """Top-level (non-nested) MapEntry nodes across every state."""
    return [
        n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None
    ]


def _conditional_blocks(sdfg: dace.SDFG):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, ConditionalBlock)]


def _guard_over_a_map(sdfg: dace.SDFG) -> bool:
    """A ConditionalBlock whose branch body contains a top-level map (guard hoisted out)."""
    for cb in _conditional_blocks(sdfg):
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


def _canon():
    return lambda s: canonicalize(s, validate=True)


CANON = [pytest.param(_canon(), id='full')]


# --------------------------------------------------------------------------
# Item 1 (independent) + Item 2 (hoist-out merge)
# --------------------------------------------------------------------------
@pytest.mark.parametrize('canon', CANON)
def test_guarded_independent_hoists_and_merges(canon):
    n = 16
    sdfg = guarded_independent.to_sdfg(simplify=False)
    canon(sdfg)

    # ONE parallel map carrying both statements, ONE merged guard, hoisted out.
    assert len(_top_map_entries(sdfg)) == 1, "the two guarded statements must fuse into one map"
    assert len(_conditional_blocks(sdfg)) == 1, "the two replicated guards must merge into one"
    assert _guard_over_a_map(sdfg), "a map-invariant guard must hoist out of the map"

    rng = np.random.default_rng(0)
    a = rng.random(n)
    for cond in (1, 0):
        b = np.zeros(n)
        c = np.zeros(n)
        sdfg(cond=np.int32(cond), a=a.copy(), b=b, c=c, N=n)
        b_ref = np.where(cond > 0, a + 1.0, 0.0)
        c_ref = np.where(cond > 0, a * 2.0, 0.0)
        assert np.allclose(b, b_ref), f"b mismatch cond={cond}"
        assert np.allclose(c, c_ref), f"c mismatch cond={cond}"


@pytest.mark.parametrize('canon', CANON)
def test_guarded_index_dependent_merges_in_map(canon):
    n = 16
    sdfg = guarded_index_dependent.to_sdfg(simplify=False)
    canon(sdfg)

    # Guard cannot hoist (depends on i): one map, one guard, merged INSIDE the map.
    assert len(_top_map_entries(sdfg)) == 1, "the two guarded statements must fuse into one map"
    assert len(_conditional_blocks(sdfg)) == 1, "the two in-map guards must merge into one"
    assert _guard_inside_a_map(sdfg), "an index-dependent guard must stay merged inside the map"
    assert not _guard_over_a_map(sdfg), "an index-dependent guard must not hoist over the map"

    rng = np.random.default_rng(1)
    a = rng.random(n)
    b = np.zeros(n)
    c = np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    even = np.arange(n) % 2 == 0
    assert np.allclose(b, np.where(even, a + 1.0, 0.0)), "b mismatch"
    assert np.allclose(c, np.where(even, a * 2.0, 0.0)), "c mismatch"


# --------------------------------------------------------------------------
# Item 1 (data-dependent but splittable)
# --------------------------------------------------------------------------
@pytest.mark.parametrize('canon', CANON)
def test_guarded_forward_read_war_splits_and_merges(canon):
    n = 16
    sdfg = guarded_forward_read_war.to_sdfg(simplify=False)
    canon(sdfg)

    # The WAR is snapshot-broken so the body fissions, then re-fuses: one map,
    # one merged guard hoisted out.
    assert len(_top_map_entries(sdfg)) == 1, "the split WAR body must re-fuse into one map"
    assert len(_conditional_blocks(sdfg)) == 1, "the replicated guards must merge into one"
    assert _guard_over_a_map(sdfg), "the map-invariant guard must hoist out"

    rng = np.random.default_rng(2)
    a0 = rng.random(n)
    for cond in (1, 0):
        a = a0.copy()
        d = np.zeros(n)
        sdfg(cond=np.int32(cond), a=a, d=d, N=n)
        a_ref = a0.copy()
        d_ref = np.zeros(n)
        if cond > 0:
            for i in range(n - 1):
                d_ref[i] = a_ref[i] + a_ref[i + 1]
                a_ref[i] = a_ref[i] * 2.0
        assert np.allclose(a, a_ref), f"a mismatch cond={cond}"
        assert np.allclose(d, d_ref), f"d mismatch cond={cond}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
