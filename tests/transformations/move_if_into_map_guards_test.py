# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Preconditions ``MoveIfIntoMap`` must enforce before rewriting.

Each case here was accepted by ``can_be_applied`` and then either crashed
mid-rewrite or produced a silently wrong / invalid SDFG.
"""
import dace
import numpy as np
import pytest

from dace.properties import CodeBlock
from dace.sdfg.nodes import MapEntry
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = 8


def _apply(sdfg):
    return PatternMatchAndApplyRepeated([MoveIfIntoMap()]).apply_pass(sdfg, {})


def test_nested_inner_maps_are_refused():
    """``if c: for j: for k:`` -- a 2-level nest inside the guard.

    Normalizing the outer map nests the inner one away, so the snapshot loop
    then calls ``exit_node`` on a node already removed from the state. The
    rewrite would raise having already mutated the branch state.
    """

    @dace.program
    def kern(b: dace.float64[N, N], flag: dace.int64):
        for t in dace.map[0:1]:
            if flag > 0:
                for j in dace.map[0:N]:
                    for k in dace.map[0:N]:
                        b[j, k] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    before = sdfg.to_json()
    _apply(sdfg)  # must not raise
    sdfg.validate()
    assert sdfg.to_json() == before, 'a refused match must not mutate the graph'


def test_data_dependent_guard_does_not_produce_an_invalid_sdfg():
    """A guard whose definition reads data cannot be threaded through
    ``symbol_mapping``; pushing it in leaves the copy reading a name that does
    not exist inside the inner body."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N], thr: dace.float64):
        for i in dace.map[0:N]:
            if a[i] > thr:
                for j in dace.map[0:N]:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    _apply(sdfg)  # must not raise
    sdfg.validate()

    a = np.array([1.0, -1.0] * (N // 2))
    out = np.zeros((N, N))
    sdfg(a=a, b=out, thr=0.0)
    assert np.allclose(out, np.where((a > 0.0)[:, None], 1.0, 0.0), rtol=1e-9, atol=1e-9, equal_nan=True)


def _two_predecessor_guard():
    """``S1 -[k=0]-> cond <-[k=1]- S2`` with guard ``k == 1``."""
    sdfg = dace.SDFG('twodefs')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('sel', dace.int64)
    sdfg.add_array('b', [N], dace.float64)

    outer = dace.SDFG('outer')
    outer.add_symbol('k', dace.int64)
    outer.add_array('bo', [N], dace.float64)
    outer.add_symbol('sel', dace.int64)
    entry = outer.add_state('entry', is_start_block=True)
    s1 = outer.add_state('s1')
    s2 = outer.add_state('s2')
    cb = ConditionalBlock('guard')
    outer.add_node(cb)
    # Diamond: both predecessors define the guard symbol, differently.
    outer.add_edge(entry, s1, InterstateEdge(condition=CodeBlock('sel > 0')))
    outer.add_edge(entry, s2, InterstateEdge(condition=CodeBlock('sel <= 0')))
    outer.add_edge(s1, cb, InterstateEdge(assignments={'k': '0'}))
    outer.add_edge(s2, cb, InterstateEdge(assignments={'k': '1'}))
    region = ControlFlowRegion('then', sdfg=outer)
    bs = region.add_state('w', is_start_block=True)
    me, mx = bs.add_map('inner', dict(j='0:%d' % N))
    t = bs.add_tasklet('one', {}, {'o'}, 'o = 1.0')
    bs.add_edge(me, None, t, None, dace.Memlet())
    bs.add_memlet_path(t, mx, bs.add_access('bo'), src_conn='o', memlet=dace.Memlet('bo[j]'))
    cb.add_branch(CodeBlock('k == 1'), region)

    st = sdfg.add_state('main', is_start_block=True)
    ome, omx = st.add_map('outer_map', dict(i='0:1'))
    ns = st.add_nested_sdfg(outer, {}, {'bo'}, symbol_mapping={'k': 'k', 'sel': 'sel'})
    st.add_edge(ome, None, ns, None, dace.Memlet())
    st.add_memlet_path(ns, omx, st.add_access('b'), src_conn='bo', memlet=dace.Memlet('b[0:%d]' % N))
    return sdfg


def test_guard_symbol_defined_on_two_edges_is_refused():
    """Two predecessors defining the guard symbol differently.

    ``apply`` collects definitions into a flat dict (last one wins) yet deletes
    every edge's copy, so both paths would evaluate the guard with whichever
    definition iteration order landed on.
    """
    sdfg = _two_predecessor_guard()
    assert _apply(sdfg) is None, 'ambiguous guard definition must be refused'


def test_guard_symbol_used_by_a_map_range_is_not_deleted():
    """The moved guard symbol must survive if dataflow still reads it.

    ``remove_symbol`` also strips the name from the parent's symbol_mapping, so
    deleting a symbol that a map range or memlet still mentions leaves it
    undefined at every level.
    """

    n2 = dace.symbol('n2')

    @dace.program
    def kern(b: dace.float64[N]):
        for i in dace.map[0:1]:
            if n2 > 0:
                for j in dace.map[0:n2]:
                    b[j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    _apply(sdfg)
    sdfg.validate()  # would fail if n2 were removed while the range uses it

    out = np.zeros(N)
    sdfg(b=out, n2=N)
    assert np.allclose(out, np.ones(N), rtol=1e-9, atol=1e-9, equal_nan=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
