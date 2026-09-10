# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Inlining a nested SDFG whose connectors describe only the window the edge memlet selects.

Such nested SDFGs are assembled under the earlier nested SDFG semantics: the connector is the part
of the container the edge selects, and the memlets inside are written relative to it. Inlining has
to keep the window, including in the nested SDFGs further down that describe the same container.
"""
import numpy as np

import dace
from dace import nodes
from dace.memlet import Memlet
from dace.sdfg import InterstateEdge, utils as sdutil
from dace.transformation.interstate import InlineSDFG

N = 20
M = 13


def _windowed_sdfg(symbolic_outer: bool = False) -> dace.SDFG:
    """
    Builds ``B[1:14, 4:17] = 2 * A[3:16, 2:15]`` with the window taken by the connectors of a nested
    SDFG, whose body is a map over a multi-state nested SDFG that also describes the window.
    """
    sdfg = dace.SDFG('inline_windowed_connector')
    if symbolic_outer:
        # The containers are described in symbols the nested SDFGs do not know yet
        outer_size = dace.symbol('outer_size')
        sdfg.add_array('A', (outer_size, outer_size), dace.float64)
        sdfg.add_array('B', (outer_size, outer_size), dace.float64)
        strides = (outer_size, 1)
    else:
        sdfg.add_array('A', (N, N), dace.float64)
        sdfg.add_array('B', (N, N), dace.float64)
        strides = (N, 1)

    child = dace.SDFG('child')
    child.add_array('a', (M, M), dace.float64, strides=strides)
    child.add_array('b', (M, M), dace.float64, strides=strides)
    child.add_symbol('i', dace.int64)
    child.add_symbol('j', dace.int64)
    first = child.add_state('first')
    second = child.add_state('second')
    # The condition reads the window too, so meta accesses have to follow the window as well
    child.add_edge(first, second, InterstateEdge(condition='a[i, j] > 0'))
    tasklet = second.add_tasklet('double', {'inp'}, {'out'}, 'out = 2 * inp')
    second.add_edge(second.add_read('a'), None, tasklet, 'inp', Memlet('a[i, j]'))
    second.add_edge(tasklet, 'out', second.add_write('b'), None, Memlet('b[i, j]'))

    inner = dace.SDFG('inner')
    inner.add_array('a', (M, M), dace.float64, strides=strides)
    inner.add_array('b', (M, M), dace.float64, strides=strides)
    istate = inner.add_state('body')
    me, mx = istate.add_map('elements', dict(i=f'0:{M}', j=f'0:{M}'))
    cnode = istate.add_nested_sdfg(child, {'a'}, {'b'}, {'i': 'i', 'j': 'j'})
    istate.add_memlet_path(istate.add_read('a'), me, cnode, dst_conn='a', memlet=Memlet('a[i, j]'))
    istate.add_memlet_path(cnode, mx, istate.add_write('b'), src_conn='b', memlet=Memlet('b[i, j]'))

    state = sdfg.add_state('call')
    nsdfg = state.add_nested_sdfg(inner, {'a'}, {'b'})
    state.add_edge(state.add_read('A'), None, nsdfg, 'a', Memlet(f'A[3:{3 + M}, 2:{2 + M}]'))
    state.add_edge(nsdfg, 'b', state.add_write('B'), None, Memlet(f'B[1:{1 + M}, 4:{4 + M}]'))
    return sdfg


def _run_and_check(sdfg: dace.SDFG, **symbols):
    A = np.random.rand(N, N) + 1.0
    B = np.zeros((N, N))
    sdfg(A=A, B=B, **symbols)
    expected = np.zeros((N, N))
    expected[1:1 + M, 4:4 + M] = 2 * A[3:3 + M, 2:2 + M]
    assert np.allclose(B, expected)


def test_inline_windowed_connector():
    sdfg = _windowed_sdfg()
    applied = sdfg.apply_transformations_repeated(InlineSDFG)
    assert applied == 1

    # The window is gone: what is left addresses the containers of the parent directly
    state = sdfg.node(0)
    child = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    for cname, outer in (('a', 'A'), ('b', 'B')):
        assert child.sdfg.arrays[cname].shape == (N, N)
        assert child.sdfg.arrays[cname].strides == (N, 1)
    edges = {e.dst_conn: e.data for e in state.in_edges(child)}
    assert edges['a'].data == 'A'
    assert str(edges['a'].subset) == 'i + 3, j + 2'
    inner_reads = [e.data for s in child.sdfg.states() for e in s.edges() if e.data.data == 'a']
    assert inner_reads and all(str(m.subset) == 'i + 3, j + 2' for m in inner_reads)
    inner_writes = [e.data for s in child.sdfg.states() for e in s.edges() if e.data.data == 'b']
    assert inner_writes and all(str(m.subset) == 'i + 1, j + 4' for m in inner_writes)
    condition = next(e.data.condition.as_string for e in child.sdfg.edges() if not e.data.is_unconditional())
    assert condition.replace(' ', '').replace('(', '').replace(')', '') == 'a[i+3,j+2]>0'

    sdfg.validate()
    _run_and_check(sdfg)


def test_inline_windowed_connector_simplify():
    sdfg = _windowed_sdfg()
    sdfg.simplify()
    sdfg.validate()
    _run_and_check(sdfg)


def test_inline_windowed_connector_symbolic_outer():
    """The containers are described in a symbol of the parent, which the nested SDFGs have to be given."""
    sdfg = _windowed_sdfg(symbolic_outer=True)
    assert sdfg.apply_transformations_repeated(InlineSDFG) == 1
    state = sdfg.node(0)
    child = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    assert str(child.sdfg.arrays['a'].shape[0]) == 'outer_size'
    assert 'outer_size' in child.sdfg.symbols
    assert str(child.symbol_mapping['outer_size']) == 'outer_size'
    sdfg.validate()
    _run_and_check(sdfg, outer_size=N)


def test_inline_squeezed_window():
    """A window taking a single element of a dimension, which the connector does not have."""
    sdfg = dace.SDFG('inline_squeezed_window')
    sdfg.add_array('A', (4, N, N), dace.float64)
    sdfg.add_array('B', (N, N), dace.float64)

    inner = dace.SDFG('inner')
    inner.add_array('a', (M, M), dace.float64, strides=(N, 1))
    inner.add_array('b', (M, M), dace.float64, strides=(N, 1))
    istate = inner.add_state('body')
    istate.add_mapped_tasklet('double',
                              dict(i=f'0:{M}', j=f'0:{M}'),
                              dict(inp=Memlet('a[i, j]')),
                              'out = 2 * inp',
                              dict(out=Memlet('b[i, j]')),
                              external_edges=True)

    state = sdfg.add_state('call')
    nsdfg = state.add_nested_sdfg(inner, {'a'}, {'b'})
    state.add_edge(state.add_read('A'), None, nsdfg, 'a', Memlet(f'A[2, 3:{3 + M}, 2:{2 + M}]'))
    state.add_edge(nsdfg, 'b', state.add_write('B'), None, Memlet(f'B[1:{1 + M}, 4:{4 + M}]'))

    assert sdutil.inline_sdfgs(sdfg) == 1
    assert not any(isinstance(n, nodes.NestedSDFG) for n in sdfg.node(0).nodes())
    sdfg.validate()

    A = np.random.rand(4, N, N)
    B = np.zeros((N, N))
    sdfg(A=A, B=B)
    expected = np.zeros((N, N))
    expected[1:1 + M, 4:4 + M] = 2 * A[2, 3:3 + M, 2:2 + M]
    assert np.allclose(B, expected)


if __name__ == '__main__':
    test_inline_windowed_connector()
    test_inline_windowed_connector_simplify()
    test_inline_windowed_connector_symbolic_outer()
    test_inline_squeezed_window()
