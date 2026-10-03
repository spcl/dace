# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end and structural tests for :class:`MapFission`.

These tests exercise :class:`MapFission` on SDFG shapes proven to fission in
``map_fission_test.py`` (two data-independent components, a Map containing a
multi-state NestedSDFG, and a multidimensional Map).  Where the shape tolerates
it, reads are routed through an ICON velocity-advection style neighbour-index
table (``out[e, k] = c1 * w[idx[e, 0], k] - c2 * w[idx[e, 1], k]``): a gathered
first dimension with a structured level dimension.  Each test checks numerical
equivalence against a pure-numpy oracle and asserts that the recursive
``MapEntry`` count strictly increased after fission.
"""

import dace
import numpy as np
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFission

_E = dace.symbol('E')
_L = dace.symbol('L')


def _count_map_entries(sdfg):
    """Count every :class:`MapEntry` reachable through the SDFG.

    :param sdfg: The SDFG to scan.
    :returns: The total number of map entries, including nested ones.
    """
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def test_two_independent_gather_components():
    """Two data-independent neighbour-gather components fission into two maps.

    Body per edge ``e``: a first component computes ``out1[e] = c1 * w[idx[e,
    0]] - c2 * w[idx[e, 1]]`` and an independent second component computes
    ``out2[e] = w[idx2[e]] + s[e]``.  Mirrors the ``test_inputs_outputs``
    shape: two tasklets sharing the outer map but with no data dependence.
    """
    sdfg = dace.SDFG('two_independent_gather_components')
    sdfg.add_array('w', [_E], dace.float64)
    sdfg.add_array('s', [_E], dace.float64)
    sdfg.add_array('idx', [_E, 2], dace.int32)
    sdfg.add_array('idx2', [_E], dace.int32)
    sdfg.add_array('out1', [_E], dace.float64)
    sdfg.add_array('out2', [_E], dace.float64)

    state = sdfg.add_state()
    w = state.add_read('w')
    s = state.add_read('s')
    idx = state.add_read('idx')
    idx2 = state.add_read('idx2')
    o1 = state.add_write('out1')
    o2 = state.add_write('out2')

    me, mx = state.add_map('edges', dict(e='0:E'))

    # Component 1: gather two neighbours of ``w`` via the ``idx`` table and
    # combine.  The full ``w`` is routed into the body and indexed with the
    # data-dependent table values inside the tasklet.
    t1 = state.add_tasklet('grad', {'wa', 'i0', 'i1'}, {'r'}, 'r = 1.5 * wa[i0] - 0.5 * wa[i1]')
    state.add_memlet_path(w, me, t1, dst_conn='wa', memlet=dace.Memlet(data='w', subset='0:E'))
    state.add_memlet_path(idx, me, t1, dst_conn='i0', memlet=dace.Memlet('idx[e, 0]'))
    state.add_memlet_path(idx, me, t1, dst_conn='i1', memlet=dace.Memlet('idx[e, 1]'))
    state.add_memlet_path(t1, mx, o1, src_conn='r', memlet=dace.Memlet('out1[e]'))

    t2 = state.add_tasklet('shift', {'wa', 'j', 'sn'}, {'r'}, 'r = wa[j] + sn')
    state.add_memlet_path(w, me, t2, dst_conn='wa', memlet=dace.Memlet(data='w', subset='0:E'))
    state.add_memlet_path(idx2, me, t2, dst_conn='j', memlet=dace.Memlet('idx2[e]'))
    state.add_memlet_path(s, me, t2, dst_conn='sn', memlet=dace.Memlet('s[e]'))
    state.add_memlet_path(t2, mx, o2, src_conn='r', memlet=dace.Memlet('out2[e]'))

    sdfg.validate()

    rng = np.random.default_rng(1234)
    e = 24
    w_v = rng.random(e)
    s_v = rng.random(e)
    idx_v = rng.integers(0, e, size=(e, 2), dtype=np.int32)
    idx2_v = rng.integers(0, e, size=e, dtype=np.int32)

    expected1 = 1.5 * w_v[idx_v[:, 0]] - 0.5 * w_v[idx_v[:, 1]]
    expected2 = w_v[idx2_v] + s_v

    before = _count_map_entries(sdfg)
    n = sdfg.apply_transformations_repeated(MapFission)
    assert n > 0
    sdfg.validate()
    after = _count_map_entries(sdfg)
    assert after > before, f"map count did not increase: {before} -> {after}"

    out1 = np.zeros(e)
    out2 = np.zeros(e)
    sdfg(w=w_v, s=s_v, idx=idx_v, idx2=idx2_v, out1=out1, out2=out2, E=e)

    assert np.allclose(out1, expected1)
    assert np.allclose(out2, expected2)


def test_nested_sdfg_multistate_fission():
    """A Map containing a multi-state NestedSDFG fissions (mirrors test_nested_transient).

    The NestedSDFG has two states chained by a transient: ``t = 2 * a`` then
    ``b = 3 * t``.  MapFission must split the outer map around the NestedSDFG.
    """
    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('a', [1], dace.float64)
    nsdfg.add_array('b', [1], dace.float64)
    nsdfg.add_transient('t', [1], dace.float64)

    nstate = nsdfg.add_state()
    irnode = nstate.add_read('a')
    task = nstate.add_tasklet('t1', {'inp'}, {'out'}, 'out = 2 * inp')
    iwnode = nstate.add_write('t')
    nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple('a', '0'))
    nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('t', '0'))

    first_state = nstate
    nstate = nsdfg.add_state()
    irnode = nstate.add_read('t')
    task = nstate.add_tasklet('t2', {'inp'}, {'out'}, 'out = 3 * inp')
    iwnode = nstate.add_write('b')
    nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple('t', '0'))
    nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('b', '0'))

    nsdfg.add_edge(first_state, nstate, dace.InterstateEdge())

    sdfg = dace.SDFG('nested_sdfg_multistate_fission')
    sdfg.add_array('A', [_E], dace.float64)
    state = sdfg.add_state()
    rnode = state.add_read('A')
    wnode = state.add_write('A')
    me, mx = state.add_map('outer', dict(i='0:E'))
    nsdfg_node = state.add_nested_sdfg(nsdfg, {'a'}, {'b'})
    state.add_memlet_path(rnode, me, nsdfg_node, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_memlet_path(nsdfg_node, mx, wnode, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    sdfg.validate()

    rng = np.random.default_rng(4321)
    e = 17
    A = rng.random(e)
    expected = A * 6

    before = _count_map_entries(sdfg)
    n = sdfg.apply_transformations_repeated(MapFission)
    assert n > 0
    sdfg.validate()
    after = _count_map_entries(sdfg)
    assert after > before, f"map count did not increase: {before} -> {after}"

    out = A.copy()
    sdfg(A=out, E=e)
    assert np.allclose(out, expected)


def test_multidim_gather_fission():
    """Multidim Map with a NestedSDFG over a gathered first dimension (mirrors test_multidim).

    The outer map runs over ``(e, k)``; a NestedSDFG with two states computes
    ``out[e, k] = c1 * w[idx[e, 0], k] - c2 * w[idx[e, 1], k]`` via a transient
    chain.  The first dimension of ``w`` is gathered through ``idx`` (passed in
    as the gathered slices), the level dimension ``k`` is structured.
    """
    nsdfg = dace.SDFG('nested_grad')
    nsdfg.add_array('w0', [1], dace.float64)
    nsdfg.add_array('w1', [1], dace.float64)
    nsdfg.add_array('r', [1], dace.float64)
    nsdfg.add_transient('tmp', [1], dace.float64)

    s0 = nsdfg.add_state()
    rd0 = s0.add_read('w0')
    tk0 = s0.add_tasklet('scale0', {'inp'}, {'out'}, 'out = 1.25 * inp')
    wr0 = s0.add_write('tmp')
    s0.add_edge(rd0, None, tk0, 'inp', dace.Memlet.simple('w0', '0'))
    s0.add_edge(tk0, 'out', wr0, None, dace.Memlet.simple('tmp', '0'))

    s1 = nsdfg.add_state()
    rt = s1.add_read('tmp')
    rw1 = s1.add_read('w1')
    tk1 = s1.add_tasklet('combine', {'a', 'b'}, {'out'}, 'out = a - 0.75 * b')
    wr = s1.add_write('r')
    s1.add_edge(rt, None, tk1, 'a', dace.Memlet.simple('tmp', '0'))
    s1.add_edge(rw1, None, tk1, 'b', dace.Memlet.simple('w1', '0'))
    s1.add_edge(tk1, 'out', wr, None, dace.Memlet.simple('r', '0'))

    nsdfg.add_edge(s0, s1, dace.InterstateEdge())

    sdfg = dace.SDFG('multidim_gather_fission')
    sdfg.add_array('w', [_E, _L], dace.float64)
    sdfg.add_array('idx', [_E, 2], dace.int32)
    sdfg.add_array('out', [_E, _L], dace.float64)

    state = sdfg.add_state()
    wnode = state.add_read('w')
    idxnode = state.add_read('idx')
    onode = state.add_write('out')
    me, mx = state.add_map('outer', dict(e='0:E', k='0:L'))

    # Resolve the gathered first dimension via a small index-load NestedSDFG-free
    # tasklet pair feeding dynamic memlets is not needed: read the full ``w``
    # plane and index it inside the body using the index table.
    g = state.add_tasklet('gather', {'wa', 'i0', 'i1'}, {'g0', 'g1'}, 'g0 = wa[i0]; g1 = wa[i1]')
    state.add_memlet_path(wnode, me, g, dst_conn='wa', memlet=dace.Memlet(data='w', subset='0:E, k'))
    state.add_memlet_path(idxnode, me, g, dst_conn='i0', memlet=dace.Memlet('idx[e, 0]'))
    state.add_memlet_path(idxnode, me, g, dst_conn='i1', memlet=dace.Memlet('idx[e, 1]'))

    sdfg.add_transient('gw0', [1], dace.float64)
    sdfg.add_transient('gw1', [1], dace.float64)
    gw0 = state.add_access('gw0')
    gw1 = state.add_access('gw1')
    state.add_edge(g, 'g0', gw0, None, dace.Memlet.simple('gw0', '0'))
    state.add_edge(g, 'g1', gw1, None, dace.Memlet.simple('gw1', '0'))

    nsdfg_node = state.add_nested_sdfg(nsdfg, {'w0', 'w1'}, {'r'})
    state.add_edge(gw0, None, nsdfg_node, 'w0', dace.Memlet.simple('gw0', '0'))
    state.add_edge(gw1, None, nsdfg_node, 'w1', dace.Memlet.simple('gw1', '0'))
    state.add_memlet_path(nsdfg_node, mx, onode, src_conn='r', memlet=dace.Memlet('out[e, k]'))

    sdfg.validate()

    rng = np.random.default_rng(99)
    e, ll = 10, 7
    w_v = rng.random((e, ll))
    idx_v = rng.integers(0, e, size=(e, 2), dtype=np.int32)
    expected = 1.25 * w_v[idx_v[:, 0], :] - 0.75 * w_v[idx_v[:, 1], :]

    before = _count_map_entries(sdfg)
    n = sdfg.apply_transformations_repeated(MapFission)
    assert n > 0
    sdfg.validate()
    after = _count_map_entries(sdfg)
    assert after > before, f"map count did not increase: {before} -> {after}"

    out = np.zeros((e, ll))
    sdfg(w=w_v, idx=idx_v, out=out, E=e, L=ll)
    assert np.allclose(out, expected)


if __name__ == '__main__':
    test_two_independent_gather_components()
    test_nested_sdfg_multistate_fission()
    test_multidim_gather_fission()
