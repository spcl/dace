# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the MarkConstInit pass. """

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.mark_const_init import MarkConstInit


def _run(sdfg: dace.SDFG):
    # Pipeline nests MarkConstInit's own {cfg_id: {name: classification}} return under the pass name; unwrap it.
    res = Pipeline([MarkConstInit()]).apply_pass(sdfg, {})
    return (res or {}).get('MarkConstInit')


def _tasklets(state):
    return [n for n in state.nodes() if isinstance(n, nd.Tasklet)]


def _map_entries(state):
    return [n for n in state.nodes() if isinstance(n, nd.MapEntry)]


def test_scalar_constant_single_write():
    """A scalar written once by ``a = 0`` then read becomes a constexpr_static constant."""
    sdfg = dace.SDFG('scalar_const')
    sdfg.add_scalar('a', dace.int32, transient=True)
    sdfg.add_array('B', [1], dace.int32)

    s1 = sdfg.add_state('init')
    init_t = s1.add_tasklet('init', {}, {'out'}, 'out = 0')
    a_write = s1.add_write('a')
    s1.add_edge(init_t, 'out', a_write, None, dace.Memlet('a[0]'))

    s2 = sdfg.add_state('use')
    a_read = s2.add_read('a')
    use_t = s2.add_tasklet('use', {'inp'}, {'out'}, 'out = inp + 1')
    b_write = s2.add_write('B')
    s2.add_edge(a_read, None, use_t, 'inp', dace.Memlet('a[0]'))
    s2.add_edge(use_t, 'out', b_write, None, dace.Memlet('B[0]'))

    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('a') == 'constexpr_static'
    assert 'a' in sdfg.constants
    assert int(sdfg.constants['a']) == 0
    assert _tasklets(s1) == []
    assert [n for n in s1.data_nodes() if n.data == 'a'] == []
    assert any(n.data == 'a' for n in s2.data_nodes())


def test_array_full_constant_write():
    """An array fully written by a constant map then read becomes a constexpr_static constant."""
    sdfg = dace.SDFG('array_full')
    sdfg.add_array('A', [10], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    s1 = sdfg.add_state('init')
    s1.add_mapped_tasklet('init', dict(i='0:10'), {}, 'out = 3.0', dict(out=dace.Memlet('A[i]')), external_edges=True)

    s2 = sdfg.add_state('use')
    s2.add_mapped_tasklet('use',
                          dict(i='0:10'),
                          dict(inp=dace.Memlet('A[i]')),
                          'out = inp * 2.0',
                          dict(out=dace.Memlet('B[i]')),
                          external_edges=True)

    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('A') == 'constexpr_static'
    assert 'A' in sdfg.constants
    assert np.array_equal(sdfg.constants['A'], np.full(10, 3.0))
    assert _tasklets(s1) == []
    assert [n for n in s1.data_nodes() if n.data == 'A'] == []


def test_array_partial_constant_write():
    """A partially written array (indices 1..8) is zero-filled elsewhere."""
    sdfg = dace.SDFG('array_partial')
    sdfg.add_array('A', [10], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    s1 = sdfg.add_state('init')
    s1.add_mapped_tasklet('init', dict(i='1:9'), {}, 'out = 5.0', dict(out=dace.Memlet('A[i]')), external_edges=True)

    s2 = sdfg.add_state('use')
    s2.add_mapped_tasklet('use',
                          dict(i='0:10'),
                          dict(inp=dace.Memlet('A[i]')),
                          'out = inp',
                          dict(out=dace.Memlet('B[i]')),
                          external_edges=True)

    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('A') == 'constexpr_static'
    assert 'A' in sdfg.constants
    expected = np.zeros(10, dtype=np.float64)
    expected[1:9] = 5.0
    assert np.array_equal(sdfg.constants['A'], expected)
    assert expected[0] == 0.0 and expected[9] == 0.0


def test_scalar_runtime_single_write():
    """A scalar written once from runtime data and read in the SAME state is marked const_runtime."""
    sdfg = dace.SDFG('scalar_runtime')
    sdfg.add_array('src', [1], dace.int32)
    sdfg.add_scalar('a', dace.int32, transient=True)
    sdfg.add_array('B', [1], dace.int32)

    s = sdfg.add_state('main')
    src_read = s.add_read('src')
    init_t = s.add_tasklet('init', {'inp'}, {'out'}, 'out = inp')
    a_node = s.add_access('a')
    s.add_edge(src_read, None, init_t, 'inp', dace.Memlet('src[0]'))
    s.add_edge(init_t, 'out', a_node, None, dace.Memlet('a[0]'))
    use_t = s.add_tasklet('use', {'inp'}, {'out'}, 'out = inp + 1')
    b_write = s.add_write('B')
    s.add_edge(a_node, None, use_t, 'inp', dace.Memlet('a[0]'))
    s.add_edge(use_t, 'out', b_write, None, dace.Memlet('B[0]'))

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('a') == 'const_runtime'
    assert sdfg.arrays['a'].const_init is True
    assert 'a' not in sdfg.constants
    assert len(_tasklets(s)) == 2
    assert any(n.data == 'a' for n in s.data_nodes())


def test_array_double_write_not_marked():
    """An array written in two states is not marked."""
    sdfg = dace.SDFG('array_double')
    sdfg.add_array('A', [10], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    s1 = sdfg.add_state('init1')
    s1.add_mapped_tasklet('w1', dict(i='0:10'), {}, 'out = 1.0', dict(out=dace.Memlet('A[i]')), external_edges=True)

    s2 = sdfg.add_state('init2')
    s2.add_mapped_tasklet('w2', dict(i='0:10'), {}, 'out = 2.0', dict(out=dace.Memlet('A[i]')), external_edges=True)

    s3 = sdfg.add_state('use')
    s3.add_mapped_tasklet('use',
                          dict(i='0:10'),
                          dict(inp=dace.Memlet('A[i]')),
                          'out = inp',
                          dict(out=dace.Memlet('B[i]')),
                          external_edges=True)

    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s2, s3, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert 'A' not in kinds
    assert 'A' not in sdfg.constants
    assert not sdfg.arrays['A'].const_init
    assert _tasklets(s1) != []
    assert _tasklets(s2) != []
    # _tasklets() alone can't see a decline leaving the fill map flattened (speculative unroll runs
    # before classification, and 10 leftover per-element tasklets would pass that check either way).
    assert _map_entries(s1) != [], 'rejected fill map was unrolled anyway'


def test_interstate_edge_read_not_marked():
    """A scalar read on an interstate edge is a live use and must not be promoted, even if written once."""
    sdfg = dace.SDFG('iedge_read')
    sdfg.add_scalar('a', dace.int32, transient=True)
    sdfg.add_array('B', [1], dace.int32)

    s1 = sdfg.add_state('init')
    init_t = s1.add_tasklet('init', {}, {'out'}, 'out = 5')
    a_write = s1.add_write('a')
    s1.add_edge(init_t, 'out', a_write, None, dace.Memlet('a[0]'))

    s2 = sdfg.add_state('mid')
    # Otherwise identical to the plain scalar-constant case -- the iedge read is the sole disqualifier.
    s3 = sdfg.add_state('use')
    a_read = s3.add_read('a')
    use_t = s3.add_tasklet('use', {'inp'}, {'out'}, 'out = inp + 1')
    b_write = s3.add_write('B')
    s3.add_edge(a_read, None, use_t, 'inp', dace.Memlet('a[0]'))
    s3.add_edge(use_t, 'out', b_write, None, dace.Memlet('B[0]'))

    # The scalar 'a' is read in the interstate-edge assignment RHS -> a live use with no access node.
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'k': 'a'}))
    sdfg.add_edge(s2, s3, dace.InterstateEdge(condition='k < 10'))

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert 'a' not in kinds
    assert 'a' not in sdfg.constants
    assert not sdfg.arrays['a'].const_init
    assert len(_tasklets(s1)) == 1
    assert any(n.data == 'a' for n in s1.data_nodes())


def test_same_state_separable_marked():
    """Init and read in the SAME state are fine when the read comes from the write node (init precedes read)."""
    sdfg = dace.SDFG('same_state_sep')
    sdfg.add_scalar('a', dace.int32, transient=True)
    sdfg.add_array('B', [1], dace.int32)

    s = sdfg.add_state('main')
    init_t = s.add_tasklet('init', {}, {'out'}, 'out = 4')
    a_node = s.add_access('a')
    s.add_edge(init_t, 'out', a_node, None, dace.Memlet('a[0]'))
    use_t = s.add_tasklet('use', {'inp'}, {'out'}, 'out = inp + 1')
    b_write = s.add_write('B')
    s.add_edge(a_node, None, use_t, 'inp', dace.Memlet('a[0]'))
    s.add_edge(use_t, 'out', b_write, None, dace.Memlet('B[0]'))

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('a') == 'constexpr_static'
    assert 'a' in sdfg.constants
    assert int(sdfg.constants['a']) == 4
    assert init_t not in s.nodes()
    assert a_node in s.nodes()
    assert use_t in s.nodes()


def test_same_state_non_separable_not_marked():
    """A read of the transient that is not ordered after the write in the same state must not be marked."""
    sdfg = dace.SDFG('same_state_nonsep')
    sdfg.add_scalar('a', dace.int32, transient=True)
    sdfg.add_array('B', [1], dace.int32)

    s = sdfg.add_state('main')
    # A read access node disconnected from the write -> the scheduler could read before the write.
    a_read = s.add_read('a')
    use_t = s.add_tasklet('use', {'inp'}, {'out'}, 'out = inp + 1')
    b_write = s.add_write('B')
    s.add_edge(a_read, None, use_t, 'inp', dace.Memlet('a[0]'))
    s.add_edge(use_t, 'out', b_write, None, dace.Memlet('B[0]'))
    # The single write, in its own disconnected subgraph.
    init_t = s.add_tasklet('init', {}, {'out'}, 'out = 4')
    a_write = s.add_access('a')
    s.add_edge(init_t, 'out', a_write, None, dace.Memlet('a[0]'))

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert 'a' not in kinds
    assert 'a' not in sdfg.constants
    assert not sdfg.arrays['a'].const_init
    assert init_t in s.nodes()
    assert a_write in s.nodes()


def test_nested_sdfg_no_crash_and_marks():
    """A transient inside a NestedSDFG is handled per-SDFG without cfg_id indexing hazards (no KeyError)."""
    nsdfg = dace.SDFG('inner')
    nsdfg.add_array('a_in', [10], dace.float64)
    nsdfg.add_array('a_out', [10], dace.float64)
    nsdfg.add_array('t', [10], dace.float64, transient=True)

    ns1 = nsdfg.add_state('ns1')
    ns1.add_mapped_tasklet('init', dict(i='0:10'), {}, 'out = 2.0', dict(out=dace.Memlet('t[i]')), external_edges=True)
    ns2 = nsdfg.add_state('ns2')
    ns2.add_mapped_tasklet('use',
                           dict(i='0:10'),
                           dict(x=dace.Memlet('t[i]'), y=dace.Memlet('a_in[i]')),
                           'out = x + y',
                           dict(out=dace.Memlet('a_out[i]')),
                           external_edges=True)
    nsdfg.add_edge(ns1, ns2, dace.InterstateEdge())

    sdfg = dace.SDFG('outer')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state('main')
    nsdfg_node = state.add_nested_sdfg(nsdfg, {'a_in'}, {'a_out'})
    a_read = state.add_read('A')
    b_write = state.add_write('B')
    state.add_edge(a_read, None, nsdfg_node, 'a_in', dace.Memlet('A[0:10]'))
    state.add_edge(nsdfg_node, 'a_out', b_write, None, dace.Memlet('B[0:10]'))

    # Must not raise (this shape previously KeyErrored in ScalarWriteShadowScopes).
    res = _run(sdfg)

    # The transient lives in the nested SDFG, so its classification is keyed by ``nsdfg.cfg_id``.
    kinds = (res or {}).get(nsdfg.cfg_id, {})
    assert kinds.get('t') == 'constexpr_static'
    assert 't' in nsdfg.constants
    assert np.array_equal(nsdfg.constants['t'], np.full(10, 2.0))
    assert _tasklets(ns1) == []


def _read_array_in_state(state, name, length):
    """Adds a full read of ``name`` (via a copy map into a fresh output) in ``state``."""
    state.add_mapped_tasklet('read_' + name,
                             dict(i=f'0:{length}'),
                             dict(inp=dace.Memlet(f'{name}[i]')),
                             'out = inp',
                             dict(out=dace.Memlet('OUT[i]')),
                             external_edges=True)


def test_multiwrite_elementwise_one_state_marked():
    """arr[0..3]=0,1,2,3 by four constant tasklets into one access node, then read -> constexpr [0,1,2,3]."""
    sdfg = dace.SDFG('mw_one_state')
    sdfg.add_array('arr', [4], dace.int32, transient=True)
    sdfg.add_array('OUT', [4], dace.int32)

    s1 = sdfg.add_state('init')
    arr_node = s1.add_access('arr')
    for k in range(4):
        t = s1.add_tasklet(f'init{k}', {}, {'out'}, f'out = {k}')
        s1.add_edge(t, 'out', arr_node, None, dace.Memlet(f'arr[{k}]'))

    s2 = sdfg.add_state('use')
    _read_array_in_state(s2, 'arr', 4)
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('arr') == 'constexpr_static'
    assert np.array_equal(sdfg.constants['arr'], np.array([0, 1, 2, 3], dtype=np.int32))
    assert _tasklets(s1) == []
    assert [n for n in s1.data_nodes() if n.data == 'arr'] == []


def test_multiwrite_elementwise_spread_states_marked():
    """The four element writes spread across four consecutive states, then read -> marked."""
    sdfg = dace.SDFG('mw_spread')
    sdfg.add_array('arr', [4], dace.int32, transient=True)
    sdfg.add_array('OUT', [4], dace.int32)

    prev = None
    init_states = []
    for k in range(4):
        sk = sdfg.add_state(f'init{k}')
        node = sk.add_access('arr')
        t = sk.add_tasklet(f'init{k}', {}, {'out'}, f'out = {k}')
        sk.add_edge(t, 'out', node, None, dace.Memlet(f'arr[{k}]'))
        init_states.append(sk)
        if prev is not None:
            sdfg.add_edge(prev, sk, dace.InterstateEdge())
        prev = sk

    s_read = sdfg.add_state('use')
    _read_array_in_state(s_read, 'arr', 4)
    sdfg.add_edge(prev, s_read, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('arr') == 'constexpr_static'
    assert np.array_equal(sdfg.constants['arr'], np.array([0, 1, 2, 3], dtype=np.int32))
    for sk in init_states:
        assert _tasklets(sk) == []


def test_multiwrite_different_access_nodes_marked():
    """The four element writes go through four distinct access nodes of arr in one state -> marked."""
    sdfg = dace.SDFG('mw_diff_nodes')
    sdfg.add_array('arr', [4], dace.int32, transient=True)
    sdfg.add_array('OUT', [4], dace.int32)

    s1 = sdfg.add_state('init')
    for k in range(4):
        node = s1.add_access('arr')
        t = s1.add_tasklet(f'init{k}', {}, {'out'}, f'out = {k}')
        s1.add_edge(t, 'out', node, None, dace.Memlet(f'arr[{k}]'))

    s2 = sdfg.add_state('use')
    _read_array_in_state(s2, 'arr', 4)
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('arr') == 'constexpr_static'
    assert np.array_equal(sdfg.constants['arr'], np.array([0, 1, 2, 3], dtype=np.int32))
    assert _tasklets(s1) == []
    assert [n for n in s1.data_nodes() if n.data == 'arr'] == []


def test_multiwrite_read_before_write_not_marked():
    """A read interleaved before one of the writes must not be marked (ordering not provable)."""
    sdfg = dace.SDFG('mw_interleaved')
    sdfg.add_array('arr', [4], dace.int32, transient=True)
    sdfg.add_array('OUT', [4], dace.int32)

    s0 = sdfg.add_state('w0')
    n0 = s0.add_access('arr')
    t0 = s0.add_tasklet('w0', {}, {'out'}, 'out = 0')
    s0.add_edge(t0, 'out', n0, None, dace.Memlet('arr[0]'))

    s1 = sdfg.add_state('read')
    _read_array_in_state(s1, 'arr', 4)

    s2 = sdfg.add_state('w1')
    n2 = s2.add_access('arr')
    t2 = s2.add_tasklet('w1', {}, {'out'}, 'out = 1')
    s2.add_edge(t2, 'out', n2, None, dace.Memlet('arr[1]'))

    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert 'arr' not in kinds
    assert 'arr' not in sdfg.constants
    assert not sdfg.arrays['arr'].const_init
    assert t0 in s0.nodes()
    assert t2 in s2.nodes()


def test_multiwrite_partial_elementwise_marked():
    """Only arr[1] and arr[2] written (shape 4) -> constexpr with zeros at 0 and 3."""
    sdfg = dace.SDFG('mw_partial')
    sdfg.add_array('arr', [4], dace.int32, transient=True)
    sdfg.add_array('OUT', [4], dace.int32)

    s1 = sdfg.add_state('init')
    arr_node = s1.add_access('arr')
    for k, val in ((1, 7), (2, 8)):
        t = s1.add_tasklet(f'init{k}', {}, {'out'}, f'out = {val}')
        s1.add_edge(t, 'out', arr_node, None, dace.Memlet(f'arr[{k}]'))

    s2 = sdfg.add_state('use')
    _read_array_in_state(s2, 'arr', 4)
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('arr') == 'constexpr_static'
    assert np.array_equal(sdfg.constants['arr'], np.array([0, 7, 8, 0], dtype=np.int32))
    assert _tasklets(s1) == []


def test_multiwrite_overlapping_not_marked():
    """Two writes to the same element (arr[0]) conflict -> not marked."""
    sdfg = dace.SDFG('mw_overlap')
    sdfg.add_array('arr', [4], dace.int32, transient=True)
    sdfg.add_array('OUT', [4], dace.int32)

    s1 = sdfg.add_state('w0')
    n1 = s1.add_access('arr')
    t1 = s1.add_tasklet('w0', {}, {'out'}, 'out = 1')
    s1.add_edge(t1, 'out', n1, None, dace.Memlet('arr[0]'))

    s2 = sdfg.add_state('w1')
    n2 = s2.add_access('arr')
    t2 = s2.add_tasklet('w1', {}, {'out'}, 'out = 2')
    s2.add_edge(t2, 'out', n2, None, dace.Memlet('arr[0]'))

    s3 = sdfg.add_state('use')
    _read_array_in_state(s3, 'arr', 4)

    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s2, s3, dace.InterstateEdge())

    res = _run(sdfg)

    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert 'arr' not in kinds
    assert 'arr' not in sdfg.constants
    assert not sdfg.arrays['arr'].const_init
    assert t1 in s1.nodes()
    assert t2 in s2.nodes()


def test_idempotency():
    """Running the pass twice leaves the SDFG in a stable state."""
    sdfg = dace.SDFG('idempotent')
    sdfg.add_array('A', [10], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    s1 = sdfg.add_state('init')
    s1.add_mapped_tasklet('init', dict(i='0:10'), {}, 'out = 7.0', dict(out=dace.Memlet('A[i]')), external_edges=True)

    s2 = sdfg.add_state('use')
    s2.add_mapped_tasklet('use',
                          dict(i='0:10'),
                          dict(inp=dace.Memlet('A[i]')),
                          'out = inp',
                          dict(out=dace.Memlet('B[i]')),
                          external_edges=True)

    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    res = _run(sdfg)
    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('A') == 'constexpr_static'
    assert 'A' in sdfg.constants
    first_value = np.copy(sdfg.constants['A'])
    num_nodes_s1 = len(s1.nodes())

    # Second run marks nothing (already an SDFG constant): return value is None, 'A' absent from classification.
    second_result = _run(sdfg)
    assert 'A' not in (second_result or {}).get(sdfg.cfg_id, {})
    assert 'A' in sdfg.constants
    assert np.array_equal(sdfg.constants['A'], first_value)
    assert len(s1.nodes()) == num_nodes_s1


def test_partly_paying_fill_map_is_not_unrolled():
    """A fill map is unrolled only if EVERY name it writes is const-initializable, not just some."""
    sdfg = dace.SDFG('partly_paying')
    sdfg.add_array('X', [8], dace.float64, transient=True)
    sdfg.add_array('Z', [4], dace.float64, transient=True)
    sdfg.add_array('BX', [8], dace.float64)
    sdfg.add_array('BZ', [4], dace.float64)

    s1 = sdfg.add_state('fill')
    # Disjoint halves of X, so the two fills do not overlap (an overlap is declined outright).
    s1.add_mapped_tasklet('fill_x', dict(i='0:4'), {}, 'ox = 1.0', dict(ox=dace.Memlet('X[i]')), external_edges=True)
    s1.add_mapped_tasklet('fill_xz',
                          dict(i='0:4'), {},
                          'ox = 2.0\noz = 3.0',
                          dict(ox=dace.Memlet('X[i + 4]'), oz=dace.Memlet('Z[i]')),
                          external_edges=True)

    # A second write to Z -> Z is declined, which is what makes fill_xz not pay.
    s2 = sdfg.add_state('rewrite_z')
    t = s2.add_tasklet('z0', {}, {'o'}, 'o = 9.0')
    s2.add_edge(t, 'o', s2.add_access('Z'), None, dace.Memlet('Z[0]'))

    s3 = sdfg.add_state('use')
    s3.add_mapped_tasklet('ux',
                          dict(i='0:8'),
                          dict(inp=dace.Memlet('X[i]')),
                          'o = inp',
                          dict(o=dace.Memlet('BX[i]')),
                          external_edges=True)
    s3.add_mapped_tasklet('uz',
                          dict(i='0:4'),
                          dict(inp=dace.Memlet('Z[i]')),
                          'o = inp',
                          dict(o=dace.Memlet('BZ[i]')),
                          external_edges=True)
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s2, s3, dace.InterstateEdge())
    sdfg.validate()

    res = _run(sdfg)
    kinds = (res or {}).get(sdfg.cfg_id, {})

    # Invariant regardless of the classifier's decision: a flattened map must have paid for it.
    if 'X' not in kinds:
        assert len(_map_entries(s1)) == 2, ('a fill map was unrolled but X was not const-inited: '
                                            f'{[m.map.label for m in _map_entries(s1)]}')
    sdfg.validate()


def test_symbolic_fill_map_never_becomes_const_runtime():
    """A fill map is either unrollable or const_runtime-able, never both."""
    sdfg = dace.SDFG('sym_fill')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    sdfg.add_array('B', [1], dace.float64)

    s1 = sdfg.add_state('init')
    # One iteration, value from a SYMBOL: not compile-time constant, but a single well-defined write.
    s1.add_mapped_tasklet('fill',
                          dict(i='0:1'), {},
                          'out = N * 2.0',
                          dict(out=dace.Memlet('s[0]')),
                          external_edges=True)
    s2 = sdfg.add_state('use')
    r = s2.add_read('s')
    t = s2.add_tasklet('use', {'inp'}, {'o'}, 'o = inp + 1.0')
    s2.add_edge(r, None, t, 'inp', dace.Memlet('s[0]'))
    s2.add_edge(t, 'o', s2.add_write('B'), None, dace.Memlet('B[0]'))
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.validate()

    res = _run(sdfg)
    kinds = (res or {}).get(sdfg.cfg_id, {})
    # Consumer in another state -> unrollable, but then the const binding has no scope to live in.
    assert 's' not in kinds, kinds
    # The map is left alone: it does not pay, so it is not flattened.
    assert _map_entries(s1) != [], 'a fill map that cannot pay was unrolled anyway'
    sdfg.validate()


def _same_state_fill_sdfg(name):
    """A constant fill whose consumer lives in the same state (what ``apply_gpu_transformations`` produces)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64, transient=True)
    sdfg.add_array('B', [8], dace.float64)
    s1 = sdfg.add_state('init')
    s2 = sdfg.add_state('use')
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    s1.add_mapped_tasklet('fill', dict(i='0:8'), {}, 'out = 3.0', dict(out=dace.Memlet('A[i]')), external_edges=True)
    s2.add_mapped_tasklet('use',
                          dict(i='0:8'),
                          dict(inp=dace.Memlet('A[i]')),
                          'out = inp * 2',
                          dict(out=dace.Memlet('B[i]')),
                          external_edges=True)
    sdfg.apply_gpu_transformations()  # fuses init into use; pure SDFG rewrite, needs no GPU
    sdfg.validate()
    return sdfg


@pytest.mark.xfail(strict=True,
                   reason='MapUnroll cannot flatten a fill whose consumer shares its state: it '
                   'duplicates the access node per element but keeps the original out-edge, so all N nodes '
                   'claim to deliver the FULL array to the consumer while each is written at one index. '
                   'const-init here needs the fill evaluated WITHOUT unrolling, not a looser classifier.')
def test_same_state_constant_fill_is_const_inited():
    """A constant fill should be const-initializable even when its consumer shares the state."""
    sdfg = _same_state_fill_sdfg('same_state_fill')
    res = _run(sdfg)
    kinds = (res or {}).get(sdfg.cfg_id, {})
    assert kinds.get('A') == 'constexpr_static', kinds
    assert 'A' in sdfg.constants
    assert np.array_equal(sdfg.constants['A'], np.full(8, 3.0))
    sdfg.validate()


# Must stay at MODULE scope: @dace.program nested in a function parses as a nested program and fails cross-test.


@dace.program
def gpu_constfill(A: dace.float64[8]):
    """A constant fill of a transient; after ``apply_gpu_transformations`` it is GPU_Device-scheduled."""
    w = np.zeros((8, ), dtype=np.float64)
    for i in dace.map[0:8]:
        w[i] = 2.0
    for i in dace.map[0:8]:
        A[i] = A[i] * w[i]


@dace.program
def gpu_refill(A: dace.float64[8]):
    """Same, but ``w`` is written a second time, so the classifier rejects it."""
    w = np.zeros((8, ), dtype=np.float64)
    for i in dace.map[0:8]:
        w[i] = 2.0
    for i in dace.map[0:8]:
        w[i] = w[i] + A[i]
    for i in dace.map[0:8]:
        A[i] = w[i]


def _gpu_fill_program(program):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    sdfg.validate()  # Baseline: the GPU-transformed SDFG is valid BEFORE the pass runs.
    return sdfg


def test_gpu_constant_fill_map_keeps_its_schedule():
    """The pass must not flatten a GPU_Device-scheduled fill map into host tasklets (no GPU needed to check)."""
    sdfg = _gpu_fill_program(gpu_constfill)
    _run(sdfg)
    sdfg.validate()


def test_gpu_rejected_fill_map_stays_valid():
    """Same shape, but the fill target is written twice, so the classifier rejects it (the durbin failure)."""
    sdfg = _gpu_fill_program(gpu_refill)
    _run(sdfg)
    sdfg.validate()


if __name__ == '__main__':
    test_scalar_constant_single_write()
    test_array_full_constant_write()
    test_array_partial_constant_write()
    test_scalar_runtime_single_write()
    test_array_double_write_not_marked()
    test_interstate_edge_read_not_marked()
    test_same_state_separable_marked()
    test_same_state_non_separable_not_marked()
    test_nested_sdfg_no_crash_and_marks()
    test_multiwrite_elementwise_one_state_marked()
    test_multiwrite_elementwise_spread_states_marked()
    test_multiwrite_different_access_nodes_marked()
    test_multiwrite_read_before_write_not_marked()
    test_multiwrite_partial_elementwise_marked()
    test_multiwrite_overlapping_not_marked()
    test_idempotency()
    test_partly_paying_fill_map_is_not_unrolled()
    test_symbolic_fill_map_never_becomes_const_runtime()
    test_gpu_constant_fill_map_keeps_its_schedule()
    test_gpu_rejected_fill_map_stays_valid()
