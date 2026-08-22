# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import sdfg_to_tree, treenodes as tn
from dace.transformation.helpers import nest_state_subgraph, nest_sdfg_subgraph, nest_sdfg_control_flow
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ControlFlowRegion, LoopRegion, StateSubgraphView
import numpy as np


@dace.program
def nest_subgraph(A: dace.float64[1], B: dace.float64[1]):
    for i in dace.map[0:1]:
        with dace.tasklet:
            a << A[i]
            b >> B[i]
            b = a


def test_nest_oneelementmap():
    A, B = np.random.rand(1), np.random.rand(1)
    sdfg: dace.SDFG = nest_subgraph.to_sdfg()
    state: dace.SDFGState
    # Nest outer region
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            subgraph = state.scope_subgraph(node)
            nest_state_subgraph(sdfg, state, subgraph)
    # Nest inner scope
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            subgraph = state.scope_subgraph(node, include_entry=False, include_exit=False)
            nest_state_subgraph(state.parent, state, subgraph)

    sdfg(A=A, B=B)
    assert np.allclose(A, B)


def test_internal_outarray():
    sdfg = dace.SDFG('internal_outarr')
    sdfg.add_array('A', [20], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map('_', dict(i='0:1'))
    t = state.add_tasklet('doit', {}, {'a'}, 'a = 0')
    w = state.add_write('A')
    state.add_nedge(me, t, dace.Memlet())
    state.add_edge(t, 'a', w, None, dace.Memlet('A[1]'))
    state.add_nedge(w, mx, dace.Memlet())

    subgraph = StateSubgraphView(state, [t, w])
    nest_state_subgraph(sdfg, state, subgraph)

    a = np.random.rand(20)
    sdfg(A=a)
    assert a[1] == 0


@pytest.mark.parametrize('full_data', [False, True])
def test_nest_keeps_ordering_memlets_empty(full_data):
    """ An empty (happens-before) memlet hanging off the same scope connector as a boundary memlet
        must stay empty after nesting, rather than being stamped with the boundary array. """
    sdfg = dace.SDFG(f'ordering_memlet_nesting_{int(full_data)}')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    state = sdfg.add_state()

    a = state.add_read('A')
    b = state.add_write('B')
    me, mx = state.add_map('m', dict(i='0:20'))
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    mx.add_in_connector('IN_B')
    mx.add_out_connector('OUT_B')
    compute = state.add_tasklet('compute', {'_in'}, {'_out'}, '_out = _in + 1')
    side = state.add_tasklet('side', {}, {'_out'}, '_out = 0')
    scal = state.add_access('tmp')

    state.add_edge(a, None, me, 'IN_A', dace.Memlet('A[0:20]'))
    state.add_edge(me, 'OUT_A', compute, '_in', dace.Memlet('A[i]'))
    # Ordering edge: empty memlet, but on the same scope connector as the boundary memlet above
    state.add_edge(me, 'OUT_A', side, None, dace.Memlet())
    state.add_edge(side, '_out', scal, None, dace.Memlet('tmp'))
    state.add_edge(scal, None, compute, None, dace.Memlet())
    state.add_edge(compute, '_out', mx, 'IN_B', dace.Memlet('B[i]'))
    state.add_edge(mx, 'OUT_B', b, None, dace.Memlet('B[0:20]'))
    sdfg.validate()

    nsdfg_node = nest_state_subgraph(sdfg, state, state.scope_subgraph(me), full_data=full_data)
    sdfg.validate()

    nstate = nsdfg_node.sdfg.states()[0]
    inner_me = next(n for n in nstate.nodes() if isinstance(n, dace.nodes.MapEntry))
    inner_side = next(n for n in nstate.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == 'side')

    ordering = [e for e in nstate.out_edges(inner_me) if e.dst is inner_side]
    assert len(ordering) == 1
    ordering = ordering[0]
    assert ordering.data.is_empty()
    assert ordering.data.data is None
    assert ordering.data.subset is None
    assert ordering.dst_conn is None

    # The sibling data memlet on the very same connector must still carry the boundary array
    inner_compute = next(n for n in nstate.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == 'compute')
    data_edge = next(e for e in nstate.out_edges(inner_me) if e.dst is inner_compute and e.dst_conn == '_in')
    assert not data_edge.data.is_empty()
    assert data_edge.data.data == 'A'

    # No empty memlet anywhere in the nested state gained a data name
    assert all(e.data.subset is not None for e in nstate.edges() if e.data.data is not None)

    A = np.random.rand(20)
    B = np.random.rand(20)
    sdfg(A=A, B=B)
    assert np.allclose(B, A + 1)


def test_symbolic_return():

    @dace.program
    def symbolic_return():
        a = 6
        for i in range(10):
            a = 5
        a -= 1
        return i, a

    sdfg = symbolic_return.to_sdfg()

    stree = sdfg_to_tree.as_schedule_tree(sdfg)
    for_scope = None
    for i, child in enumerate(stree.children):
        if isinstance(child, tn.LoopScope):
            for_scope = child
            break
    assert for_scope

    assert i < len(stree.children) - 1
    exit_scope = stree.children[i + 1]
    assert isinstance(exit_scope, (tn.AssignNode, tn.TaskletNode))

    states = for_scope.loop.nodes()

    nest_sdfg_subgraph(sdfg, SubgraphView(for_scope.loop, states))

    result = sdfg()
    val = result[1][0]
    _, ref = symbolic_return.f()
    assert val == ref


def test_nest_cf_simple_for_loop():

    @dace.program
    def simple_for_loop():
        A = np.ndarray((10, ), dtype=np.int32)
        for i in range(10):
            A[i] = i
        return A

    sdfg = simple_for_loop.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert np.array_equal(sdfg(), np.arange(10, dtype=np.int32))


def test_nest_cf_simple_while_loop():

    def force_callback(f):
        return f

    @force_callback
    def update(x):
        return x + 1

    @dace.program
    def simple_while_loop():
        i = 0
        A = np.ndarray((10, ), dtype=np.int32)
        while i < 10:
            A[i] = i
            i = update(A[i])
        return A

    with pytest.warns(match="Automatically creating callback"):
        sdfg = simple_while_loop.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert np.array_equal(sdfg(update=update), np.arange(10, dtype=np.int32))


def test_nest_cf_simple_if():

    @dace.program
    def simple_if(i: dace.int64):
        if i < 5:
            return 0
        else:
            return 1

    sdfg = simple_if.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert sdfg(2)[0] == 0
    assert sdfg(5)[0] == 1


def test_nest_cf_simple_if_elif():

    @dace.program
    def simple_if_elif(i: dace.int64):
        if i < 2:
            return 0
        elif i < 4:
            return 1
        elif i < 6:
            return 2
        elif i < 8:
            return 3
        else:
            return 4

    sdfg = simple_if_elif.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert sdfg(0)[0] == 0
    assert sdfg(2)[0] == 1
    assert sdfg(4)[0] == 2
    assert sdfg(7)[0] == 3
    assert sdfg(15)[0] == 4


def test_nest_cf_simple_if_chain():

    @dace.program
    def simple_if_chain(i: dace.int64):
        if i < 2:
            return 0
        if i < 4:
            return 1
        if i < 6:
            return 2
        if i < 8:
            return 3
        return 4

    sdfg = simple_if_chain.to_sdfg()
    nest_sdfg_control_flow(sdfg)

    assert sdfg(0)[0] == 0
    assert sdfg(2)[0] == 1
    assert sdfg(4)[0] == 2
    assert sdfg(7)[0] == 3
    assert sdfg(15)[0] == 4


def test_region_local_transient_is_not_exported_as_a_connector():
    """A transient only the outlined REGION touches moves INTO the nest, it does not cross it.

    The locality test compares ``sdfg.states()`` -- which yields the states nested inside a
    control-flow region -- against the subgraph at BLOCK level, where that whole region is one
    node. For a region subgraph the two never matched, so every region-local transient came back
    out as an in AND out connector: a value handed across a boundary it never crosses, and one that
    any per-clone duplication of the nest would then have several writers for.
    """
    N = dace.symbol('N')
    sdfg = dace.SDFG('region_local_transient')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    st = loop.add_state('body', is_start_block=True)
    t0 = st.add_tasklet('t0', {'inp'}, {'out'}, 'out = inp * 2.0')
    wtmp = st.add_access('tmp')
    st.add_edge(st.add_read('x'), None, t0, 'inp', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', wtmp, None, dace.Memlet('tmp'))
    t1 = st.add_tasklet('t1', {'v'}, {'out'}, 'out = v + 1.0')
    st.add_edge(wtmp, None, t1, 'v', dace.Memlet('tmp'))
    st.add_edge(t1, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))

    outer = nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop]))
    sdfg.reset_cfg_list()
    node = next(n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG))
    assert 'tmp' not in node.in_connectors
    assert 'tmp' not in node.out_connectors
    # Moved in, not merely disconnected: the nest owns it and the parent no longer declares it.
    assert 'tmp' not in sdfg.arrays
    assert node.sdfg.arrays['tmp'].transient
    sdfg.validate()

    n = 16
    rng = np.random.default_rng(4)
    x = rng.random(n)
    a = np.zeros(n)
    sdfg(a=a, x=x, N=n)
    assert np.allclose(a, x * 2.0 + 1.0)


def test_region_local_transient_read_by_an_outside_edge_stays_a_connector():
    """A promoted scalar is read by NAME on an interstate edge, never through an access node.

    The node walk alone calls it unused, so it would be moved into the nest and the outside read
    left dangling.
    """
    N = dace.symbol('N')
    sdfg = dace.SDFG('region_local_promoted')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_scalar('flag', dace.float64, transient=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    st = loop.add_state('body', is_start_block=True)
    t0 = st.add_tasklet('t0', {'inp'}, {'out'}, 'out = inp')
    st.add_edge(st.add_read('x'), None, t0, 'inp', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', st.add_write('flag'), None, dace.Memlet('flag'))
    t1 = st.add_tasklet('t1', {'inp'}, {'out'}, 'out = inp')
    st.add_edge(st.add_read('x'), None, t1, 'inp', dace.Memlet('x[i]'))
    st.add_edge(t1, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    after = sdfg.add_state('after')
    sdfg.add_edge(loop, after, dace.InterstateEdge(condition='flag > 0.0'))

    nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop]))
    sdfg.reset_cfg_list()
    assert 'flag' in sdfg.arrays
    sdfg.validate()


def test_symbol_needing_an_incoming_value_stays_an_argument():
    """A symbol the subgraph reassigns but READS FIRST takes its value from the caller.

    "Nothing outside the subgraph reads it" is not enough to call a symbol subgraph-owned: here every
    use of ``N`` sits inside the region, yet the first read precedes the assignment, so the value
    comes in from the caller. Treating it as owned drops it from ``sdfg.symbols`` while it is still
    free, and the SDFG can no longer build its own argument list.
    """
    sdfg = dace.SDFG('incoming_symbol')
    sdfg.add_array('A', [4], dace.int32)
    sdfg.add_symbol('N', dace.int32)
    pre = sdfg.add_state('pre', is_start_block=True)
    reg = ControlFlowRegion('reg', sdfg)
    sdfg.add_node(reg)
    sdfg.add_edge(pre, reg, dace.InterstateEdge())
    # Read N, rebind it, read it again -- all inside the region, nothing outside touches it.
    s0 = reg.add_state('s0', is_start_block=True)
    t0 = s0.add_tasklet('use0', {}, {'o'}, 'o = N')
    s0.add_edge(t0, 'o', s0.add_access('A'), None, dace.Memlet('A[0]'))
    s1 = reg.add_state('s1')
    reg.add_edge(s0, s1, dace.InterstateEdge(assignments={'N': '7'}))
    t1 = s1.add_tasklet('use1', {}, {'o'}, 'o = N')
    s1.add_edge(t1, 'o', s1.add_access('A'), None, dace.Memlet('A[1]'))
    sdfg.add_edge(reg, sdfg.add_state('post'), dace.InterstateEdge())

    reference = np.zeros(4, dtype=np.int32)
    dace.SDFG.from_json(sdfg.to_json())(A=reference, N=3)

    nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [reg]))
    sdfg.reset_cfg_list()
    # Still free, so it must still be declared and still be a required argument.
    assert 'N' in sdfg.free_symbols
    assert 'N' in sdfg.symbols
    assert 'N' in sdfg.arglist()
    nested = next(n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.NestedSDFG))
    assert 'N' in nested.symbol_mapping
    sdfg.validate()

    A = np.zeros(4, dtype=np.int32)
    sdfg(A=A, N=3)
    assert np.array_equal(A, reference)


def test_view_on_the_boundary_is_nested_as_an_array():
    """A View feeding the subgraph from outside becomes a plain array inside the nested SDFG.

    A View is a view only next to the data it views: the ``views`` edge stays in the parent, which
    resolves it before the memlet ever reaches the nested SDFG's connector. Carrying the View class
    across the boundary leaves an access node whose single edge runs to a code node, which
    ``get_view_edge`` cannot resolve -- validation rejects it as an ambiguous View edge.
    """
    sdfg = dace.SDFG('view_boundary')
    sdfg.add_array('A', [4, 4], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    sdfg.add_view('A_flat', [16], dace.float64)
    state = sdfg.add_state()

    a = state.add_access('A')
    view = state.add_access('A_flat')
    state.add_edge(a, None, view, 'views', dace.Memlet('A[0:4, 0:4] -> [0:16]'))
    tasklet = state.add_tasklet('copy', {'i'}, {'o'}, 'o = i * 2.0')
    state.add_edge(view, None, tasklet, 'i', dace.Memlet('A_flat[0]'))
    state.add_edge(tasklet, 'o', state.add_access('B'), None, dace.Memlet('B[0]'))

    # Only the tasklet: the view and its ``views`` edge stay outside, so the tasklet's read crosses
    # the boundary and has to be given a descriptor of its own.
    nsdfg = nest_state_subgraph(sdfg, state, StateSubgraphView(state, [tasklet]))
    sdfg.validate()

    inner = [nsdfg.sdfg.arrays[c] for c in nsdfg.in_connectors]
    assert inner, 'the boundary read should have produced an input connector'
    assert not any(isinstance(desc, dace.data.View) for desc in inner)

    A = np.arange(16, dtype=np.float64).reshape(4, 4)
    B = np.zeros(16, dtype=np.float64)
    sdfg(A=A, B=B)
    assert B[0] == 0.0


if __name__ == '__main__':
    test_nest_oneelementmap()
    test_internal_outarray()
    test_nest_keeps_ordering_memlets_empty(False)
    test_nest_keeps_ordering_memlets_empty(True)
    test_symbolic_return()
    test_nest_cf_simple_for_loop()
    test_nest_cf_simple_while_loop()
    test_nest_cf_simple_if()
    test_nest_cf_simple_if_elif()
    test_nest_cf_simple_if_chain()
    test_region_local_transient_is_not_exported_as_a_connector()
    test_region_local_transient_read_by_an_outside_edge_stays_a_connector()
    test_symbol_needing_an_incoming_value_stays_an_argument()
    test_view_on_the_boundary_is_nested_as_an_array()
