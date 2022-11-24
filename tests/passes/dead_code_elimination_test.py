# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Various tests for dead code elimination passes. """

import pytest
import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination


def test_dse_simple():
    sdfg = dace.SDFG('dsetester')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_symbol('s', dace.uint64)
    init = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    s1.add_mapped_tasklet('doit', dict(i='0:20'), {}, 'out = 1', dict(out=dace.Memlet('a[i]')), external_edges=True)
    s2.add_mapped_tasklet('doit', dict(i='0:20'), {}, 'out = 2', dict(out=dace.Memlet('a[i]')), external_edges=True)

    sdfg.add_edge(init, s1, dace.InterstateEdge('s > s'))  # Always false
    sdfg.add_edge(init, s2, dace.InterstateEdge('s <= s'))

    DeadStateElimination().apply_pass(sdfg, {})
    assert set(sdfg.nodes()) == {init, s2}


def test_dse_unconditional():
    sdfg = dace.SDFG('dse_tester')
    sdfg.add_symbol('a', dace.int32)
    s = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    s3 = sdfg.add_state()
    e = sdfg.add_state()
    sdfg.add_edge(s, s1, dace.InterstateEdge('a > 0'))
    sdfg.add_edge(s, s2, dace.InterstateEdge('a >= a'))  # Always True
    sdfg.add_edge(s, s3, dace.InterstateEdge('a < 0'))
    sdfg.add_edge(s1, e, dace.InterstateEdge())
    sdfg.add_edge(s2, e, dace.InterstateEdge())
    sdfg.add_edge(s3, e, dace.InterstateEdge())

    DeadStateElimination().apply_pass(sdfg, {})
    assert set(sdfg.states()) == {s, s2, e}


def test_dde_simple():

    @dace.program
    def dde_tester(a: dace.float64[20], b: dace.float64[20]):
        c = a + b
        b[:] = a

    sdfg = dde_tester.to_sdfg()
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    # Access node with c should not exist
    assert all(n.data != 'c' for n in sdfg.node(0).data_nodes())


def test_dde_libnode():

    @dace.program
    def dde_tester(a: dace.float64[20], b: dace.float64[20]):
        c = a @ b
        b[:] = a

    sdfg = dde_tester.to_sdfg()
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    # Access node with c should not exist
    assert all(n.data != 'c' for n in sdfg.node(0).data_nodes())


@pytest.mark.parametrize('second_tasklet', (False, True))
def test_dde_access_node_in_scope(second_tasklet):
    sdfg = dace.SDFG('dde_anis')
    sdfg.add_transient('a', [20], dace.float64)
    state = sdfg.add_state()

    # map->map->tasklet[,tasklet]->mapexit->(accessnode)->mapexit
    ome, omx = state.add_map('outer', dict(i='0:10'))
    ime, imx = state.add_map('inner', dict(j='0:2'))
    t = state.add_tasklet('doit', {}, {'out'}, 'out = 1')
    w = state.add_write('a')
    state.add_memlet_path(ome, ime, t, memlet=dace.Memlet())
    state.add_memlet_path(t, imx, w, memlet=dace.Memlet('a[i*2+j]'), src_conn='out')
    state.add_nedge(w, omx, dace.Memlet())

    if second_tasklet:
        t2 = state.add_tasklet('stateful', {}, {}, '', side_effects=True)
        state.add_nedge(ime, t2, dace.Memlet())
        state.add_nedge(t2, imx, dace.Memlet())

    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    if second_tasklet:
        assert set(state.nodes()) == {ome, ime, t2, imx, omx}
    else:
        assert state.number_of_nodes() == 0
    sdfg.validate()


def test_dde_connectors():
    """ Tests removal of connectors on tasklets and nested SDFGs. """

    @dace.program
    def dde_conntest(a: dace.float64[20], b: dace.float64[20]):
        c = dace.ndarray([20], dace.float64)
        for i in dace.map[0:20]:
            with dace.tasklet:
                inp << a[i]

                out1 = inp * 2
                out2 = inp + 3

                out1 >> c[i]
                out2 >> b[i]

    sdfg = dde_conntest.to_sdfg()  # Nested SDFG or tasklet depends on simplify configuration
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert all('c' not in [n.data for n in state.data_nodes()] for state in sdfg.nodes())
    sdfg.validate()


def test_dde_scope_reconnect():
    """
    Corner case:
    map {
        tasklet(callback()) -> tasklet(do nothing)
    }
    expected map to stay connected
    """
    sdfg = dace.SDFG('dde_scope_tester')
    sdfg.add_symbol('cb', dace.callback(dace.float64))
    sdfg.add_scalar('s', dace.float64, transient=True)

    state = sdfg.add_state()
    me, mx = state.add_map('doit', dict(i='0:2'))
    # Tasklet has a callback and cannot be removed
    t1 = state.add_tasklet('callback', {}, {'o'}, 'o = cb()', side_effects=True)
    # Tasklet has no output and thus can be removed
    t2 = state.add_tasklet('nothing', {'inp'}, {}, '')
    state.add_nedge(me, t1, dace.Memlet())
    state.add_edge(t1, 'o', t2, 'inp', dace.Memlet('s'))
    state.add_nedge(t2, mx, dace.Memlet())

    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert set(state.nodes()) == {me, t1, mx}
    sdfg.validate()


@pytest.mark.parametrize('libnode', (False, True))
def test_dde_inout(libnode):
    """ Tests nested SDFG with the same array as input and output. """
    sdfg = dace.SDFG('dde_inout')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_transient('b', [20], dace.float64)
    state = sdfg.add_state()

    if not libnode:

        @dace.program
        def nested(a: dace.float64[20], b: dace.float64[20]):
            for i in range(1, 20):
                b[i] = a[i - 1] + 1
            for i in range(19):
                a[i] = b[i + 1] + 1

        nsdfg = nested.to_sdfg(simplify=False)
        node = state.add_nested_sdfg(nsdfg, None, {'b'}, {'a', 'b'})
        outconn = 'b'
    else:
        node = dace.nodes.LibraryNode('tester')  # Library node without side effects
        node.add_in_connector('b')
        node.add_out_connector('a')
        node.add_out_connector('bout')
        state.add_node(node)
        outconn = 'bout'  # Library node has different output connector name

    state.add_edge(node, 'a', state.add_write('a'), None, dace.Memlet('a'))
    state.add_edge(state.add_read('b'), None, node, 'b', dace.Memlet('b'))
    state.add_edge(node, outconn, state.add_write('b'), None, dace.Memlet('b'))

    assert sorted([n.data for n in state.data_nodes()]) == ['a', 'b', 'b']
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert sorted([n.data for n in state.data_nodes()]) == ['a', 'b', 'b']
    sdfg.validate()


def test_dce():
    """ End-to-end test evaluating both dataflow and state elimination. """
    # Code should end up as b[:] = a + 2; b += 1
    @dace.program
    def dce_tester(a: dace.float64[20], b: dace.float64[20]):
        c = a + 1
        d = 4
        e = 5
        f = c + e
        # Unused branch
        if a[0] > 1:
            b[:] = 5
        b[:] = a + 1  # Ends up overwritten
        b[:] = a + 2
        b += 1

    sdfg = dce_tester.to_sdfg(simplify=False)
    result = Pipeline([DeadDataflowElimination(), DeadStateElimination()]).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() <= 5

    # Check that arrays were removed
    assert all('c' not in [n.data for n in state.data_nodes()] for state in sdfg.nodes())
    assert any('f' in [n.data for n in rstate if isinstance(n, dace.nodes.AccessNode)]
               for rstate in result['DeadDataflowElimination'].values())


def test_dce_callback():

    def dace_inhibitor(f):
        return f

    @dace_inhibitor
    def callback(arr):
        # Do something with the array
        print(arr)

    @dace.program
    def dce_cb(a: dace.float64[20]):
        callback(a)

    sdfg = dce_cb.to_sdfg()
    num_tasklets_before = len([n for n, p in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)])
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    num_tasklets_after = len([n for n, p in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)])
    assert num_tasklets_after > 0 and num_tasklets_after == num_tasklets_before


def test_dce_callback_manual():
    sdfg = dace.SDFG('dce_cbman')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_symbol('cb', dace.callback(None, dace.float64))
    state = sdfg.add_state()

    r = state.add_read('a')
    t = state.add_tasklet('callback', {'inp'}, {}, 'cb(inp)')
    state.add_edge(r, None, t, 'inp', dace.Memlet('a[0:20]'))

    sdfg.validate()
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert set(state.nodes()) == {r, t}
    sdfg.validate()


if __name__ == '__main__':
    test_dse_simple()
    test_dse_unconditional()
    test_dde_simple()
    test_dde_libnode()
    test_dde_access_node_in_scope(False)
    test_dde_access_node_in_scope(True)
    test_dde_connectors()
    test_dde_scope_reconnect()
    test_dde_inout(False)
    test_dde_inout(True)
    test_dce()
    test_dce_callback()
    test_dce_callback_manual()
