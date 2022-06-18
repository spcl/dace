# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Various tests for dead code elimination passes. """

import pytest
import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination


def test_dse_simple():
    @dace.program
    def dsetester(a: dace.float64[20], b: dace.float64[20], s: dace.uint32):
        if s > 2 * s:  # Always False
            b[:] = s
        else:
            b[:] = a

    sdfg = dsetester.to_sdfg(simplify=True)
    DeadStateElimination().apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    src_nodes = sdfg.node(0).source_nodes()
    assert len(src_nodes) == 1
    assert isinstance(src_nodes[0], dace.nodes.AccessNode) and src_nodes[0].data == 'a'


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
    removed = Pipeline(DeadDataflowElimination()).apply_pass(sdfg, {})
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
    removed = Pipeline(DeadDataflowElimination()).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    # Access node with c should not exist
    assert all(n.data != 'c' for n in sdfg.node(0).data_nodes())


@pytest.mark.parametrize('second_tasklet', (False, True))
def test_dde_access_node_in_scope(second_tasklet):
    # TODO: map->map->tasklet[,tasklet]->mapexit->(accessnode)->mapexit
    pass


def test_dde_scope_reconnect():
    '''
    Corner case:
    map {
        tasklet(callback()) -> tasklet(do nothing)
    }
    expected map to stay connected
    '''
    # TODO
    # Pipeline(DeadDataflowElimination()).apply_pass(sdfg, {})
    pass


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
        if f[0] > 1:
            b[:] = 5
        b[:] = a + 1  # Ends up overwritten
        b[:] = a + 2
        b += 1

    sdfg = dce_tester.to_sdfg(simplify=False)
    Pipeline(DeadDataflowElimination()).apply_pass(sdfg, {})
    DeadStateElimination().apply_pass(sdfg, {})
    sdfg.simplify()

    sdfg.save('bla.sdfg')


if __name__ == '__main__':
    test_dse_simple()
    test_dse_unconditional()
    test_dde_simple()
    test_dde_libnode()
    test_dde_access_node_in_scope(False)
    test_dde_access_node_in_scope(True)
    test_dde_scope_reconnect()
    test_dce()
