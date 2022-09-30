# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.codegen import control_flow as cf
from dace.transformation.helpers import nest_state_subgraph, nest_sdfg_subgraph, nest_sdfg_control_flow
from dace.sdfg import utils
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import StateSubgraphView
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


def test_symbolic_return():

    @dace.program
    def symbolic_return():
        a = 6
        for i in range(10):
            a = 5
        a -= 1
        return i, a

    sdfg = symbolic_return.to_sdfg()

    cft = cf.structured_control_flow_tree(sdfg, None)
    for_scope = None
    for i, child in enumerate(cft.children):
        if isinstance(child, (cf.ForScope, cf.WhileScope)):
            for_scope = child
            break
    assert for_scope

    assert i < len(cft.children) - 1
    exit_scope = cft.children[i+1]
    assert isinstance(exit_scope, cf.SingleState)

    guard = for_scope.guard
    fexit = exit_scope.first_state
    states = list(utils.dfs_conditional(sdfg, [guard], lambda p, _: p is not fexit))
    
    nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, states), start=guard)

    result = sdfg()
    val = result[1][0]
    _, ref = symbolic_return.f()
    assert val == ref


def test_nest_cf_simple_for_loop():

    @dace.program
    def simple_for_loop():
        A = np.ndarray((10,), dtype=np.int32)
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
        A = np.ndarray((10,), dtype=np.int32)
        while i < 10:
            A[i] = i
            i = update(A[i])
        return A
    
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


if __name__ == '__main__':
    test_nest_oneelementmap()
    test_internal_outarray()
    test_symbolic_return()
    test_nest_cf_simple_for_loop()
    test_nest_cf_simple_while_loop()
    test_nest_cf_simple_if()
    test_nest_cf_simple_if_elif()
    test_nest_cf_simple_if_chain()
