# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.passes.constant_propagation import ConstantPropagation, _UnknownValue
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
import numpy as np


def test_simple_constants():
    @dace.program
    def program(A: dace.float64[20]):
        val = 5
        cval = val % 4
        if cval + 1 == 1:
            A[:] = cval
        else:
            A[:] = cval + 4

    sdfg = program.to_sdfg()
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ConstantPropagation().apply_pass(sdfg, {})

    assert len(sdfg.symbols) == 0
    for e in sdfg.edges():
        assert len(e.data.assignments) == 0
    a = np.random.rand(20)
    sdfg(a)
    assert np.allclose(a, 5)


def test_nested_constants():
    @dace.program
    def program(A: dace.int64[20]):
        i = A[0]
        j = i + 1
        k = j + 1  # Should become i + 2
        l = i + k  # 2*i + 2
        A[l] = k

    sdfg = program.to_sdfg()
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ConstantPropagation().apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i'}

    # Test memlet
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 2
    last_state = sdfg.sink_nodes()[0]
    sink = last_state.sink_nodes()[0]
    memlet = last_state.in_edges(sink)[0].data
    assert memlet.data == 'A'
    assert str(memlet.subset) == '2*i + 2'


def test_simple_loop():
    @dace.program
    def program(a: dace.float64[20]):
        for i in range(3):
            a[i] = 5
        i = 5
        a[0] = i  # Use i - should be const

    sdfg = program.to_sdfg()
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ConstantPropagation().apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i'}
    # Test tasklets
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.Tasklet):
            code = node.code.as_string
            assert '5' in code and 'i' not in code


def test_cprop_inside_loop():
    @dace.program
    def program(a: dace.float64[20]):
        for i in range(5):
            a[i] = i  # Use i - not const
            i = 8
            a[i] = i  # Use i - const
        a[i] = i  # Use i - not const

    sdfg = program.to_sdfg()
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ConstantPropagation().apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i'}

    # Test tasklets
    i_found = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.Tasklet):
            if 'i' in node.code.as_string:
                i_found += 1
    assert i_found == 2


def test_cprop_outside_loop():
    @dace.program
    def program(a: dace.float64[20, 20]):
        k = 5
        j = 2
        for i in range(2):
            j = 5
        # Use j - not const
        # Use k - const
        a[j, k] = 1

    sdfg = program.to_sdfg()
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ConstantPropagation().apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i', 'j'}

    # Test memlet
    last_state = sdfg.sink_nodes()[0]
    sink = last_state.sink_nodes()[0]
    memlet = last_state.in_edges(sink)[0].data
    assert memlet.data == 'a'
    assert str(memlet.subset) == 'j, 5'


def test_cond():
    @dace.program
    def program(a: dace.float64[20, 20], scal: dace.int32):
        if scal > 0:
            i = 0
            j = 2
        else:
            i = 1
            j = 2

        # After condition, i is not constant and j is
        a[i, j] = 3

    sdfg = program.to_sdfg()
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ConstantPropagation().apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i'}

    # Test memlet
    last_state = sdfg.sink_nodes()[0]
    sink = last_state.sink_nodes()[0]
    memlet = last_state.in_edges(sink)[0].data
    assert memlet.data == 'a'
    assert str(memlet.subset) == 'i, 2'


def test_complex_case():
    """ Tests a complex control flow case. """
    sdfg = dace.SDFG('program')
    sdfg.add_scalar('a', dace.float64)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    branch2 = sdfg.add_state('branch2')
    branch2_1 = sdfg.add_state('branch2_1')
    afterloop = sdfg.add_state('afterloop')  # uses i, should not be constant
    inside_loop1 = sdfg.add_state('inside_loop1')
    inside_loop2 = sdfg.add_state('inside_loop2')
    merge = sdfg.add_state('merge')
    usei = sdfg.add_state('usei')  # uses i, should be constant
    loop2 = sdfg.add_state('loop2')
    last = sdfg.add_state('last')
    sdfg.add_edge(init, guard, dace.InterstateEdge('a > 0', {'i': 5}))
    sdfg.add_edge(init, branch2, dace.InterstateEdge('a <= 0', {'i': 7}))
    sdfg.add_edge(branch2, branch2_1, dace.InterstateEdge())
    sdfg.add_edge(guard, inside_loop1, dace.InterstateEdge('i < 6'))
    sdfg.add_edge(guard, afterloop, dace.InterstateEdge('i >= 6'))
    sdfg.add_edge(inside_loop1, inside_loop2, dace.InterstateEdge(assignments={'i': 6}))
    sdfg.add_edge(inside_loop2, guard, dace.InterstateEdge(assignments={'i': 'i+1'}))

    sdfg.add_edge(afterloop, merge, dace.InterstateEdge(assignments={'i': 7, 'j': 1}))
    sdfg.add_edge(branch2_1, merge, dace.InterstateEdge(assignments={'j': 1}))

    sdfg.add_edge(merge, loop2, dace.InterstateEdge('j < 2'))
    sdfg.add_edge(loop2, usei, dace.InterstateEdge())
    sdfg.add_edge(usei, merge, dace.InterstateEdge(assignments={'j': 'j+1'}))
    sdfg.add_edge(merge, last, dace.InterstateEdge('j >= 2'))

    propagated = ConstantPropagation().collect_constants(sdfg)  #, reachability
    assert len(propagated[init]) == 0
    assert propagated[branch2]['i'] == '7'
    assert propagated[guard]['i'] is _UnknownValue
    assert propagated[inside_loop1]['i'] is _UnknownValue
    assert propagated[inside_loop2]['i'] == '6'
    assert propagated[usei]['i'] == '7'
    assert propagated[afterloop]['i'] is _UnknownValue
    assert propagated[merge]['i'] == '7'
    assert propagated[last]['i'] == '7'
    for pstate in propagated.values():
        if 'j' in pstate:
            assert pstate['j'] is _UnknownValue


def test_early_exit():
    a = np.random.rand(20)

    @dace.program
    def should_not_apply(a):
        return a + 1

    sdfg_no = should_not_apply.to_sdfg(a, simplify=False)
    ScalarToSymbolPromotion().apply_pass(sdfg_no, {})
    assert ConstantPropagation().should_apply(sdfg_no) is False

    @dace.program
    def should_apply(a):
        i = 1
        return a + i

    sdfg_yes = should_apply.to_sdfg(a, simplify=False)
    ScalarToSymbolPromotion().apply_pass(sdfg_yes, {})
    assert ConstantPropagation().should_apply(sdfg_yes) is True


def test_recursive_cprop():
    sdfg = dace.SDFG('program')
    a = sdfg.add_state()
    b = sdfg.add_state()
    sdfg.add_edge(a, b, dace.InterstateEdge(assignments=dict(i=1)))

    nsdfg = dace.SDFG('nested')
    b.add_nested_sdfg(nsdfg, None, {}, {}, symbol_mapping={'i': 'i + 1'})

    nstate = nsdfg.add_state()
    t = nstate.add_tasklet('doprint', {}, {}, 'printf("%d\\n", i)')

    ConstantPropagation().apply_pass(sdfg, {})

    assert len(sdfg.symbols) == 0
    assert len(nsdfg.symbols) == 0
    assert '2' in t.code.as_string


if __name__ == '__main__':
    test_simple_constants()
    test_nested_constants()
    test_simple_loop()
    test_cprop_inside_loop()
    test_cprop_outside_loop()
    test_cond()
    test_complex_case()
    test_early_exit()
    test_recursive_cprop()
