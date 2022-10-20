# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
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

    assert len(sdfg.symbols.keys()) == 1

    # Test memlet
    last_state = sdfg.sink_nodes()[0]
    sink = last_state.sink_nodes()[0]
    memlet = last_state.in_edges(sink)[0].data
    assert memlet.data == 'a'
    assert str(memlet.subset).endswith(', 2')


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


def test_allocation_static():
    """
    Allocate an array with a constant-propagated symbolic size.
    """
    sdfg = dace.SDFG('cprop_static_alloc')
    N = dace.symbol('N', dace.int32)
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('tmp', [N], dace.int32, transient=True)
    sdfg.add_array('output', [1], dace.int32)

    a = sdfg.add_state()
    b = sdfg.add_state()
    c = sdfg.add_state_after(b)

    # First state, N=1
    sdfg.add_edge(a, b, dace.InterstateEdge(assignments=dict(N=1)))
    t = b.add_tasklet('somecode', {}, {'out'}, 'out = 2')
    w = b.add_write('tmp')
    b.add_edge(t, 'out', w, None, dace.Memlet('tmp'))

    # Third state outputs value
    c.add_nedge(c.add_read('tmp'), c.add_write('output'), dace.Memlet('tmp[0]'))

    # Do not perform scalar-to-symbol promotion
    ConstantPropagation().apply_pass(sdfg, {})

    assert len(sdfg.symbols) == 0

    val = np.random.rand(1).astype(np.int32)
    sdfg(output=val)
    assert np.allclose(val, 2)


@pytest.mark.parametrize('parametric', [False, True])
def test_allocation_varying(parametric):
    """
    Allocate an array with an initial (symbolic) size, then allocate an array with another size, and ensure
    constants are propagated properly.
    """
    sdfg = dace.SDFG(f'cprop_alloc_{parametric}')
    N = dace.symbol('N', dace.int32)
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('tmp1', [N], dace.int32, transient=True)
    sdfg.add_array('tmp2', [N], dace.int32, transient=True)
    sdfg.add_array('output', [1], dace.int32)

    a = sdfg.add_state()
    b = sdfg.add_state()
    c = sdfg.add_state()

    # First state, N=1
    sdfg.add_edge(a, b, dace.InterstateEdge(assignments=dict(N=1)))
    t = b.add_tasklet('somecode', {}, {'out'}, 'out = 2')
    w = b.add_write('tmp1')
    b.add_edge(t, 'out', w, None, dace.Memlet('tmp1[0]'))

    # Second state, N=tmp1[0] (=2)
    if parametric:
        sdfg.add_edge(b, c, dace.InterstateEdge(assignments=dict(N='tmp1[0]')))
    else:
        sdfg.add_edge(b, c, dace.InterstateEdge(assignments=dict(N=2)))
    t2 = c.add_tasklet('somecode2', {}, {'out'}, 'out = 3')
    t3 = c.add_tasklet('somecode2', {}, {'out'}, 'out = 4')
    w = c.add_write('tmp2')
    c.add_edge(t2, 'out', w, None, dace.Memlet('tmp2[0]'))
    c.add_edge(t3, 'out', w, None, dace.Memlet('tmp2[1]'))

    # Third state outputs value
    c.add_nedge(w, c.add_write('output'), dace.Memlet('tmp2[1]'))

    # Do not perform scalar-to-symbol promotion
    ConstantPropagation().apply_pass(sdfg, {})

    assert len(sdfg.symbols) == 1

    val = np.random.rand(1).astype(np.int32)
    sdfg(output=val)
    assert np.allclose(val, 4)


def test_for_with_external_init():

    N = dace.symbol('N')

    sdfg = dace.SDFG('for_with_external_init')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_array('A', {
        N,
    }, dace.int32)
    init = sdfg.add_state('init')
    body = sdfg.add_state('body')
    sdfg.add_loop(init, body, None, 'i', None, 'i < N', 'i + 1')

    a = body.add_read('A')
    t = body.add_tasklet('tasklet', {}, {'__out'}, '__out = i')
    body.add_edge(t, '__out', a, None, dace.Memlet('A[i]'))
    sdfg.validate()

    init_i = 4
    ref = np.arange(10, dtype=np.int32)
    ref[:init_i] = 0
    val0 = np.zeros((10, ), dtype=np.int32)
    sdfg(A=val0, N=10, i=init_i)
    assert np.allclose(val0, ref)
    ConstantPropagation().apply_pass(sdfg, {})
    val1 = np.zeros((10, ), dtype=np.int32)
    sdfg(A=val1, N=10, i=init_i)
    assert np.allclose(val1, ref)


def test_for_with_external_init_nested():

    N = dace.symbol('N')

    sdfg = dace.SDFG('for_with_external_init_nested')
    sdfg.add_array('A', (N, ), dace.int32)
    init = sdfg.add_state('init', is_start_state=True)
    main = sdfg.add_state('main')
    sdfg.add_edge(init, main, dace.InterstateEdge(assignments={'i': 'N-1'}))

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_array('inner_A', (N,), dace.int32)
    ninit = nsdfg.add_state('nested_init', is_start_state=True)
    nguard = nsdfg.add_state('nested_guard')
    nbody = nsdfg.add_state('nested_body')
    nexit = nsdfg.add_state('nested_exit')
    nsdfg.add_edge(ninit, nguard, dace.InterstateEdge())
    nsdfg.add_edge(nguard, nbody, dace.InterstateEdge(condition='i >= 0'))
    nsdfg.add_edge(nbody, nguard, dace.InterstateEdge(assignments={'i': 'i-1'}))
    nsdfg.add_edge(nguard, nexit, dace.InterstateEdge(condition='i < 0'))

    na = nbody.add_access('inner_A')
    nt = nbody.add_tasklet('tasklet', {}, {'__out'}, '__out = i')
    nbody.add_edge(nt, '__out', na, None, dace.Memlet('inner_A[i]'))

    a = main.add_access('A')
    t = main.add_nested_sdfg(nsdfg, None, {}, {'inner_A'}, {'N': 'N', 'i': 'i'})
    main.add_edge(t, 'inner_A', a, None, dace.Memlet.from_array('A', sdfg.arrays['A']))

    sdfg.validate()

    ref = np.arange(10, dtype=np.int32)
    val0 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val0, N=10)
    assert np.allclose(val0, ref)
    ConstantPropagation().apply_pass(sdfg, {})
    val1 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val1, N=10)
    assert np.allclose(val1, ref)


def test_for_with_external_init_nested_start_with_guard():
    """
    This test differs from the one above in lacking an initialization SDFGState in the NestedSDFG. Instead, the guard
    of the nested for-loop is explicitly set as the start-state of the NestedSDFG.
    """

    N = dace.symbol('N')

    sdfg = dace.SDFG('for_with_external_init_nested_start_with_guard')
    sdfg.add_array('A', (N, ), dace.int32)
    init = sdfg.add_state('init', is_start_state=True)
    main = sdfg.add_state('main')
    sdfg.add_edge(init, main, dace.InterstateEdge(assignments={'i': '1'}))

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_array('inner_A', (N,), dace.int32)
    nguard = nsdfg.add_state('nested_guard', is_start_state=True)
    nbody = nsdfg.add_state('nested_body')
    nexit = nsdfg.add_state('nested_exit')
    nsdfg.add_edge(nguard, nbody, dace.InterstateEdge(condition='i <= N'))
    nsdfg.add_edge(nbody, nguard, dace.InterstateEdge(assignments={'i': 'i+1'}))
    nsdfg.add_edge(nguard, nexit, dace.InterstateEdge(condition='i > N'))

    na = nbody.add_access('inner_A')
    nt = nbody.add_tasklet('tasklet', {}, {'__out'}, '__out = i-1')
    nbody.add_edge(nt, '__out', na, None, dace.Memlet('inner_A[i-1]'))

    a = main.add_access('A')
    t = main.add_nested_sdfg(nsdfg, None, {}, {'inner_A'}, {'N': 'N', 'i': 'i'})
    main.add_edge(t, 'inner_A', a, None, dace.Memlet.from_array('A', sdfg.arrays['A']))

    sdfg.validate()

    ref = np.arange(10, dtype=np.int32)
    val0 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val0, N=10)
    assert np.allclose(val0, ref)
    ConstantPropagation().apply_pass(sdfg, {})
    val1 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val1, N=10)
    assert np.allclose(val1, ref)


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
    test_allocation_static()
    test_allocation_varying(False)
    test_allocation_varying(True)
    test_for_with_external_init()
    test_for_with_external_init_nested()
    test_for_with_external_init_nested_start_with_guard()
