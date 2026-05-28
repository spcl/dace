# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math

N, M, K, L, unused = (dace.symbol(s) for s in 'NMKLU')


@dace.program
def fsymtest(A: dace.float32[20, N]):
    for i, j in dace.map[0:20, 0:N]:
        with dace.tasklet:
            a << A[i, i + M]
            b >> A[i, j]
            b = a + math.exp(K)


@dace.program
def fsymtest_multistate(A: dace.float32[20, N]):
    for k in range(K):
        for i, j in dace.map[0:20, 0:N]:
            with dace.tasklet:
                a << A[i, k + M]
                b >> A[i, j]
                b = L + a


def test_single_state():
    sdfg: dace.SDFG = fsymtest.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    state = sdfg.node(0)
    assert state.free_symbols == {'M', 'N', 'K'}


def test_state_subgraph():
    sdfg = dace.SDFG('fsymtest2')
    state = sdfg.add_state()

    # Add a nested SDFG
    nsdfg = dace.SDFG('nsdfg')
    nstate = nsdfg.add_state()
    me, mx = state.add_map('map', dict(i='0:N'))
    nsdfg = state.add_nested_sdfg(nsdfg, {}, {}, symbol_mapping=dict(l=L / 2, i='i'))
    state.add_nedge(me, nsdfg, dace.Memlet())
    state.add_nedge(nsdfg, mx, dace.Memlet())

    # Entire graph
    assert state.free_symbols == {'L', 'N'}

    # Try a subgraph containing only the map contents
    assert state.scope_subgraph(me, include_entry=False, include_exit=False).free_symbols == {'L', 'i'}


def test_sdfg():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.simplify()
    # Test each state separately
    for state in sdfg.states():
        assert (state.free_symbols == {'k', 'N', 'M', 'L'} or state.free_symbols == set())
    # The SDFG itself should have another free symbol
    assert sdfg.free_symbols == {'K', 'M', 'N', 'L'}


def test_constants():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.simplify()
    sdfg.add_constant('K', 5)
    sdfg.add_constant('L', 20)

    for state in sdfg.states():
        assert (state.free_symbols == {'k', 'N', 'M'} or state.free_symbols == set())
    assert sdfg.free_symbols == {'M', 'N'}


def test_interstate_edge_symbols():
    i, j, k = (dace.symbol(s) for s in 'ijk')

    edge = dace.InterstateEdge(assignments={'i': 'j + k'})
    assert 'j' in edge.free_symbols
    assert 'k' in edge.free_symbols
    assert 'i' not in edge.free_symbols

    edge = dace.InterstateEdge(assignments={'i': 'i+1'})
    assert 'i' in edge.free_symbols

    edge = dace.InterstateEdge(condition='i < j', assignments={'i': '3'})
    assert 'i' in edge.free_symbols
    assert 'j' in edge.free_symbols

    edge = dace.InterstateEdge(assignments={'j': 'i + 1', 'i': '3'})
    assert 'i' in edge.free_symbols
    assert 'j' not in edge.free_symbols


def test_nested_sdfg_free_symbols():
    i, j, k = (dace.symbol(s) for s in 'ijk')

    outer_sdfg = dace.SDFG('outer')
    outer_init_state = outer_sdfg.add_state('outer_init')
    outer_guard_state = outer_sdfg.add_state('outer_guard')
    outer_body_state_1 = outer_sdfg.add_state('outer_body_1')
    outer_body_state_2 = outer_sdfg.add_state('outer_body_2')
    outer_exit_state = outer_sdfg.add_state('outer_exit')
    outer_sdfg.add_edge(outer_init_state, outer_guard_state, dace.InterstateEdge(assignments={'i': '0'}))
    outer_sdfg.add_edge(outer_guard_state, outer_body_state_1,
                        dace.InterstateEdge(condition='i < 10', assignments={'j': 'i + 1'}))
    outer_sdfg.add_edge(outer_guard_state, outer_exit_state, dace.InterstateEdge(condition='i >= 10'))
    outer_sdfg.add_edge(outer_body_state_1, outer_guard_state,
                        dace.InterstateEdge(condition='j >= 10', assignments={'i': 'i + 1'}))
    outer_sdfg.add_edge(outer_body_state_1, outer_body_state_2, dace.InterstateEdge(condition='j < 10'))
    outer_sdfg.add_edge(outer_body_state_2, outer_body_state_1, dace.InterstateEdge(assignments={'j': 'j + 1'}))

    inner_sdfg = dace.SDFG('inner')
    inner_init_state = inner_sdfg.add_state('inner_init')
    inner_guard_state = inner_sdfg.add_state('inner_guard')
    inner_body_state = inner_sdfg.add_state('inner_body')
    inner_exit_state = inner_sdfg.add_state('inner_exit')
    inner_sdfg.add_edge(inner_init_state, inner_guard_state, dace.InterstateEdge(assignments={'k': 'j + 1'}))
    inner_sdfg.add_edge(inner_guard_state, inner_body_state, dace.InterstateEdge(condition='k < 10'))
    inner_sdfg.add_edge(inner_guard_state, inner_exit_state,
                        dace.InterstateEdge(condition='k >= 10', assignments={'j': 'j + 1'}))
    inner_sdfg.add_edge(inner_body_state, inner_guard_state, dace.InterstateEdge(assignments={'k': 'k + 1'}))

    outer_body_state_2.add_nested_sdfg(inner_sdfg, {}, {}, symbol_mapping={'j': 'j'})

    assert not outer_sdfg.free_symbols
    assert 'i' not in inner_sdfg.free_symbols
    assert 'j' in inner_sdfg.free_symbols
    assert 'k' not in inner_sdfg.free_symbols


def _build_with_optional_unused_array(create_unused_transient: bool) -> dace.SDFG:
    """The issue #2382 reproducer: two used arrays + an optional unused transient
    ``x`` whose shape uses ``x_shape``.

    :param create_unused_transient: If True, declare the unused ``x`` array.
    :returns: The constructed SDFG.
    """
    sdfg = dace.SDFG('unused_transient')
    state = sdfg.add_state()
    sdfg.add_array('a', (10, ), dace.float64, transient=False)
    sdfg.add_array('b', (10, ), dace.float64, transient=False)
    sdfg.add_symbol('x_shape', dace.int32)
    if create_unused_transient:
        sdfg.add_array('x', ('x_shape', ), dace.float32, transient=True)
    state.add_mapped_tasklet('map', {'__i': '0:10'}, {'__in': dace.Memlet('a[__i]')},
                             '__out = __in + 1.90', {'__out': dace.Memlet('b[__i]')},
                             external_edges=True)
    return sdfg


def test_unused_array_does_not_leak_shape_symbol():
    """Issue #2382: declaring an unused array must not leak its shape symbol into
    the signature -- it must not change the arguments needed to invoke the SDFG."""
    without = _build_with_optional_unused_array(False)
    with_unused = _build_with_optional_unused_array(True)

    # The unused array's shape symbol must not be treated as a used argument.
    assert 'x_shape' not in without.used_symbols(all_symbols=False)
    assert 'x_shape' not in with_unused.used_symbols(all_symbols=False)

    # Declaring the unused array must not perturb the signature at all.
    assert 'x_shape' not in with_unused.arglist()
    assert list(without.arglist().keys()) == list(with_unused.arglist().keys())
    assert without.signature_arglist() == with_unused.signature_arglist()
    assert without.init_signature() == with_unused.init_signature()
    assert 'x_shape' not in with_unused.init_signature()


def test_used_codeblock_array_keeps_shape_symbol():
    """A used array's stride symbol must survive even when its only reference is a
    code block: a guard indexes a 2D array with stride ``S``, so ``S`` must be kept."""
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

    sdfg = dace.SDFG('used_codeblock_array')
    sdfg.add_symbol('S', dace.int32)
    sdfg.add_array('A', (10, 10), dace.int32, strides=(1, dace.symbol('S')))
    sdfg.add_scalar('acc', dace.int32, transient=True)

    loop = LoopRegion('loop', condition_expr='k < 5', loop_var='k', initialize_expr='k = 0', update_expr='k = k + 1')
    sdfg.add_node(loop, is_start_block=True)

    cb = ConditionalBlock('cb')
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion('branch', sdfg=sdfg)
    cb.add_branch(CodeBlock('A[0, k] == 1'), branch)

    set_one = branch.add_state('set_one', is_start_block=True)
    t1 = set_one.add_tasklet('t_set', {}, {'o'}, 'o = 1')
    set_one.add_edge(t1, 'o', set_one.add_write('acc'), None, dace.Memlet('acc[0]'))

    sdfg.validate()

    # ``A`` is referenced only in the conditional guard, but it is genuinely
    # used; its stride symbol ``S`` must therefore be kept.
    assert 'S' in sdfg.used_symbols(all_symbols=False)
    assert 'S' in sdfg.init_signature()


def test_used_array_keeps_symbolic_extent():
    """Guards against the #2382 fix being too aggressive: an array used only through
    a map memlet (no access node, no code-block ref) must still keep its shape/stride
    symbols in the signature."""
    n = dace.symbol('n')
    s = dace.symbol('s')

    sdfg = dace.SDFG('used_via_map')
    sdfg.add_array('a', (n, ), dace.float64, strides=(s, ), transient=False)
    sdfg.add_array('b', (n, ), dace.float64, transient=False)
    state = sdfg.add_state()
    state.add_mapped_tasklet('m', {'__i': '0:n'}, {'__in': dace.Memlet('a[__i]')},
                             '__out = __in + 1.0', {'__out': dace.Memlet('b[__i]')},
                             external_edges=True)
    sdfg.validate()

    used = sdfg.used_symbols(all_symbols=False)
    assert 'n' in used
    assert 's' in used
    assert 'n' in sdfg.arglist()
    assert 's' in sdfg.arglist()


if __name__ == '__main__':
    test_single_state()
    test_state_subgraph()
    test_sdfg()
    test_constants()
    test_interstate_edge_symbols()
    test_nested_sdfg_free_symbols()
    test_unused_array_does_not_leak_shape_symbol()
    test_used_codeblock_array_keeps_shape_symbol()
    test_used_array_keeps_symbolic_extent()
