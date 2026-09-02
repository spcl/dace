# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Red-green tests for name clashes between symbols and data containers when nesting SDFGs
(``add_nested_sdfg`` / ``dace.sdfg.dealias``), including deeply nested SDFGs.
"""

import numpy as np

import dace
from dace.sdfg import SDFG, dealias, nodes
from dace.sdfg.state import LoopRegion


def _parent(shape=(10, )):
    sdfg = dace.SDFG('parent')
    sdfg.add_array('A', shape, dace.float64)
    sdfg.add_array('B', shape, dace.float64)
    state = sdfg.add_state()
    return sdfg, state


def _inner_with_map(range_end='N', param='k'):
    """Inner SDFG with a map whose range uses ``range_end`` and whose parameter is ``param``."""
    inner = dace.SDFG('inner')
    inner.add_array('inA', [10], dace.float64)
    inner.add_array('outB', [10], dace.float64)
    state = inner.add_state(is_start_block=True)
    r = state.add_read('inA')
    w = state.add_write('outB')
    me, mx = state.add_map('m', {param: f'0:{range_end}'})
    t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(r, me, t, dst_conn='a', memlet=dace.Memlet('inA[0]'))
    state.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet('outB[0]'))
    return inner


def _connect(state, node):
    """Connects A -> inA and outB -> B around a nested SDFG node."""
    sdfg = state.sdfg
    r = state.add_read('A')
    w = state.add_write('B')
    state.add_edge(r, None, node, 'inA', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(node, 'outB', w, None, dace.Memlet.from_array('B', sdfg.arrays['B']))


def _all_map_params(sdfg: SDFG):
    result = set()
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            result.update(n.map.params)
    return result


def test_no_clash_identity_is_noop():
    """Identity mappings must not rename anything (guards against excessive renaming)."""
    sdfg, state = _parent()
    sdfg.add_symbol('N', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    inner.add_transient('T', [10], dace.float64)

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'N'})
    _connect(state, node)

    assert 'T' in inner.arrays
    assert 'i' in _all_map_params(inner)
    assert 'N' in inner.symbols
    sdfg.validate()


def test_symbol_clash_with_map_param():
    """Mapping value symbol 'k' clashes with an unrelated internal map parameter 'k'."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)
    inner = _inner_with_map(range_end='N', param='k')
    inner.add_symbol('N', dace.int32)

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    params = _all_map_params(inner)
    # The map parameter must have been renamed away from the introduced symbol
    me = next(n for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert me.map.params[0] != 'k'
    # The range end must now be the parent symbol 'k'
    assert str(me.map.range[0][1] + 1) == 'k'
    sdfg.validate()


def test_symbol_clash_with_interstate_assignment():
    """Mapping value symbol 'k' clashes with an internal interstate-edge assignment 'k'."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    extra = inner.add_state_before(inner.start_state)
    for e in inner.out_edges(extra):
        e.data.assignments['k'] = '5'

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    for e in inner.all_interstate_edges():
        assert 'k' not in e.data.assignments
    me = next(n for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert str(me.map.range[0][1] + 1) == 'k'
    sdfg.validate()


def test_symbol_clash_with_loop_variable():
    """Mapping value symbol 'k' clashes with an internal loop region variable 'k'."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    loop = LoopRegion('loop', 'k < 10', 'k', 'k = 0', 'k = k + 1')
    inner.add_node(loop)
    loop.add_state('body')
    inner.add_edge(inner.start_block, loop, dace.InterstateEdge())

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    for n, _ in inner.all_nodes_recursive():
        if isinstance(n, LoopRegion):
            assert n.loop_variable != 'k'
    sdfg.validate()


def test_symbol_clash_with_transient():
    """Mapping value symbol 'k' clashes with an unrelated internal transient 'k'."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    inner.add_transient('k', [10], dace.float64)

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    assert 'k' not in inner.arrays
    me = next(n for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert str(me.map.range[0][1] + 1) == 'k'
    sdfg.validate()


def test_symbol_clash_with_constant():
    """Mapping value symbol 'k' clashes with an internal constant 'k'."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    inner.add_constant('k', 5)

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    assert 'k' not in inner.constants_prop
    me = next(n for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert str(me.map.range[0][1] + 1) == 'k'
    sdfg.validate()


def test_symbol_mapping_swap():
    """Swapped symbol mapping {M: N, N: M} must be applied simultaneously."""
    sdfg, state = _parent()
    sdfg.add_symbol('M', dace.int32)
    sdfg.add_symbol('N', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    inner.add_symbol('M', dace.int32)
    # Second map using M
    st2 = inner.add_state()
    inner.add_edge(inner.start_block, st2, dace.InterstateEdge())
    me2, mx2 = st2.add_map('m2', {'j': '0:M'})
    t2 = st2.add_tasklet('t2', {}, {}, '')
    st2.add_edge(me2, None, t2, None, dace.Memlet())
    st2.add_edge(t2, None, mx2, None, dace.Memlet())

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'M': 'N', 'N': 'M'})
    _connect(state, node)

    maps = {n.map.params[0]: n.map for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
    # Ranges must be swapped, not unified
    assert str(maps['i'].range[0][1] + 1) == 'M'
    assert str(maps['j'].range[0][1] + 1) == 'N'
    sdfg.validate()


def test_partial_symbol_mapping_identity_completion():
    """Symbols not listed in a partial symbol mapping should behave as identity-mapped."""
    sdfg, state = _parent()
    sdfg.add_symbol('M', dace.int32)
    sdfg.add_symbol('k', dace.int32)
    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    inner.add_symbol('k', dace.int32)
    # Second map uses free symbol k, which is not part of the mapping
    st2 = inner.add_state()
    inner.add_edge(inner.start_block, st2, dace.InterstateEdge())
    me2, mx2 = st2.add_map('m2', {'j': '0:k'})
    t2 = st2.add_tasklet('t2', {}, {}, '')
    st2.add_edge(me2, None, t2, None, dace.Memlet())
    st2.add_edge(t2, None, mx2, None, dace.Memlet())

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'M'})
    _connect(state, node)

    # The inner free symbol 'k' must remain usable (identity semantics), not be renamed away
    maps = {n.map.params[0]: n.map for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
    assert str(maps['j'].range[0][1] + 1) == 'k'
    sdfg.validate()


def test_integrate_parent_data_clash_with_inner_map_param():
    """Parent array name 'A' clashes with an internal map parameter 'A' during integration."""
    sdfg = dace.SDFG('parent')
    sdfg.add_array('A', [2, 10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    # Inner connector descriptor is inequivalent (view integration path), map param is named 'A'
    inner = _inner_with_map(range_end='10', param='A')

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {})
    r = state.add_read('A')
    w = state.add_write('B')
    state.add_edge(r, None, node, 'inA', dace.Memlet('A[0, 0:10]'))
    state.add_edge(node, 'outB', w, None, dace.Memlet.from_array('B', sdfg.arrays['B']))
    node.integrate_into_parent()

    # No data container in the inner SDFG may alias a map parameter
    assert not (_all_map_params(inner) & inner.arrays.keys())
    sdfg.validate()


def test_integrate_parent_memlet_symbol_clash_with_inner_transient():
    """Outer map param 'i' used in the parent memlet clashes with an internal transient 'i'."""
    sdfg = dace.SDFG('parent')
    sdfg.add_array('A', [2, 10], dace.float64)
    sdfg.add_array('B', [2, 10], dace.float64)
    state = sdfg.add_state()
    inner = _inner_with_map(range_end='10', param='p')
    inner.add_transient('i', [10], dace.float64)

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {})
    me, mx = state.add_map('outer', {'i': '0:2'})
    r = state.add_read('A')
    w = state.add_write('B')
    state.add_memlet_path(r, me, node, dst_conn='inA', memlet=dace.Memlet('A[i, 0:10]'))
    state.add_memlet_path(node, mx, w, src_conn='outB', memlet=dace.Memlet('B[i, 0:10]'))
    node.integrate_into_parent()

    # The unrelated internal transient may not alias the outer map parameter
    assert 'i' not in inner.arrays
    sdfg.validate()


def test_deeply_nested_symbol_clash():
    """Mapping value symbol clashes with a name defined in a grandchild SDFG."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)

    # Grandchild: uses free symbol N, defines map param 'k'
    grand = _inner_with_map(range_end='N', param='k')
    grand.add_symbol('N', dace.int32)

    # Middle: connects grandchild identity-style
    mid = dace.SDFG('mid')
    mid.add_array('inA', [10], dace.float64)
    mid.add_array('outB', [10], dace.float64)
    mid.add_symbol('N', dace.int32)
    mstate = mid.add_state()
    gnode = mstate.add_nested_sdfg(grand, {'inA'}, {'outB'}, {'N': 'N'})
    r = mstate.add_read('inA')
    w = mstate.add_write('outB')
    mstate.add_edge(r, None, gnode, 'inA', dace.Memlet.from_array('inA', mid.arrays['inA']))
    mstate.add_edge(gnode, 'outB', w, None, dace.Memlet.from_array('outB', mid.arrays['outB']))

    # Now nest the middle SDFG with a clashing mapping N -> k
    node = state.add_nested_sdfg(mid, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    sdfg.validate()
    # The grandchild map range must resolve to the parent's 'k' without aliasing its own param
    gme = next(n for n, _ in grand.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    if gme.map.params[0] == 'k':
        # If the param kept its name, the range may not refer to 'k' (would self-alias)
        assert 'k' not in str(gme.map.range[0][1])


def test_integrate_parent_data_clash_with_inner_registered_symbol():
    """Parent array name 'C' clashes with a registered (but unused) internal symbol 'C'."""
    sdfg = dace.SDFG('parent')
    sdfg.add_array('C', [2, 10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    inner = _inner_with_map(range_end='10', param='p')
    inner.add_symbol('C', dace.int32)  # Registered but unused

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {})
    r = state.add_read('C')
    w = state.add_write('B')
    state.add_edge(r, None, node, 'inA', dace.Memlet('C[0, 0:10]'))
    state.add_edge(node, 'outB', w, None, dace.Memlet.from_array('B', sdfg.arrays['B']))
    node.integrate_into_parent()

    # Connectors may not refer to registered symbols
    sdfg.validate()


def test_integrate_new_name_clash_with_defined_symbol():
    """The fallback name chosen for an integrated parent array clashes with an inner map param."""
    sdfg = dace.SDFG('parent')
    sdfg.add_array('inA', [2, 10], dace.float64)  # Same name as the inner connector
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    # 'inA_0' is the natural fallback name for integrating parent array 'inA'
    inner = _inner_with_map(range_end='10', param='inA_0')

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {})
    r = state.add_read('inA')
    w = state.add_write('B')
    state.add_edge(r, None, node, 'inA', dace.Memlet('inA[0, 0:10]'))
    state.add_edge(node, 'outB', w, None, dace.Memlet.from_array('B', sdfg.arrays['B']))
    node.integrate_into_parent()

    # No data container in the inner SDFG may alias a map parameter
    assert not (_all_map_params(inner) & inner.arrays.keys())
    sdfg.validate()


def test_symbol_rename_target_clash_with_grandchild():
    """Renaming a clashing inner name must not collide with names defined in a grandchild SDFG."""
    sdfg, state = _parent()
    sdfg.add_symbol('k', dace.int32)

    # Grandchild defines 'k_0', the natural rename fallback for 'k'
    grand = _inner_with_map(range_end='N', param='k_0')
    grand.add_symbol('N', dace.int32)

    mid = dace.SDFG('mid')
    mid.add_array('inA', [10], dace.float64)
    mid.add_array('outB', [10], dace.float64)
    mid.add_symbol('N', dace.int32)
    mid.add_transient('k', [10], dace.float64)  # Clashes with mapping value symbol 'k'
    mstate = mid.add_state(is_start_block=True)
    gnode = mstate.add_nested_sdfg(grand, {'inA'}, {'outB'}, {'N': 'N'})
    r = mstate.add_read('inA')
    w = mstate.add_write('outB')
    mstate.add_edge(r, None, gnode, 'inA', dace.Memlet.from_array('inA', mid.arrays['inA']))
    mstate.add_edge(gnode, 'outB', w, None, dace.Memlet.from_array('outB', mid.arrays['outB']))

    node = state.add_nested_sdfg(mid, {'inA'}, {'outB'}, {'N': 'k'})
    _connect(state, node)

    # The renamed transient may not alias any name defined in the SDFG subtree
    new_names = set(mid.arrays.keys()) - {'inA', 'outB'}
    subtree_names = _all_map_params(grand) | grand.arrays.keys()
    assert not (new_names & subtree_names)
    sdfg.validate()


def test_free_symbol_shared_with_mapping_value_is_not_renamed():
    """An inner free symbol that also appears in a mapping value is the parent's symbol, not a clash.

    Mirrors the shape produced by the Python frontend when a callee's argument descriptor is
    specialized with the caller's symbols: the callee ends up using ``H`` directly while its own
    symbol ``N`` is mapped to an expression over the same ``H``.
    """
    sdfg = dace.SDFG('parent')
    sdfg.add_symbol('H', dace.int32)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()

    inner = _inner_with_map(range_end='N', param='i')
    inner.add_symbol('N', dace.int32)
    # A second map uses the parent's 'H' directly; it is free inside and therefore identity-mapped
    st2 = inner.add_state()
    inner.add_edge(inner.start_block, st2, dace.InterstateEdge())
    me2, mx2 = st2.add_map('m2', {'j': '0:H'})
    t2 = st2.add_tasklet('t2', {}, {}, '')
    st2.add_edge(me2, None, t2, None, dace.Memlet())
    st2.add_edge(t2, None, mx2, None, dace.Memlet())

    node = state.add_nested_sdfg(inner, {'inA'}, {'outB'}, {'N': 'H + 2'})
    _connect(state, node)

    maps = {n.map.params[0]: n.map for n, _ in inner.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
    assert str(maps['j'].range[0][1] + 1) == 'H'
    assert str(maps['i'].range[0][1] + 1) == 'H + 2'
    # No new symbol may have leaked into the parent
    assert sdfg.free_symbols == {'H'}
    sdfg.validate()


def _connector_clash_sdfg(connector: str, outer_memlet: str, shape):
    """Nested SDFG whose input connector is called ``connector``, reading ``outer_memlet`` of ``A``."""
    sdfg = dace.SDFG('parent')
    sdfg.add_array('A', shape, dace.float64)
    sdfg.add_array('C', [10], dace.float64)
    state = sdfg.add_state()

    inner = dace.SDFG('inner')
    inner.add_array(connector, [10], dace.float64)
    inner.add_array('o', [10], dace.float64)
    istate = inner.add_state()
    r, w = istate.add_read(connector), istate.add_write('o')
    me, mx = istate.add_map('m', {'k': '0:10'})
    t = istate.add_tasklet('t', {'a'}, {'b'}, 'b = a * 2')
    istate.add_memlet_path(r, me, t, dst_conn='a', memlet=dace.Memlet(f'{connector}[k]'))
    istate.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet('o[k]'))

    node = state.add_nested_sdfg(inner, {connector}, {'o'}, {})
    state.add_edge(state.add_read('A'), None, node, connector, dace.Memlet(outer_memlet))
    state.add_edge(node, 'o', state.add_write('C'), None, dace.Memlet('C[0:10]'))
    return sdfg, state, node


def test_connector_clash_with_outer_memlet_symbol():
    """The connector is named after a symbol the outer memlet is written in.

    Integration expresses the memlets inside in the parent's coordinates, so that symbol has to be
    usable within the nested SDFG. A connector of the same name shadows it: the view that replaces
    the connector keeps the name, so the symbol has nowhere to live and code generation emits the
    view's pointer where the symbol was meant.
    """
    sdfg, state, node = _connector_clash_sdfg('s', 'A[s:s + 10]', [20])
    sdfg.add_symbol('s', dace.int64)

    dealias.integrate_nested_sdfg(node.sdfg)

    assert 's' in node.sdfg.symbols
    assert 's' not in node.sdfg.arrays
    sdfg.validate()

    A = np.arange(20, dtype=np.float64)
    C = np.zeros(10, dtype=np.float64)
    sdfg(A=A, C=C, s=5)
    assert np.allclose(C, A[5:15] * 2)


def test_connector_clash_with_parent_shape_symbol():
    """The connector is named after a symbol in the shape of the container being integrated."""
    N = dace.symbol('N')
    sdfg, state, node = _connector_clash_sdfg('N', 'A[0:10]', [N])

    dealias.integrate_nested_sdfg(node.sdfg)

    assert 'N' not in node.sdfg.arrays
    sdfg.validate()

    A = np.arange(10, dtype=np.float64)
    C = np.zeros(10, dtype=np.float64)
    sdfg(A=A, C=C, N=10)
    assert np.allclose(C, A * 2)


def test_connector_clash_with_enclosing_map_parameter():
    """The connector is named after the parameter of the map the nested SDFG sits in."""
    sdfg = dace.SDFG('parent')
    sdfg.add_array('A', [4, 10], dace.float64)
    sdfg.add_array('C', [4, 10], dace.float64)
    state = sdfg.add_state()

    inner = dace.SDFG('inner')
    inner.add_array('i', [10], dace.float64)
    inner.add_array('o', [10], dace.float64)
    istate = inner.add_state()
    r, w = istate.add_read('i'), istate.add_write('o')
    me, mx = istate.add_map('m', {'k': '0:10'})
    t = istate.add_tasklet('t', {'a'}, {'b'}, 'b = a * 2')
    istate.add_memlet_path(r, me, t, dst_conn='a', memlet=dace.Memlet('i[k]'))
    istate.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet('o[k]'))

    node = state.add_nested_sdfg(inner, {'i'}, {'o'}, {})
    entry, exit_ = state.add_map('outer', {'i': '0:4'})
    state.add_memlet_path(state.add_read('A'), entry, node, dst_conn='i', memlet=dace.Memlet('A[i, 0:10]'))
    state.add_memlet_path(node, exit_, state.add_write('C'), src_conn='o', memlet=dace.Memlet('C[i, 0:10]'))

    dealias.integrate_nested_sdfg(node.sdfg)

    assert 'i' not in node.sdfg.arrays
    sdfg.validate()

    A = np.arange(40, dtype=np.float64).reshape(4, 10).copy()
    C = np.zeros((4, 10), dtype=np.float64)
    sdfg(A=A, C=C)
    assert np.allclose(C, A * 2)


if __name__ == '__main__':
    import traceback
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception:
                print(f'FAIL {name}')
                traceback.print_exc()
