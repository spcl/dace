# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import dace
from dace import SDFG, Memlet, nodes
from dace.transformation.interstate import MoveReduceInitOutOfNestedSDFG, InlineSDFG


def create_nested_sdfg_with_reduce_init():
    """Create a nested SDFG that has an initialization state and a reduction state."""
    outer_sdfg = SDFG('outer')
    outer_sdfg.add_array('A', [10, 10], dace.float64)
    outer_sdfg.add_array('B', [10], dace.float64)

    nsdfg = SDFG('nested_reduce')
    nsdfg.add_array('_A', [10, 10], dace.float64)
    nsdfg.add_array('_B', [10], dace.float64)

    init_state = nsdfg.add_state('reduce_init')
    init_state.add_mapped_tasklet('reduce_init_map', {'i': '0:10'}, {},
                                  '__out = 0.0', {'__out': Memlet('_B[i]')},
                                  external_edges=True)

    reduce_state = nsdfg.add_state('reduce')
    reduce_state.add_mapped_tasklet('reduce_map', {
        'i': '0:10',
        'j': '0:10'
    }, {'__in': Memlet('_A[i, j]')},
                                    '__out = __in', {'__out': Memlet('_B[i]', wcr='lambda a, b: a + b')},
                                    external_edges=True)

    nsdfg.add_edge(init_state, reduce_state, dace.InterstateEdge())

    outer_state = outer_sdfg.add_state('main')
    read_a = outer_state.add_read('A')
    write_b = outer_state.add_write('B')

    nsdfg_node = outer_state.add_nested_sdfg(nsdfg, {'_A'}, {'_B'})
    outer_state.add_edge(read_a, None, nsdfg_node, '_A', Memlet('A[0:10, 0:10]'))
    outer_state.add_edge(nsdfg_node, '_B', write_b, None, Memlet('B[0:10]'))

    return outer_sdfg


def test_move_reduce_init_basic():
    """Test basic application of the transformation."""
    sdfg = create_nested_sdfg_with_reduce_init()

    num_states_before = len(list(sdfg.states()))
    assert num_states_before == 1

    nsdfg_node = None
    for node in sdfg.states()[0].nodes():
        if isinstance(node, nodes.NestedSDFG):
            nsdfg_node = node
            break

    assert nsdfg_node is not None
    assert len(list(nsdfg_node.sdfg.nodes())) == 2

    applied = sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 1

    num_states_after = len(list(sdfg.states()))
    assert num_states_after == 2

    nsdfg_node = None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                nsdfg_node = node
                break

    assert nsdfg_node is not None
    assert len(list(nsdfg_node.sdfg.nodes())) == 1


def test_move_reduce_init_enables_inlining():
    """Test that after moving reduce init, the nested SDFG can be inlined."""
    sdfg = create_nested_sdfg_with_reduce_init()

    inline_before = sdfg.apply_transformations(InlineSDFG)
    assert inline_before == 0

    sdfg = create_nested_sdfg_with_reduce_init()

    applied = sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 1

    inline_after = sdfg.apply_transformations(InlineSDFG)
    assert inline_after == 1

    has_nested = False
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                has_nested = True

    assert not has_nested


def test_move_reduce_init_correctness():
    """Test that the transformation preserves correctness."""
    sdfg = create_nested_sdfg_with_reduce_init()

    A = np.random.rand(10, 10)
    B_before = np.zeros(10)
    sdfg(A=A.copy(), B=B_before)

    sdfg = create_nested_sdfg_with_reduce_init()
    sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)

    B_after = np.zeros(10)
    sdfg(A=A.copy(), B=B_after)

    assert np.allclose(B_before, B_after)
    assert np.allclose(B_after, A.sum(axis=1))


def test_move_reduce_init_not_applied_when_written_before():
    """Test that transformation is not applied when the output is written before."""
    outer_sdfg = SDFG('outer')
    outer_sdfg.add_array('A', [10, 10], dace.float64)
    outer_sdfg.add_array('B', [10], dace.float64)

    nsdfg = SDFG('nested_reduce')
    nsdfg.add_array('_A', [10, 10], dace.float64)
    nsdfg.add_array('_B', [10], dace.float64)

    init_state = nsdfg.add_state('reduce_init')
    init_state.add_mapped_tasklet('reduce_init_map', {'i': '0:10'}, {},
                                  '__out = 0.0', {'__out': Memlet('_B[i]')},
                                  external_edges=True)

    reduce_state = nsdfg.add_state('reduce')
    reduce_state.add_mapped_tasklet('reduce_map', {
        'i': '0:10',
        'j': '0:10'
    }, {'__in': Memlet('_A[i, j]')},
                                    '__out = __in', {'__out': Memlet('_B[i]', wcr='lambda a, b: a + b')},
                                    external_edges=True)

    nsdfg.add_edge(init_state, reduce_state, dace.InterstateEdge())

    pre_state = outer_sdfg.add_state('pre_write')
    pre_state.add_mapped_tasklet('pre_init', {'i': '0:10'}, {},
                                 '__out = 1.0', {'__out': Memlet('B[i]')},
                                 external_edges=True)

    main_state = outer_sdfg.add_state('main')
    read_a = main_state.add_read('A')
    read_b = main_state.add_read('B')
    write_b = main_state.add_write('B')

    nsdfg_node = main_state.add_nested_sdfg(nsdfg, {'_A', '_B'}, {'_B'})
    main_state.add_edge(read_a, None, nsdfg_node, '_A', Memlet('A[0:10, 0:10]'))
    main_state.add_edge(read_b, None, nsdfg_node, '_B', Memlet('B[0:10]'))
    main_state.add_edge(nsdfg_node, '_B', write_b, None, Memlet('B[0:10]'))

    outer_sdfg.add_edge(pre_state, main_state, dace.InterstateEdge())

    applied = outer_sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 0


def test_move_reduce_init_not_applied_non_init_first_state():
    """Test that transformation is not applied when first state is not an init state."""
    outer_sdfg = SDFG('outer')
    outer_sdfg.add_array('A', [10, 10], dace.float64)
    outer_sdfg.add_array('B', [10], dace.float64)

    nsdfg = SDFG('nested_reduce')
    nsdfg.add_array('_A', [10, 10], dace.float64)
    nsdfg.add_array('_B', [10], dace.float64)
    nsdfg.add_transient('_tmp', [10, 10], dace.float64)

    state1 = nsdfg.add_state('compute1')
    state1.add_mapped_tasklet('compute_map', {
        'i': '0:10',
        'j': '0:10'
    }, {'__in': Memlet('_A[i, j]')},
                              '__out = __in * 2', {'__out': Memlet('_tmp[i, j]')},
                              external_edges=True)

    state2 = nsdfg.add_state('compute2')
    state2.add_mapped_tasklet('reduce_map', {
        'i': '0:10',
        'j': '0:10'
    }, {'__in': Memlet('_tmp[i, j]')},
                              '__out = __in', {'__out': Memlet('_B[i]', wcr='lambda a, b: a + b')},
                              external_edges=True)

    nsdfg.add_edge(state1, state2, dace.InterstateEdge())

    outer_state = outer_sdfg.add_state('main')
    read_a = outer_state.add_read('A')
    write_b = outer_state.add_write('B')

    nsdfg_node = outer_state.add_nested_sdfg(nsdfg, {'_A'}, {'_B'})
    outer_state.add_edge(read_a, None, nsdfg_node, '_A', Memlet('A[0:10, 0:10]'))
    outer_state.add_edge(nsdfg_node, '_B', write_b, None, Memlet('B[0:10]'))

    applied = outer_sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 0


def test_move_reduce_init_from_library_node():
    """Test transformation on reduce library node expanded via ExpandReducePure."""

    @dace.program
    def reduce_sum(A: dace.float64[10, 10], B: dace.float64[10]):
        B[:] = dace.reduce(lambda a, b: a + b, A, axis=1, identity=0)

    sdfg = reduce_sum.to_sdfg()
    sdfg.simplify()

    sdfg.expand_library_nodes()

    nsdfg_node = None
    main_state = None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                nsdfg_node = node
                main_state = state
                break

    assert nsdfg_node is not None
    num_states_before = len(list(nsdfg_node.sdfg.nodes()))
    assert num_states_before >= 2

    init_state = nsdfg_node.sdfg.start_state
    found_reduce_init = False
    for node in init_state.nodes():
        if isinstance(node, nodes.MapEntry):
            if 'reduce_init' in node.map.label.lower():
                found_reduce_init = True
                break
    assert found_reduce_init

    A = np.random.rand(10, 10)
    B_before = np.zeros(10)
    sdfg(A=A.copy(), B=B_before)

    applied = sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 1

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                nsdfg_node = node
                break

    assert len(list(nsdfg_node.sdfg.nodes())) == num_states_before - 1

    B_after = np.zeros(10)
    sdfg(A=A.copy(), B=B_after)

    assert np.allclose(B_before, B_after)
    assert np.allclose(B_after, A.sum(axis=1))


def test_move_reduce_init_from_library_node_enables_inlining():
    """Test that after applying transformation on expanded reduce, inlining works."""

    @dace.program
    def reduce_sum(A: dace.float64[10, 10], B: dace.float64[10]):
        B[:] = dace.reduce(lambda a, b: a + b, A, axis=1, identity=0)

    sdfg = reduce_sum.to_sdfg()
    sdfg.simplify()

    sdfg.expand_library_nodes()

    inline_before = sdfg.apply_transformations(InlineSDFG)
    assert inline_before == 0

    sdfg = reduce_sum.to_sdfg()
    sdfg.simplify()
    sdfg.expand_library_nodes()

    applied = sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 1

    inline_after = sdfg.apply_transformations(InlineSDFG)
    assert inline_after == 1

    has_nested = False
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                has_nested = True

    assert not has_nested


def test_move_reduce_init_from_library_node_max_reduction():
    """Test transformation on max reduce library node."""

    @dace.program
    def reduce_max(A: dace.float64[10, 10], B: dace.float64[10]):
        B[:] = dace.reduce(lambda a, b: max(a, b), A, axis=1, identity=-np.inf)

    sdfg = reduce_max.to_sdfg()
    sdfg.simplify()

    sdfg.expand_library_nodes()

    A = np.random.rand(10, 10)
    B_before = np.zeros(10)
    sdfg(A=A.copy(), B=B_before)

    applied = sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 1

    B_after = np.zeros(10)
    sdfg(A=A.copy(), B=B_after)

    assert np.allclose(B_before, B_after)
    assert np.allclose(B_after, A.max(axis=1))


def test_move_reduce_init_multiple_reductions():
    """Test transformation with multiple reduce operations applied repeatedly."""

    @dace.program
    def multi_reduce(A: dace.float64[10, 10], B: dace.float64[10], C: dace.float64[10], D: dace.float64[1]):
        B[:] = dace.reduce(lambda a, b: a + b, A, axis=1, identity=0)
        C[:] = dace.reduce(lambda a, b: max(a, b), A, axis=1, identity=-np.inf)
        D[0] = dace.reduce(lambda a, b: a + b, B, identity=0)

    sdfg = multi_reduce.to_sdfg()
    sdfg.simplify()
    sdfg.expand_library_nodes()

    nsdfg_count = 0
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                nsdfg_count += 1

    assert nsdfg_count >= 2

    A = np.random.rand(10, 10)
    B_before = np.zeros(10)
    C_before = np.zeros(10)
    D_before = np.zeros(1)
    sdfg(A=A.copy(), B=B_before, C=C_before, D=D_before)

    applied = sdfg.apply_transformations_repeated(MoveReduceInitOutOfNestedSDFG)
    assert applied >= 2

    B_after = np.zeros(10)
    C_after = np.zeros(10)
    D_after = np.zeros(1)
    sdfg(A=A.copy(), B=B_after, C=C_after, D=D_after)

    assert np.allclose(B_before, B_after)
    assert np.allclose(C_before, C_after)
    assert np.allclose(D_before, D_after)
    assert np.allclose(B_after, A.sum(axis=1))
    assert np.allclose(C_after, A.max(axis=1))
    assert np.allclose(D_after, A.sum())


def test_move_reduce_init_dimension_mismatch():
    """Test transformation when inner array has fewer dimensions than outer array.

    This tests the case where the nested SDFG's array is a squeezed view of the
    outer array (e.g., inner 3D array mapping to outer 4D array with size-1 dim).
    """
    outer_sdfg = SDFG('outer')
    outer_sdfg.add_array('A', [2, 8, 128, 64], dace.float64)
    outer_sdfg.add_array('B', [2, 8, 128, 1], dace.float64)

    nsdfg = SDFG('nested_reduce')
    nsdfg.add_array('_A', [2, 8, 128, 64], dace.float64)
    nsdfg.add_array('_B', [2, 8, 128], dace.float64)

    init_state = nsdfg.add_state('reduce_init')
    init_state.add_mapped_tasklet('reduce_init_map', {
        '_o0': '0:2',
        '_o1': '0:8',
        '_o2': '0:128'
    }, {},
                                  '__out = 0.0', {'__out': Memlet('_B[_o0, _o1, _o2]')},
                                  external_edges=True)

    reduce_state = nsdfg.add_state('reduce')
    reduce_state.add_mapped_tasklet('reduce_map', {
        'i': '0:2',
        'j': '0:8',
        'k': '0:128',
        'l': '0:64'
    }, {'__in': Memlet('_A[i, j, k, l]')},
                                    '__out = __in', {'__out': Memlet('_B[i, j, k]', wcr='lambda a, b: a + b')},
                                    external_edges=True)

    nsdfg.add_edge(init_state, reduce_state, dace.InterstateEdge())

    outer_state = outer_sdfg.add_state('main')
    read_a = outer_state.add_read('A')
    write_b = outer_state.add_write('B')

    nsdfg_node = outer_state.add_nested_sdfg(nsdfg, {'_A'}, {'_B'})
    outer_state.add_edge(read_a, None, nsdfg_node, '_A', Memlet('A[0:2, 0:8, 0:128, 0:64]'))
    outer_state.add_edge(nsdfg_node, '_B', write_b, None, Memlet('B[0:2, 0:8, 0:128, 0]'))

    A = np.random.rand(2, 8, 128, 64)
    B_before = np.zeros((2, 8, 128, 1))
    outer_sdfg(A=A.copy(), B=B_before)

    applied = outer_sdfg.apply_transformations(MoveReduceInitOutOfNestedSDFG)
    assert applied == 1

    outer_sdfg.validate()

    B_after = np.zeros((2, 8, 128, 1))
    outer_sdfg(A=A.copy(), B=B_after)

    assert np.allclose(B_before, B_after)
    expected = A.sum(axis=3, keepdims=True)
    assert np.allclose(B_after, expected)


if __name__ == '__main__':
    test_move_reduce_init_basic()
    test_move_reduce_init_enables_inlining()
    test_move_reduce_init_correctness()
    test_move_reduce_init_not_applied_when_written_before()
    test_move_reduce_init_not_applied_non_init_first_state()
    test_move_reduce_init_from_library_node()
    test_move_reduce_init_from_library_node_enables_inlining()
    test_move_reduce_init_from_library_node_max_reduction()
    test_move_reduce_init_multiple_reductions()
    test_move_reduce_init_dimension_mismatch()
