# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import MoveAssignmentOutsideIf
from dace.sdfg import InterstateEdge
from dace.memlet import Memlet
from dace.sdfg.nodes import Tasklet


def one_variable_simple_test(const_value: int = 0):
    """ Test with one variable which has formula and const branch. Uses the given const value """
    sdfg = dace.SDFG('one_variable_simple_test')
    # Create guard state and one state where A is set to 0 and another where it is set using B and some formula
    guard = sdfg.add_state('guard', is_start_block=True)
    formula_state = sdfg.add_state('formula', is_start_block=False)
    const_state = sdfg.add_state('const', is_start_block=False)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)

    # Add tasklet inside states
    formula_tasklet = formula_state.add_tasklet('formula_assign', {'b'}, {'a'}, 'a = 2*b')
    formula_state.add_memlet_path(formula_state.add_read('B'),
                                  formula_tasklet,
                                  memlet=Memlet(data='B', subset='0'),
                                  dst_conn='b')
    formula_state.add_memlet_path(formula_tasklet,
                                  formula_state.add_write('A'),
                                  memlet=Memlet(data='A', subset='0'),
                                  src_conn='a')
    const_tasklet = const_state.add_tasklet('const_assign', {}, {'a'}, f"a = {const_value}")
    const_state.add_memlet_path(const_tasklet,
                                const_state.add_write('A'),
                                memlet=Memlet(data='A', subset='0'),
                                src_conn='a')

    # Create if-else condition such that either the formula state or the const state is executed
    sdfg.add_edge(guard, formula_state, InterstateEdge(condition='B[0] < 0.5'))
    sdfg.add_edge(guard, const_state, InterstateEdge(condition='B[0] >= 0.5'))
    sdfg.simplify()
    sdfg.validate()

    # Assure transformation is applied
    assert sdfg.apply_transformations_repeated([MoveAssignmentOutsideIf]) == 1
    sdfg.simplify()
    # SDFG now starts with a state containing the const_tasklet
    assert const_tasklet in sdfg.start_block.nodes()
    # There should now only be one conditional branch remaining in the entire SDFG.
    conditional = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break
    assert conditional is not None
    assert len(conditional.branches) == 1
    assert conditional.branches[0][0].as_string == '(B[0] < 0.5)'


def multiple_variable_test():
    """ Test with multiple variables where not all appear in the const branch """
    sdfg = dace.SDFG('one_variable_simple_test')
    # Create guard state and one state where A is set to 0 and another where it is set using B and some formula
    guard = sdfg.add_state('guard', is_start_block=True)
    formula_state = sdfg.add_state('formula', is_start_block=False)
    const_state = sdfg.add_state('const', is_start_block=False)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('C', [1], dace.float64)
    sdfg.add_array('D', [1], dace.float64)

    A = formula_state.add_access('A')
    B = formula_state.add_access('B')
    C = formula_state.add_access('C')
    D = formula_state.add_access('D')
    formula_tasklet_a = formula_state.add_tasklet('formula_assign', {'b'}, {'a'}, 'a = 2*b')
    formula_state.add_memlet_path(B, formula_tasklet_a, memlet=Memlet(data='B', subset='0'), dst_conn='b')
    formula_state.add_memlet_path(formula_tasklet_a, A, memlet=Memlet(data='A', subset='0'), src_conn='a')
    formula_tasklet_b = formula_state.add_tasklet('formula_assign', {'c'}, {'b'}, 'a = 2*c')
    formula_state.add_memlet_path(C, formula_tasklet_b, memlet=Memlet(data='C', subset='0'), dst_conn='c')
    formula_state.add_memlet_path(formula_tasklet_b, B, memlet=Memlet(data='B', subset='0'), src_conn='b')
    formula_tasklet_c = formula_state.add_tasklet('formula_assign', {'d'}, {'c'}, 'a = 2*d')
    formula_state.add_memlet_path(D, formula_tasklet_c, memlet=Memlet(data='D', subset='0'), dst_conn='d')
    formula_state.add_memlet_path(formula_tasklet_c, C, memlet=Memlet(data='C', subset='0'), src_conn='c')

    const_tasklet_a = const_state.add_tasklet('const_assign', {}, {'a'}, 'a = 0')
    const_state.add_memlet_path(const_tasklet_a,
                                const_state.add_write('A'),
                                memlet=Memlet(data='A', subset='0'),
                                src_conn='a')
    const_tasklet_b = const_state.add_tasklet('const_assign', {}, {'b'}, 'b = 0')
    const_state.add_memlet_path(const_tasklet_b,
                                const_state.add_write('B'),
                                memlet=Memlet(data='B', subset='0'),
                                src_conn='b')

    # Create if-else condition such that either the formula state or the const state is executed
    sdfg.add_edge(guard, formula_state, InterstateEdge(condition='D[0] < 0.5'))
    sdfg.add_edge(guard, const_state, InterstateEdge(condition='D[0] >= 0.5'))
    sdfg.simplify()
    sdfg.validate()

    # Assure transformation is applied
    assert sdfg.apply_transformations_repeated([MoveAssignmentOutsideIf]) == 1
    sdfg.simplify()
    # There are no other tasklets in the start state beside the const assignment tasklet as there are no other const
    # assignments
    for node in sdfg.start_block.nodes():
        if isinstance(node, Tasklet):
            assert node == const_tasklet_a or node == const_tasklet_b
    # There should now only be one conditional branch remaining in the entire SDFG.
    conditional = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break
    assert conditional is not None
    assert len(conditional.branches) == 1
    assert conditional.branches[0][0].as_string == '(D[0] < 0.5)'


def multiple_variable_not_all_const_test():
    """ Test with multiple variables where not all get const-assigned in const branch """
    sdfg = dace.SDFG('one_variable_simple_test')
    # Create guard state and one state where A is set to 0 and another where it is set using B and some formula
    guard = sdfg.add_state('guard', is_start_block=True)
    formula_state = sdfg.add_state('formula', is_start_block=False)
    const_state = sdfg.add_state('const', is_start_block=False)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('C', [1], dace.float64)

    A = formula_state.add_access('A')
    B = formula_state.add_access('B')
    C = formula_state.add_access('C')
    formula_tasklet_a = formula_state.add_tasklet('formula_assign', {'b'}, {'a'}, 'a = 2*b')
    formula_state.add_memlet_path(B, formula_tasklet_a, memlet=Memlet(data='B', subset='0'), dst_conn='b')
    formula_state.add_memlet_path(formula_tasklet_a, A, memlet=Memlet(data='A', subset='0'), src_conn='a')
    formula_tasklet_b = formula_state.add_tasklet('formula_assign', {'c'}, {'b'}, 'a = 2*c')
    formula_state.add_memlet_path(C, formula_tasklet_b, memlet=Memlet(data='C', subset='0'), dst_conn='c')
    formula_state.add_memlet_path(formula_tasklet_b, B, memlet=Memlet(data='B', subset='0'), src_conn='b')

    const_tasklet_a = const_state.add_tasklet('const_assign', {}, {'a'}, 'a = 0')
    const_state.add_memlet_path(const_tasklet_a,
                                const_state.add_write('A'),
                                memlet=Memlet(data='A', subset='0'),
                                src_conn='a')
    const_tasklet_b = const_state.add_tasklet('const_assign', {'c'}, {'b'}, 'b = 1.5 * c')
    const_state.add_memlet_path(const_state.add_read('C'),
                                const_tasklet_b,
                                memlet=Memlet(data='C', subset='0'),
                                dst_conn='c')
    const_state.add_memlet_path(const_tasklet_b,
                                const_state.add_write('B'),
                                memlet=Memlet(data='B', subset='0'),
                                src_conn='b')

    # Create if-else condition such that either the formula state or the const state is executed
    sdfg.add_edge(guard, formula_state, InterstateEdge(condition='C[0] < 0.5'))
    sdfg.add_edge(guard, const_state, InterstateEdge(condition='C[0] >= 0.5'))
    sdfg.simplify()
    sdfg.validate()

    # Assure transformation is applied
    assert sdfg.apply_transformations_repeated([MoveAssignmentOutsideIf]) == 1
    # There are no other tasklets in the start state beside the const assignment tasklet as there are no other const
    # assignments
    for node in sdfg.start_state.nodes():
        if isinstance(node, Tasklet):
            assert node == const_tasklet_a or node == const_tasklet_b
    # The conditional should still have two conditional branches
    conditional = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break
    assert conditional is not None
    assert len(conditional.branches) == 2


if __name__ == '__main__':
    one_variable_simple_test(0)
    one_variable_simple_test(2)
    multiple_variable_test()
    multiple_variable_not_all_const_test()
