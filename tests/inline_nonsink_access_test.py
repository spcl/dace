# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def make_sdfg(outer_shape, inner_shape, outer_index, inner_index):
    sdfg = dace.SDFG('inline_nonsink_access_test')
    sdfg.add_array('A', outer_shape, dace.float32)
    sdfg.add_array('B', outer_shape, dace.float32)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    B_out = state.add_write('B')
    t = state.add_tasklet('add', {'a', 'b'}, {'c'}, 'c = a + b')

    state.add_edge(A, None, t, 'a', dace.Memlet.simple('A', outer_index))
    state.add_edge(B, None, t, 'b', dace.Memlet.simple('B', outer_index))
    state.add_edge(t, 'c', B_out, None, dace.Memlet.simple('B', outer_index))

    # Add nested SDFG
    nsdfg = dace.SDFG('nested_ina_test')
    nsdfg.add_array('C', inner_shape, dace.float32)
    nsdfg.add_array('D', inner_shape, dace.float32)

    nstate = nsdfg.add_state()
    t_init = nstate.add_tasklet('init', {}, {'o'}, 'o = 2')
    t_square = nstate.add_tasklet('square', {'i'}, {'o'}, 'o = i * i')
    t_cube = nstate.add_tasklet('cube', {'i'}, {'o'}, 'o = i * i * i')
    C = nstate.add_access('C')
    C2 = nstate.add_access('C')
    D = nstate.add_write('D')

    nstate.add_edge(t_init, 'o', C, None, dace.Memlet.simple('C', inner_index))
    nstate.add_edge(C, None, t_square, 'i', dace.Memlet.simple('C', inner_index))
    nstate.add_edge(t_square, 'o', C2, None, dace.Memlet.simple('C', inner_index))
    nstate.add_edge(C2, None, t_cube, 'i', dace.Memlet.simple('C', inner_index))
    nstate.add_edge(t_cube, 'o', D, None, dace.Memlet.simple('D', inner_index))

    # Add nested SDFG to SDFG
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {'C', 'D'})
    state.add_edge(nsdfg_node, 'C', A, None, dace.Memlet('A'))
    state.add_edge(nsdfg_node, 'D', B, None, dace.Memlet('B'))

    sdfg.save('_dacegraphs/program.sdfg')
    return sdfg


def test_same_shape():
    A = np.random.rand(1).astype(np.float32)
    B = np.random.rand(1).astype(np.float32)

    sdfg = make_sdfg([1], [1], '0', '0')
    sdfg.simplify()

    sdfg.save('_dacegraphs/program.sdfg')
    sdfg(A=A, B=B)

    assert len(sdfg.node(0).nodes()) == 8

    expected = np.array([2**2, (2**2) + (2**6)], dtype=np.float32)
    result = np.array([A[0], B[0]], dtype=np.float32)
    diff = np.linalg.norm(expected - result)
    print('Difference:', diff)
    assert diff <= 1e-6


def test_different_shape():
    A = np.random.rand(20, 3).astype(np.float32)
    B = np.random.rand(20, 3).astype(np.float32)

    sdfg = make_sdfg([20, 3], [60], '1, 0', '3')
    sdfg.simplify()

    sdfg(A=A, B=B)

    assert all(not isinstance(node, dace.nodes.NestedSDFG) for node in sdfg.node(0).nodes())

    expected = np.array([2**2, (2**2) + (2**6)], dtype=np.float32)
    result = np.array([A[1, 0], B[1, 0]], dtype=np.float32)
    diff = np.linalg.norm(expected - result)
    print('Difference:', diff)
    assert diff <= 1e-6


if __name__ == "__main__":
    test_same_shape()
    test_different_shape()
