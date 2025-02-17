# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import dace.sdfg.nodes
import numpy as np

# Python version of the SDFG below
# @dace.program
# def reduce_with_offsets(A: dace.float64[50, 50], B: dace.float64[25]):
#     B[4:11] = dace.reduce(lambda a,b: a+b, A[25:50, 13:20], axis=0,
#                           identity=0)

reduce_with_offsets = dace.SDFG('reduce_with_offsets')
reduce_with_offsets.add_array('A', [50, 50], dace.float64)
reduce_with_offsets.add_array('B', [25], dace.float64)

state = reduce_with_offsets.add_state()
node_a = state.add_read('A')
node_b = state.add_write('B')
red = state.add_reduce('lambda a,b: a+b', [0], 0)
state.add_nedge(node_a, red, dace.Memlet.simple('A', '25:50, 13:20'))
state.add_nedge(red, node_b, dace.Memlet.simple('B', '4:11'))


def test_offset_reduce():
    A = np.random.rand(50, 50)
    B = np.random.rand(25)

    sdfg = copy.deepcopy(reduce_with_offsets)
    sdfg(A=A, B=B)

    assert np.allclose(B[4:11], np.sum(A[25:50, 13:20], axis=0))


def test_offset_reduce_sequential():
    A = np.random.rand(50, 50)
    B = np.random.rand(25)

    sdfg = copy.deepcopy(reduce_with_offsets)
    sdfg.expand_library_nodes()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.Sequential

    sdfg(A=A, B=B)

    assert np.allclose(B[4:11], np.sum(A[25:50, 13:20], axis=0))


def test_offset_reduce_indices():
    A = np.random.rand(10, 10, 10, 10)
    B = np.ndarray([1], dtype=np.float64)
    B[0] = -np.inf

    reduce_with_indices = dace.SDFG('reduce_with_indices')
    reduce_with_indices.add_array('A', [10, 10, 10, 10], dace.float64)
    reduce_with_indices.add_array('B', [1], dace.float64)

    state = reduce_with_indices.add_state()
    node_a = state.add_read('A')
    node_b = state.add_write('B')
    red = state.add_reduce('lambda a,b: max(a,b)', [0, 1, 2, 3])
    state.add_nedge(node_a, red, dace.Memlet.simple('A', '0, 1, 2, 0:10'))
    state.add_nedge(red, node_b, dace.Memlet.simple('B', '0'))

    reduce_with_indices(A=A, B=B)

    assert np.allclose(B, np.max(A[0, 1, 2, :], axis=0))


if __name__ == '__main__':
    test_offset_reduce()
    test_offset_reduce_sequential()
    test_offset_reduce_indices()
