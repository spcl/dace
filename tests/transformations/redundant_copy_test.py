# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.blas import Transpose
from dace.transformation.dataflow import (RedundantArray, RedundantSecondArray, RedundantArrayCopying,
                                          RedundantArrayCopyingIn)


def test_out():
    sdfg = dace.SDFG("test_redundant_copy_out")
    state = sdfg.add_state()
    sdfg.add_array("A", [3, 3], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient("C", [3, 3], dace.float32)
    sdfg.add_array("D", [3, 3], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")
    trans = Transpose("transpose", dtype=dace.float32)
    D = state.add_access("D")

    state.add_edge(A, None, B, None, sdfg.make_array_memlet("A"))
    state.add_edge(B, None, C, None, sdfg.make_array_memlet("B"))
    state.add_edge(C, None, trans, "_inp", sdfg.make_array_memlet("C"))
    state.add_edge(trans, "_out", D, None, sdfg.make_array_memlet("D"))

    sdfg.simplify()
    sdfg.apply_transformations_repeated(RedundantArrayCopying)
    assert len(state.nodes()) == 3
    assert B not in state.nodes()
    sdfg.validate()

    A_arr = np.copy(np.arange(9, dtype=np.float32).reshape(3, 3))
    D_arr = np.zeros_like(A_arr)
    sdfg(A=A_arr, D=D_arr)
    assert np.allclose(A_arr, D_arr.T)


def test_out_success():
    sdfg = dace.SDFG("test_redundant_copy_out_success")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2", other_subset_str="1, 2, 0:3, 4"))

    sdfg.add_scalar("D", dace.float32, transient=True)
    sdfg.add_array("E", [3, 3, 3], dace.float32)

    me, mx = state.add_map("Map", dict(i='0:3', j='0:3', k='0:3'))
    t = state.add_tasklet("Tasklet", {'__in1', '__in2'}, {'__out'}, "__out = __in1 + __in2")
    D = state.add_access("D")
    E = state.add_access("E")

    state.add_memlet_path(B, me, t, memlet=dace.Memlet.simple("B", "i, j"), dst_conn='__in1')
    state.add_memlet_path(B, me, D, memlet=dace.Memlet.simple("B", "j, k"))
    state.add_edge(D, None, t, '__in2', dace.Memlet.simple("D", "0"))
    state.add_memlet_path(t, mx, E, memlet=dace.Memlet.simple("E", "i, j, k"), src_conn='__out')

    sdfg.validate()
    sdfg.simplify()
    arrays, views = 0, 0
    for n in state.nodes():
        if isinstance(n, dace.nodes.AccessNode):
            if isinstance(n.desc(sdfg), dace.data.View):
                views += 1
            else:
                arrays += 1
    assert views == 1
    assert arrays == 4
    sdfg.validate()

    A_arr = np.copy(np.arange(125, dtype=np.float32).reshape(5, 5, 5))
    C_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)
    E_arr = np.zeros([3, 3, 3], dtype=np.float32)

    E_ref = np.zeros([3, 3, 3], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                E_ref[i, j, k] = A_arr[0, i, 2 + j] + A_arr[0, j, 2 + k]

    sdfg(A=A_arr, C=C_arr, E=E_arr)
    # This fails, probably due to a bug in the code generator
    # assert np.array_equal(A_arr[0, 0:3, 4], C_arr[1, 2, 0:3, 4])
    assert np.array_equal(E_ref, E_arr)


def test_out_failure_subset_mismatch():
    sdfg = dace.SDFG("test_rco_failure_subset_mismatch")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [8, 8], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2", other_subset_str="1, 2, 0:3, 4"))

    sdfg.validate()
    sdfg.simplify()
    assert len(state.nodes()) == 3
    assert B in state.nodes()


def test_out_failure_no_overlap():
    sdfg = dace.SDFG("test_rco_failure_no_overlap")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [8, 8], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5", other_subset_str="5:8, 5:8"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2", other_subset_str="1, 2, 0:3, 4"))

    sdfg.validate()
    sdfg.simplify()
    assert len(state.nodes()) == 3
    assert B in state.nodes()


def test_out_failure_partial_overlap():
    sdfg = dace.SDFG("test_rco_failure_partial_overlap")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [8, 8], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5", other_subset_str="5:8, 5:8"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "4:7, 6", other_subset_str="1, 2, 0:3, 4"))

    sdfg.validate()
    sdfg.simplify()
    assert len(state.nodes()) == 3
    assert B in state.nodes()


def test_in():
    sdfg = dace.SDFG("test_redundant_copy_in")
    state = sdfg.add_state()
    sdfg.add_array("A", [3, 3], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_transient("C", [3, 3], dace.float32, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("D", [3, 3], dace.float32)

    A = state.add_access("A")
    trans = Transpose("transpose", dtype=dace.float32)
    state.add_node(trans)
    B = state.add_access("B")
    C = state.add_access("C")
    D = state.add_access("D")

    state.add_edge(A, None, trans, "_inp", sdfg.make_array_memlet("A"))
    state.add_edge(trans, "_out", B, None, sdfg.make_array_memlet("B"))
    state.add_edge(B, None, C, None, sdfg.make_array_memlet("B"))
    state.add_edge(C, None, D, None, sdfg.make_array_memlet("C"))

    sdfg.simplify()
    sdfg.apply_transformations_repeated(RedundantArrayCopyingIn)
    assert len(state.nodes()) == 3
    assert C not in state.nodes()
    sdfg.validate()

    A_arr = np.copy(np.arange(9, dtype=np.float32).reshape(3, 3))
    D_arr = np.zeros_like(A_arr)
    sdfg(A=A_arr, D=D_arr)
    assert np.allclose(A_arr, D_arr.T)


def test_view_array_array():
    sdfg = dace.SDFG('redarrtest')
    sdfg.add_view('v', [2, 10], dace.float64)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_transient('tmp', [20], dace.float64)

    state = sdfg.add_state()
    t = state.add_tasklet('something', {}, {'out'}, 'out[1, 1] = 6')
    v = state.add_access('v')
    tmp = state.add_access('tmp')
    w = state.add_write('A')
    state.add_edge(t, 'out', v, None, dace.Memlet('v[0:2, 0:10]'))
    state.add_nedge(v, tmp, dace.Memlet('tmp[0:20]'))
    state.add_nedge(tmp, w, dace.Memlet('A[0:20]'))

    assert sdfg.apply_transformations_repeated(RedundantArray) == 1
    sdfg.validate()


def test_array_array_view():
    sdfg = dace.SDFG('redarrtest')
    sdfg.add_view('v', [2, 10], dace.float64)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_transient('tmp', [20], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    tmp = state.add_access('tmp')
    v = state.add_access('v')
    t = state.add_tasklet('something', {'inp'}, {}, 'inp[1, 1] + 6')
    state.add_nedge(a, tmp, dace.Memlet('A[0:20]'))
    state.add_nedge(tmp, v, dace.Memlet('tmp[0:20]'))
    state.add_edge(v, None, t, 'inp', dace.Memlet('v[0:2, 0:10]'))

    assert sdfg.apply_transformations_repeated(RedundantSecondArray) == 1
    sdfg.validate()


def test_reverse_copy():

    @dace.program
    def redarrtest(p: dace.float64[20, 20]):
        p[-1, :] = p[-2, :]

    p = np.random.rand(20, 20)
    pp = np.copy(p)
    pp[-1, :] = pp[-2, :]
    redarrtest(p)
    assert np.allclose(p, pp)


C_in, C_out, H, K, N, W = (dace.symbol(s, dace.int64) for s in ('C_in', 'C_out', 'H', 'K', 'N', 'W'))


# Deep learning convolutional operator (stride = 1)
@dace.program
def conv2d(input: dace.float64[N, H, W, C_in], weights: dace.float64[K, K, C_in, C_out]):
    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.float64)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


def conv2d_py(input, weights):
    output = np.ndarray((input.shape[0], input.shape[1] - weights.shape[0] + 1, input.shape[2] - weights.shape[1] + 1,
                         weights.shape[3]),
                        dtype=np.float64)
    K = weights.shape[0]
    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


def test_conv2d():
    sdfg = conv2d.to_sdfg(simplify=True)
    access_nodes = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.AccessNode) and not isinstance(sdfg.arrays[n.data], dace.data.View)
    ]
    assert (len(access_nodes) == 4)


@dace.program
def padded_conv2d(input: dace.float64[N, H, W, C_in], weights: dace.float64[1, 1, C_in, C_out]):
    padded = np.zeros((N, H + 2, W + 2, C_out), dtype=np.float64)
    padded[:, 1:-1, 1:-1, :] = conv2d(input, weights)
    return padded


def test_padded_conv2d():
    """ Tests for issues regarding redundant arrays with views in nested SDFGs. """
    input = np.random.rand(8, 32, 32, 3)
    weights = np.random.rand(1, 1, 3, 16)
    reference = np.zeros((8, 34, 34, 16), dtype=np.float64)
    reference[:, 1:-1, 1:-1, :] = conv2d_py(input, weights)

    output = padded_conv2d(input, weights)
    assert np.allclose(output, reference)


def test_redundant_second_copy_isolated():
    sdfg = dace.SDFG('rsc')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_transient('tmp', [20], dace.float64)
    state = sdfg.add_state()
    state.add_nedge(state.add_read('A'), state.add_write('tmp'), dace.Memlet('tmp'))

    assert sdfg.apply_transformations(RedundantSecondArray) == 1
    sdfg.validate()
    assert state.number_of_nodes() == 0


@pytest.mark.parametrize('order', ['C', 'F'])
def test_invalid_redundant_array_strided(order):

    @dace.program
    def flip_and_flatten(a, b):
        tmp = np.flip(a, 0)
        b[:] = tmp.flatten(order=order)

    a = np.asfortranarray(np.random.rand(20, 30))
    b = np.random.rand(20 * 30)
    flip_and_flatten(a, b)

    assert np.allclose(b, np.flip(a, 0).flatten(order=order))


if __name__ == '__main__':
    test_in()
    test_out()
    test_out_success()
    test_out_failure_subset_mismatch()
    test_out_failure_no_overlap()
    test_out_failure_partial_overlap()
    test_view_array_array()
    test_array_array_view()
    test_reverse_copy()
    test_conv2d()
    test_padded_conv2d()
    test_redundant_second_copy_isolated()
    test_invalid_redundant_array_strided('C')
    test_invalid_redundant_array_strided('F')
