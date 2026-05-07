# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import copy
from typing import Tuple

import dace
from dace import nodes, data as dace_data
from dace.libraries.standard import Transpose
from dace.transformation.dataflow import (RedundantArray, RedundantSecondArray, RedundantArrayCopying,
                                          RedundantArrayCopyingIn)

from . import utility


def test_reshaping_with_redundant_arrays():

    def make_sdfg() -> Tuple[dace.SDFG, dace.nodes.AccessNode, dace.nodes.AccessNode, dace.nodes.AccessNode]:
        sdfg = dace.SDFG("slicing_sdfg")
        _, input_desc = sdfg.add_array(
            "input",
            shape=(6, 6, 6),
            transient=False,
            strides=None,
            dtype=dace.float64,
        )
        _, a_desc = sdfg.add_array(
            "a",
            shape=(6, 6, 6),
            transient=True,
            strides=None,
            dtype=dace.float64,
        )
        _, b_desc = sdfg.add_array(
            "b",
            shape=(36, 1, 6),
            transient=True,
            strides=None,
            dtype=dace.float64,
        )
        _, output_desc = sdfg.add_array(
            "output",
            shape=(36, 1, 6),
            transient=False,
            strides=None,
            dtype=dace.float64,
        )
        state = sdfg.add_state("state", is_start_block=True)
        input_an = state.add_access("input")
        a_an = state.add_access("a")
        b_an = state.add_access("b")
        output_an = state.add_access("output")

        state.add_edge(
            input_an,
            None,
            a_an,
            None,
            dace.Memlet.from_array("input", input_desc),
        )
        state.add_edge(a_an, None, b_an, None,
                       dace.Memlet.simple(
                           "a",
                           subset_str="0:6, 0:6, 0:6",
                           other_subset_str="0:36, 0, 0:6",
                       ))
        state.add_edge(
            b_an,
            None,
            output_an,
            None,
            dace.Memlet.from_array("b", b_desc),
        )
        sdfg.validate()
        assert state.number_of_nodes() == 4
        assert len(sdfg.arrays) == 4
        return sdfg, a_an, b_an, output_an

    def apply_trafo(
        sdfg: dace.SDFG,
        in_array: dace.nodes.AccessNode,
        out_array: dace.nodes.AccessNode,
        will_not_apply: bool = False,
        will_create_view: bool = False,
    ) -> dace.SDFG:
        trafo = RedundantArray()

        candidate = {type(trafo).in_array: in_array, type(trafo).out_array: out_array}
        state = sdfg.start_block
        state_id = sdfg.node_id(state)
        initial_arrays = len(sdfg.arrays)
        initial_access_nodes = state.number_of_nodes()

        trafo.setup_match(sdfg, sdfg.cfg_id, state_id, candidate, 0, override=True)
        if trafo.can_be_applied(state, 0, sdfg):
            ret = trafo.apply(state, sdfg)
            if ret is not None:  # A view was created
                if will_create_view:
                    return sdfg
                assert False, f"A view was created instead removing '{in_array.data}'."
            sdfg.validate()
            assert state.number_of_nodes() == initial_access_nodes - 1
            assert len(sdfg.arrays) == initial_arrays - 1
            assert in_array.data not in sdfg.arrays
            return sdfg

        if will_not_apply:
            return sdfg
        assert False, "Could not apply the transformation."

    input_array = np.array(np.random.rand(6, 6, 6), dtype=np.float64, order='C')
    ref = input_array.reshape((36, 1, 6)).copy()
    output_step1 = np.zeros_like(ref)
    output_step2 = np.zeros_like(ref)

    # The Memlet between `a` and `b` is a reshaping Memlet, that are not handled.
    sdfg, a_an, b_an, output_an = make_sdfg()
    sdfg = apply_trafo(sdfg, in_array=a_an, out_array=b_an, will_create_view=True)

    sdfg(input=input_array, output=output_step1)
    assert np.all(ref == output_step1)

    # The Memlet between `b` and `output` is not reshaping, and thus `b` should be removed.
    sdfg = apply_trafo(sdfg, in_array=b_an, out_array=output_an)

    sdfg(input=input_array, output=output_step2)
    assert np.all(ref == output_step2)


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
    assert views == 0
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
    assert np.array_equal(A_arr[0, 0:3, 4], C_arr[1, 2, 0:3, 4])


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


def _make_reshaping_not_zero_started_input_sdfg(
    a_has_larger_rank_than_b: bool, ) -> Tuple[dace.SDFG, dace.SDFGState, nodes.AccessNode, nodes.MapEntry]:
    sdfg = dace.SDFG(utility.unique_name("non_zero_offset_reshaping"))
    state = sdfg.add_state(is_start_block=True)

    a_shape = (10, 1, 2, 20) if a_has_larger_rank_than_b else (10, 20)
    sdfg.add_array(
        "a",
        shape=a_shape,
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(5, 1, 10),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "c",
        shape=(
            10,
            20,
        ),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c = (state.add_access(name) for name in "abc")

    state.add_edge(
        a, None, b, None,
        dace.Memlet("a[5:10, 0, 1, 3:13] -> [0:5, 0, 0:10]")
        if a_has_larger_rank_than_b else dace.Memlet("a[5:10, 3:13] -> [0:5, 0, 0:10]"))

    _, me, _ = state.add_mapped_tasklet(
        "comp",
        map_ranges={
            "__i": "5:10",
            "__j": "3:13"
        },
        inputs={"__in": dace.Memlet("b[__i - 5, 0, __j - 3]")},
        code="__out = __in + 1.3",
        outputs={"__out": dace.Memlet("c[__i, __j]")},
        external_edges=True,
        input_nodes={b},
        output_nodes={c},
    )
    sdfg.validate()

    return sdfg, state, a, me


@pytest.mark.parametrize("a_has_larger_rank_than_b", [True, False])
def test_reshaping_not_zero_started_input(a_has_larger_rank_than_b: bool):
    sdfg, state, a, me = _make_reshaping_not_zero_started_input_sdfg(a_has_larger_rank_than_b=a_has_larger_rank_than_b)

    assert utility.count_nodes(state, nodes.AccessNode) == 3
    assert utility.count_nodes(state, nodes.MapEntry) == 1

    assert "b" in sdfg.arrays
    assert not isinstance(sdfg.arrays["b"], dace_data.View)
    b_original_strides = copy.deepcopy(sdfg.arrays["b"].strides)

    ref, res = utility.make_sdfg_args(sdfg)
    utility.compile_and_run_sdfg(sdfg, **ref)

    nb_applies = sdfg.apply_transformations_repeated(RedundantSecondArray, validate=True, validate_all=True)
    assert nb_applies == 1

    assert utility.count_nodes(state, nodes.AccessNode) == 3
    assert utility.count_nodes(state, nodes.MapEntry) == 1

    assert "b" in sdfg.arrays
    assert isinstance(sdfg.arrays["b"], dace_data.View)
    assert len(sdfg.arrays["b"].strides) == len(b_original_strides)
    assert not all(ob == cb for ob, cb in zip(b_original_strides, sdfg.arrays["b"].strides))

    utility.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


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
    test_reshaping_not_zero_started_input(True)
    test_reshaping_not_zero_started_input(False)
