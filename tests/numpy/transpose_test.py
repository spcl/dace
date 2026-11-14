# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
from common import compare_numpy_output

import dace

M, N = 24, 24


@dace.program
def transpose_test(A: dace.float32[M, N], B: dace.float32[M, N]):
    B[:] = np.transpose(A)


@compare_numpy_output()
def test_transpose_axes0(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 1, 2, 0])


@compare_numpy_output()
def test_transpose_axes1(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 1, 0, 2])


@compare_numpy_output()
def test_transpose_axes2(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 0, 2])


@compare_numpy_output()
def test_transpose_none(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A)


@compare_numpy_output()
def test_transpose_no(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[0, 1, 2, 3])


def test_transpose():
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros([M, N], dtype=np.float32)
    transpose_test(A, B)

    realB = np.transpose(A)
    rel_error = np.linalg.norm(B - realB) / np.linalg.norm(realB)
    print("Relative_error:", rel_error)
    assert rel_error <= 1e-5


# TODO: Enable after fixing HPTT in CI
# @pytest.mark.hptt
@pytest.mark.skip("HPTT is not supported in CI")
def test_hptt():
    with dace.config.set_temporary("library", "ttranspose", "default_implementation", value="HPTT"):
        test_transpose_axes0()
        test_transpose_axes1()
        test_transpose_axes2()
        test_transpose_none()


def test_an_identity_transpose_is_the_array_itself():
    """numpy returns the array itself, as a view, for an identity permutation: no copy is made."""

    @dace.program
    def identity(A: dace.float64[3, 4, 5]):
        return np.transpose(A, (0, 1, 2)) + 1.0

    from dace.libraries.linalg import TensorTranspose

    sdfg = identity.to_sdfg()
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorTranspose)]
    A = np.random.default_rng(3).random((3, 4, 5))
    assert np.allclose(sdfg(A=A), A + 1.0)


def test_a_tensordot_whose_axes_are_in_gemm_order_transposes_nothing():
    """TTGT permutes an operand only when its contracted axes are not already where the GEMM reads them."""
    from dace.libraries.linalg import TensorDot, TensorTranspose

    sdfg = dace.SDFG("tensordot_gemm_order")
    sdfg.add_array("A", [3, 4, 5], dace.float64)
    sdfg.add_array("B", [5, 6], dace.float64)
    sdfg.add_array("C", [3, 4, 6], dace.float64)
    state = sdfg.add_state()
    node = TensorDot("tdot", left_axes=[2], right_axes=[0])
    node.implementation = "TTGT"
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_left_tensor", dace.Memlet("A[0:3, 0:4, 0:5]"))
    state.add_edge(state.add_read("B"), None, node, "_right_tensor", dace.Memlet("B[0:5, 0:6]"))
    state.add_edge(node, "_out_tensor", state.add_write("C"), None, dace.Memlet("C[0:3, 0:4, 0:6]"))
    sdfg.expand_library_nodes(recursive=False)
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorTranspose)]
    rng = np.random.default_rng(5)
    A, B = rng.random((3, 4, 5)), rng.random((5, 6))
    C = np.zeros((3, 4, 6))
    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, np.tensordot(A, B, axes=([2], [0])))


if __name__ == "__main__":
    test_transpose_axes0()
    test_transpose_axes1()
    test_transpose_axes2()
    test_transpose()
    test_transpose_none()
    test_transpose_no()
    test_hptt()
    test_an_identity_transpose_is_the_array_itself()
    test_a_tensordot_whose_axes_are_in_gemm_order_transposes_nothing()
