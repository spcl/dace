# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import dace
import dace.libraries.onnx as donnx
from dace.libraries import blas
from dace.util import utils


def assert_allclose(a, b, rtol=1e-4, atol=1e-5):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


# ==============================================================================
# MatMul Tests (various dimensions and broadcasting)
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("a_shape, b_shape", [
    ([10, 20], [20, 30]),
    ([5, 10], [10, 5]),
    ([100, 50], [50, 20]),
])
def test_matmul_2d(a_shape, b_shape, sdfg_name):
    """Test MatMul operation with 2D matrices."""
    blas.Gemm.default_implementation = "pure"

    sdfg = dace.SDFG(sdfg_name)

    X = np.random.randn(*a_shape).astype(np.float32)
    Z = np.random.randn(*b_shape).astype(np.float32)
    expected_result = X @ Z

    sdfg.add_array("X", a_shape, dace.float32)
    sdfg.add_array("Z", b_shape, dace.float32)
    sdfg.add_array("__return", expected_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))
    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(X=X, Z=Z)

    assert_allclose(expected_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("a_shape, b_shape", [
    ([2, 10, 20], [2, 20, 30]),
    ([3, 5, 10], [3, 10, 5]),
])
def test_matmul_3d_batched(a_shape, b_shape, sdfg_name):
    """Test MatMul operation with 3D batched matrices."""
    blas.Gemm.default_implementation = "pure"

    sdfg = dace.SDFG(sdfg_name)

    X = np.random.randn(*a_shape).astype(np.float32)
    Z = np.random.randn(*b_shape).astype(np.float32)
    expected_result = X @ Z

    sdfg.add_array("X", a_shape, dace.float32)
    sdfg.add_array("Z", b_shape, dace.float32)
    sdfg.add_array("__return", expected_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))
    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(X=X, Z=Z)

    assert_allclose(expected_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("a_shape, b_shape", [
    ([2, 3, 10, 20], [2, 3, 20, 30]),
    ([1, 4, 5, 10], [1, 4, 10, 5]),
])
def test_matmul_4d_batched(a_shape, b_shape, sdfg_name):
    """Test MatMul operation with 4D batched matrices."""
    blas.Gemm.default_implementation = "pure"

    sdfg = dace.SDFG(sdfg_name)

    X = np.random.randn(*a_shape).astype(np.float32)
    Z = np.random.randn(*b_shape).astype(np.float32)
    expected_result = X @ Z

    sdfg.add_array("X", a_shape, dace.float32)
    sdfg.add_array("Z", b_shape, dace.float32)
    sdfg.add_array("__return", expected_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))
    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(X=X, Z=Z)

    assert_allclose(expected_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("a_shape, b_shape", [
    ([10, 20], [1, 20, 30]),
    pytest.param([2, 1, 10, 20], [2, 3, 20, 30]),
])
def test_matmul_broadcast(a_shape, b_shape, sdfg_name):
    """Test MatMul operation with broadcasting."""
    blas.Gemm.default_implementation = "pure"

    sdfg = dace.SDFG(sdfg_name)

    X = np.random.randn(*a_shape).astype(np.float32)
    Z = np.random.randn(*b_shape).astype(np.float32)
    expected_result = X @ Z

    sdfg.add_array("X", a_shape, dace.float32)
    sdfg.add_array("Z", b_shape, dace.float32)
    sdfg.add_array("__return", expected_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))
    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(X=X, Z=Z)

    assert_allclose(expected_result, result)


# ==============================================================================
# Gemm Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("alpha, beta, transA, transB", [
    (1.0, 1.0, 0, 0),
    (2.0, 0.5, 0, 0),
    (1.0, 1.0, 1, 0),
    (1.0, 1.0, 0, 1),
    (1.0, 1.0, 1, 1),
    (2.0, 3.0, 0, 0),
],
                         ids=["a1b1_00", "a2b0p5_00", "a1b1_10", "a1b1_01", "a1b1_11", "a2b3_00"])
def test_gemm(alpha, beta, transA, transB, sdfg_name):
    """Test Gemm operation: Y = alpha * A' * B' + beta * C."""
    blas.Gemm.default_implementation = "pure"

    sdfg = dace.SDFG(sdfg_name)

    M, N, K = 10, 15, 20

    if transA:
        A_shape = [K, M]
        A = np.random.randn(K, M).astype(np.float32)
        A_op = A.T
    else:
        A_shape = [M, K]
        A = np.random.randn(M, K).astype(np.float32)
        A_op = A

    if transB:
        B_shape = [N, K]
        B = np.random.randn(N, K).astype(np.float32)
        B_op = B.T
    else:
        B_shape = [K, N]
        B = np.random.randn(K, N).astype(np.float32)
        B_op = B

    C = np.random.randn(M, N).astype(np.float32)
    expected_result = alpha * (A_op @ B_op) + beta * C

    sdfg.add_array("A", A_shape, dace.float32)
    sdfg.add_array("B", B_shape, dace.float32)
    sdfg.add_array("C", [M, N], dace.float32)
    sdfg.add_array("__return", [M, N], dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXGemm("Gemm", alpha=alpha, beta=beta, transA=transA, transB=transB)
    # Add connectors
    op_node.add_in_connector("A")
    op_node.add_in_connector("B")
    op_node.add_in_connector("C")
    op_node.add_out_connector("Y")
    state.add_node(op_node)

    state.add_edge(state.add_read("A"), None, op_node, "A", sdfg.make_array_memlet("A"))
    state.add_edge(state.add_read("B"), None, op_node, "B", sdfg.make_array_memlet("B"))
    state.add_edge(state.add_read("C"), None, op_node, "C", sdfg.make_array_memlet("C"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(A=A, B=B, C=C)

    assert_allclose(expected_result, result, rtol=1e-4, atol=1e-5)


@pytest.mark.onnx
@pytest.mark.parametrize("a_shape, b_shape", [([2, 4], [4, 3])])
def test_matmul_expansion(a_shape, b_shape, sdfg_name):
    blas.Gemm.default_implementation = "pure"
    sdfg = dace.SDFG(sdfg_name)

    X = np.random.rand(*a_shape).astype(np.float32)
    Z = np.random.rand(*b_shape).astype(np.float32)
    expected_result = X @ Z
    sdfg.add_array("X", a_shape, dace.float32)
    sdfg.add_array("Z", b_shape, dace.float32)
    sdfg.add_array("__return", expected_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))

    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(X=X, Z=Z)

    assert_allclose(expected_result, result)


@pytest.mark.onnx
def test_gemm_no_bias(sdfg_name):
    """Test Gemm operation without bias (C)."""
    blas.Gemm.default_implementation = "pure"

    sdfg = dace.SDFG(sdfg_name)

    M, N, K = 10, 15, 20
    alpha = 2.0

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    expected_result = alpha * (A @ B)

    sdfg.add_array("A", [M, K], dace.float32)
    sdfg.add_array("B", [K, N], dace.float32)
    sdfg.add_array("__return", [M, N], dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXGemm("Gemm", alpha=alpha, beta=0.0, transA=0, transB=0)
    state.add_node(op_node)

    state.add_edge(state.add_read("A"), None, op_node, "A", sdfg.make_array_memlet("A"))
    state.add_edge(state.add_read("B"), None, op_node, "B", sdfg.make_array_memlet("B"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()

    result = sdfg(A=A, B=B)

    assert_allclose(expected_result, result, rtol=1e-4, atol=1e-5)


@pytest.mark.onnx
def test_einsum():

    @dace.program
    def test_einsum(A: dace.float64[5, 4, 3], B: dace.float64[3, 2]):
        Y = dace.define_local([5, 4, 2], dace.float64)
        donnx.ONNXEinsum(Inputs__0=A, Inputs__1=B, Output=Y, equation="bij, jk -> bik")
        return Y

    sdfg = test_einsum.to_sdfg()
    utils.expand_onnx_nodes(sdfg)
    assert any(isinstance(n, blas.Gemm) for n, _ in sdfg.all_nodes_recursive())

    A = np.random.rand(5, 4, 3).astype(np.float64)
    B = np.random.rand(3, 2).astype(np.float64)
    result = test_einsum(A.copy(), B.copy())
    assert_allclose(result, np.einsum("bij ,jk -> bik", A, B))


if __name__ == "__main__":
    matmul_2d_shapes = [([10, 20], [20, 30]), ([5, 10], [10, 5]), ([100, 50], [50, 20])]
    for a_shape, b_shape in matmul_2d_shapes:
        test_matmul_2d(a_shape=a_shape, b_shape=b_shape, sdfg_name=f"test_matmul_2d_{a_shape}_{b_shape}")

    matmul_3d_shapes = [([2, 10, 20], [2, 20, 30]), ([3, 5, 10], [3, 10, 5])]
    for a_shape, b_shape in matmul_3d_shapes:
        test_matmul_3d_batched(a_shape=a_shape, b_shape=b_shape, sdfg_name=f"test_matmul_3d_{a_shape}_{b_shape}")

    matmul_4d_shapes = [([2, 3, 10, 20], [2, 3, 20, 30]), ([1, 4, 5, 10], [1, 4, 10, 5])]
    for a_shape, b_shape in matmul_4d_shapes:
        test_matmul_4d_batched(a_shape=a_shape, b_shape=b_shape, sdfg_name=f"test_matmul_4d_{a_shape}_{b_shape}")

    matmul_broadcast_shapes = [([10, 20], [1, 20, 30]), ([2, 1, 10, 20], [2, 3, 20, 30])]
    for a_shape, b_shape in matmul_broadcast_shapes:
        test_matmul_broadcast(a_shape=a_shape, b_shape=b_shape, sdfg_name=f"test_matmul_broadcast_{a_shape}_{b_shape}")

    gemm_params = [
        (1.0, 1.0, 0, 0),
        (2.0, 0.5, 0, 0),
        (1.0, 1.0, 1, 0),
        (1.0, 1.0, 0, 1),
        (1.0, 1.0, 1, 1),
        (2.0, 3.0, 0, 0),
    ]
    for alpha, beta, transA, transB in gemm_params:
        test_gemm(alpha=alpha,
                  beta=beta,
                  transA=transA,
                  transB=transB,
                  sdfg_name=f"test_gemm_{alpha}_{beta}_{transA}_{transB}")

    test_gemm_no_bias(sdfg_name="test_gemm_no_bias")
