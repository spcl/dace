# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for matmul accumulation (beta factor) across all matmul implementations."""
import pytest
import numpy as np
import dace
import dace.libraries.blas as blas
from dace.library import change_default


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float32),
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("cuBLAS", dace.float32, marks=pytest.mark.gpu),
    pytest.param("cuBLAS", dace.float64, marks=pytest.mark.gpu),
    pytest.param("OpenBLAS", dace.float32, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack)
])
def test_batched_matmul_accumulate(implementation: str, dtype):
    """Test batched matmul with non-zero beta (accumulation into existing output)"""
    b, m, n, k = 3, 8, 8, 8

    sdfg = dace.SDFG('batched_matmul_accumulate')

    # Add arrays
    sdfg.add_array("A", [b, m, k], dtype)
    sdfg.add_array("B", [b, k, n], dtype)
    sdfg.add_array("C", [b, m, n], dtype)

    state = sdfg.add_state()

    a_in = state.add_read("A")
    b_in = state.add_read("B")
    c_out = state.add_write("C")

    bmm_node = blas.nodes.batched_matmul.BatchedMatMul('bmm', )
    bmm_node.alpha = 1.0
    bmm_node.beta = 2.0

    state.add_node(bmm_node)
    state.add_edge(a_in, None, bmm_node, '_a', dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(b_in, None, bmm_node, '_b', dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(bmm_node, '_c', c_out, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))

    with change_default(blas, implementation):
        sdfg.expand_library_nodes()
        sdfg.validate()

        A = np.random.rand(b, m, k).astype(dtype.as_numpy_dtype())
        B = np.random.rand(b, k, n).astype(dtype.as_numpy_dtype())
        C_initial = np.random.rand(b, m, n).astype(dtype.as_numpy_dtype())
        C = C_initial.copy()

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)

        # C = alpha * A @ B + beta * C_initial = 1.0 * A @ B + 2.0 * C_initial
        ref = A @ B + 2.0 * C_initial

        assert np.allclose(ref, C), f"Test failed for {implementation} with dtype {dtype}"


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float32),
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("cuBLAS", dace.float32, marks=pytest.mark.gpu),
    pytest.param("cuBLAS", dace.float64, marks=pytest.mark.gpu),
    pytest.param("OpenBLAS", dace.float32, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack)
])
def test_gemm_accumulate(implementation: str, dtype):
    """Test GEMM with non-zero beta (accumulation into existing output)"""
    m, n, k = 16, 16, 16

    sdfg = dace.SDFG('gemm_accumulate')

    # Add arrays
    sdfg.add_array("A", [m, k], dtype)
    sdfg.add_array("B", [k, n], dtype)
    sdfg.add_array("C", [m, n], dtype)

    state = sdfg.add_state()

    a_in = state.add_read("A")
    b_in = state.add_read("B")
    c_out = state.add_write("C")

    # Create GEMM node with alpha=1.5 and beta=2.0
    # For BLAS implementations, cin=False even when beta != 0, because BLAS handles reading C
    gemm_node = blas.nodes.gemm.Gemm('gemm', alpha=1.5, beta=2.0, cin=False)

    state.add_node(gemm_node)
    state.add_edge(a_in, None, gemm_node, '_a', dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(b_in, None, gemm_node, '_b', dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(gemm_node, '_c', c_out, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))

    with change_default(blas, implementation):
        sdfg.expand_library_nodes()
        sdfg.validate()

        A = np.random.rand(m, k).astype(dtype.as_numpy_dtype())
        B = np.random.rand(k, n).astype(dtype.as_numpy_dtype())
        C_initial = np.random.rand(m, n).astype(dtype.as_numpy_dtype())
        C = C_initial.copy()

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)

        # C = alpha * A @ B + beta * C_initial = 1.5 * A @ B + 2.0 * C_initial
        ref = 1.5 * (A @ B) + 2.0 * C_initial

        assert np.allclose(ref, C), f"Test failed for {implementation} with dtype {dtype}"


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float32),
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("cuBLAS", dace.float32, marks=pytest.mark.gpu),
    pytest.param("cuBLAS", dace.float64, marks=pytest.mark.gpu),
    pytest.param("OpenBLAS", dace.float32, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack)
])
def test_gemv_accumulate(implementation: str, dtype):
    """Test GEMV with non-zero beta (accumulation into existing output)"""
    m, n = 32, 32

    sdfg = dace.SDFG('gemv_accumulate')

    # Add arrays
    sdfg.add_array("A", [m, n], dtype)
    sdfg.add_array("x", [n], dtype)
    sdfg.add_array("y", [m], dtype)

    state = sdfg.add_state()

    a_in = state.add_read("A")
    x_in = state.add_read("x")
    y_in = state.add_read("y")
    y_out = state.add_write("y")

    gemv_node = blas.nodes.gemv.Gemv('gemv', alpha=1.5, beta=2.0)

    state.add_node(gemv_node)
    state.add_edge(a_in, None, gemv_node, '_A', dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(x_in, None, gemv_node, '_x', dace.Memlet.from_array("x", sdfg.arrays["x"]))
    # For GEMV, when beta != 0, _y is both an input and output
    state.add_edge(y_in, None, gemv_node, '_y', dace.Memlet.from_array("y", sdfg.arrays["y"]))
    state.add_edge(gemv_node, '_y', y_out, None, dace.Memlet.from_array("y", sdfg.arrays["y"]))

    with change_default(blas, implementation):
        sdfg.expand_library_nodes()
        sdfg.validate()

        A = np.random.rand(m, n).astype(dtype.as_numpy_dtype())
        x = np.random.rand(n).astype(dtype.as_numpy_dtype())
        y_initial = np.random.rand(m).astype(dtype.as_numpy_dtype())
        y = y_initial.copy()

        csdfg = sdfg.compile()
        csdfg(A=A, x=x, y=y)

        # y = alpha * A @ x + beta * y_initial = 1.5 * A @ x + 2.0 * y_initial
        ref = 1.5 * (A @ x) + 2.0 * y_initial

        assert np.allclose(ref, y), f"Test failed for {implementation} with dtype {dtype}"

@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float32),
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("cuBLAS", dace.float32, marks=pytest.mark.gpu),
    pytest.param("cuBLAS", dace.float64, marks=pytest.mark.gpu),
    pytest.param("OpenBLAS", dace.float32, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack)
])
def test_batched_matmul_accumulate(implementation: str, dtype):
    """Test batched matmul with non-zero beta (accumulation into existing output)"""
    b, m, n, k = 3, 8, 8, 8

    sdfg = dace.SDFG('batched_matmul_accumulate')

    # Add arrays
    sdfg.add_array("A", [b, m, k], dtype)
    sdfg.add_array("B", [b, k, n], dtype)
    sdfg.add_array("C", [b, m, n], dtype)

    # Create state and add BatchedMatMul node with beta=2.0
    state = sdfg.add_state()

    a_in = state.add_read("A")
    b_in = state.add_read("B")
    c_out = state.add_write("C")

    # Create BatchedMatMul node with alpha=1.0 and beta=2.0
    bmm_node = blas.nodes.batched_matmul.BatchedMatMul('bmm')
    bmm_node.alpha = 1.0
    bmm_node.beta = 2.0

    state.add_node(bmm_node)
    state.add_edge(a_in, None, bmm_node, '_a', dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(b_in, None, bmm_node, '_b', dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(bmm_node, '_c', c_out, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))

    # Set the implementation
    if implementation != "pure":
        # Expand library node with specific implementation
        bmm_node.implementation = implementation

    with change_default(blas, implementation):
        sdfg.expand_library_nodes()
        sdfg.validate()

        # Create test data
        A = np.random.rand(b, m, k).astype(dtype.as_numpy_dtype())
        B = np.random.rand(b, k, n).astype(dtype.as_numpy_dtype())
        C_initial = np.random.rand(b, m, n).astype(dtype.as_numpy_dtype())
        C = C_initial.copy()

        # Compile and run
        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)

        # Expected result: C = alpha * A @ B + beta * C_initial = 1.0 * A @ B + 2.0 * C_initial
        ref = A @ B + 2.0 * C_initial

        assert np.allclose(ref, C), f"Test failed for {implementation} with dtype {dtype}"
        
if __name__ == "__main__":
    test_batched_matmul_accumulate("pure", dace.float32)
    test_batched_matmul_accumulate("pure", dace.float64)

    test_batched_matmul_accumulate("MKL", dace.float32)
    test_batched_matmul_accumulate("MKL", dace.float64)

    test_batched_matmul_accumulate("OpenBLAS", dace.float32)
    test_batched_matmul_accumulate("OpenBLAS", dace.float64)

    test_gemm_accumulate("pure", dace.float32)
    test_gemm_accumulate("pure", dace.float64)

    test_gemm_accumulate("MKL", dace.float32)
    test_gemm_accumulate("MKL", dace.float64)

    test_gemm_accumulate("OpenBLAS", dace.float32)
    test_gemm_accumulate("OpenBLAS", dace.float64)

    test_gemv_accumulate("pure", dace.float32)
    test_gemv_accumulate("pure", dace.float64)

    test_gemv_accumulate("MKL", dace.float32)
    test_gemv_accumulate("MKL", dace.float64)

    test_gemv_accumulate("OpenBLAS", dace.float32)
    test_gemv_accumulate("OpenBLAS", dace.float64)
