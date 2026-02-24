# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
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
def test_batchmm(implementation: str, dtype):
    """Test standard 3D batched matmul: [b, m, k] @ [b, k, n]"""
    b, m, n, k = tuple(dace.symbol(k) for k in 'bmnk')

    @dace.program
    def bmm(A: dtype[b, m, k], B: dtype[b, k, n], C: dtype[b, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        b, m, n, k = 3, 32, 31, 30

        x = np.random.rand(b, m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(b, k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z, b=b, m=m, n=n, k=k)

        ref = x @ y

        assert np.allclose(ref, z)


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
def test_batchmm_broadcast_rhs(implementation: str, dtype):
    """Test 3D batched matmul with broadcast on RHS: [b, m, k] @ [k, n]"""
    b, m, n, k = tuple(dace.symbol(k) for k in 'bmnk')

    @dace.program
    def bmm_broadcast(A: dtype[b, m, k], B: dtype[k, n], C: dtype[b, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm_broadcast.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        b, m, n, k = 3, 16, 32, 64

        x = np.random.rand(b, m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z, b=b, m=m, n=n, k=k)

        ref = x @ y

        assert np.allclose(ref, z)


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
def test_batchmm_broadcast_lhs(implementation: str, dtype):
    """Test 3D batched matmul with broadcast on LHS: [m, k] @ [b, k, n]"""
    b, m, n, k = tuple(dace.symbol(k) for k in 'bmnk')

    @dace.program
    def bmm_broadcast(A: dtype[m, k], B: dtype[b, k, n], C: dtype[b, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm_broadcast.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        b, m, n, k = 3, 16, 32, 64

        x = np.random.rand(m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(b, k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z, b=b, m=m, n=n, k=k)

        ref = x @ y

        assert np.allclose(ref, z)


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
def test_batchmm_4d(implementation: str, dtype):
    """Test 4D batched matmul: [b1, b2, m, k] @ [b1, b2, k, n]"""
    b1, b2, m, n, k = 4, 2, 64, 128, 64

    @dace.program
    def bmm_4d(A: dtype[b1, b2, m, k], B: dtype[b1, b2, k, n], C: dtype[b1, b2, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm_4d.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        x = np.random.rand(b1, b2, m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(b1, b2, k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b1, b2, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z)

        ref = x @ y

        assert np.allclose(ref, z)


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
def test_batchmm_4d_broadcast_rhs(implementation: str, dtype):
    """Test 4D batched matmul with broadcast on RHS: [b1, b2, m, k] @ [k, n]"""
    b1, b2, m, n, k = 4, 2, 64, 128, 64

    @dace.program
    def bmm_4d_broadcast(A: dtype[b1, b2, m, k], B: dtype[k, n], C: dtype[b1, b2, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm_4d_broadcast.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        x = np.random.rand(b1, b2, m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b1, b2, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z)

        ref = x @ y

        assert np.allclose(ref, z)


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
def test_batchmm_4d_broadcast_lhs(implementation: str, dtype):
    """Test 4D batched matmul with broadcast on LHS: [m, k] @ [b1, b2, k, n]"""
    b1, b2, m, n, k = 4, 2, 64, 128, 64

    @dace.program
    def bmm_4d_broadcast(A: dtype[m, k], B: dtype[b1, b2, k, n], C: dtype[b1, b2, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm_4d_broadcast.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        x = np.random.rand(m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(b1, b2, k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b1, b2, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z)

        ref = x @ y

        assert np.allclose(ref, z)


if __name__ == "__main__":
    test_batchmm("pure", dace.float32)
    test_batchmm("pure", dace.float64)
    test_batchmm("MKL", dace.float32)
    test_batchmm("MKL", dace.float64)
    test_batchmm("cuBLAS", dace.float32)
    test_batchmm("cuBLAS", dace.float64)
    test_batchmm_broadcast_rhs("pure", dace.float32)
    test_batchmm_broadcast_rhs("pure", dace.float64)
    test_batchmm_broadcast_rhs("MKL", dace.float32)
    test_batchmm_broadcast_rhs("MKL", dace.float64)
    test_batchmm_broadcast_rhs("cuBLAS", dace.float32)
    test_batchmm_broadcast_rhs("cuBLAS", dace.float64)
    test_batchmm_broadcast_lhs("pure", dace.float32)
    test_batchmm_broadcast_lhs("pure", dace.float64)
    test_batchmm_broadcast_lhs("MKL", dace.float32)
    test_batchmm_broadcast_lhs("MKL", dace.float64)
    test_batchmm_broadcast_lhs("cuBLAS", dace.float32)
    test_batchmm_broadcast_lhs("cuBLAS", dace.float64)
    test_batchmm_4d("pure", dace.float32)
    test_batchmm_4d("pure", dace.float64)
    test_batchmm_4d("MKL", dace.float32)
    test_batchmm_4d("MKL", dace.float64)
    test_batchmm_4d("cuBLAS", dace.float32)
    test_batchmm_4d("cuBLAS", dace.float64)
    test_batchmm_4d_broadcast_rhs("pure", dace.float32)
    test_batchmm_4d_broadcast_rhs("pure", dace.float64)
    test_batchmm_4d_broadcast_rhs("MKL", dace.float32)
    test_batchmm_4d_broadcast_rhs("MKL", dace.float64)
    test_batchmm_4d_broadcast_rhs("cuBLAS", dace.float32)
    test_batchmm_4d_broadcast_rhs("cuBLAS", dace.float64)
    test_batchmm_4d_broadcast_lhs("pure", dace.float32)
    test_batchmm_4d_broadcast_lhs("pure", dace.float64)
    test_batchmm_4d_broadcast_lhs("MKL", dace.float32)
    test_batchmm_4d_broadcast_lhs("MKL", dace.float64)
    test_batchmm_4d_broadcast_lhs("cuBLAS", dace.float32)
    test_batchmm_4d_broadcast_lhs("cuBLAS", dace.float64)
