# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np

import dace
import dace.libraries.blas as blas

from dace.library import change_default
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul


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


@pytest.mark.mkl
@pytest.mark.parametrize("dtype, rtol", [(dace.complex64, 1e-5), (dace.complex128, 1e-12)])
def test_batchmm_complex_mkl(dtype, rtol: float):
    """MKL's ``?gemm_batch`` takes MKL_Complex8/16 coefficients and operands by value, not dace::complex."""
    b, m, n, k = tuple(dace.symbol(k) for k in 'bmnk')

    @dace.program
    def bmm_complex(A: dtype[b, m, k], B: dtype[b, k, n], C: dtype[b, m, n]):
        C[:] = A @ B

    sdfg = bmm_complex.to_sdfg()
    sdfg.name = f'bmm_complex_mkl_{dtype.to_string()}'
    with change_default(blas, 'MKL'):
        sdfg.expand_library_nodes()

    rng = np.random.default_rng(0)
    npdtype = dtype.as_numpy_dtype()
    x = (rng.standard_normal((3, 32, 30)) + 1j * rng.standard_normal((3, 32, 30))).astype(npdtype)
    y = (rng.standard_normal((3, 30, 31)) + 1j * rng.standard_normal((3, 30, 31))).astype(npdtype)
    z = np.zeros((3, 32, 31), dtype=npdtype)
    sdfg(A=x, B=y, C=z, b=3, m=32, n=31, k=30)

    ref = x @ y
    np.testing.assert_allclose(z, ref, rtol=rtol, atol=rtol * np.abs(ref).max())


def create_bmm_sdfg(dtype, A_shape, B_shape, C_shape, alpha, beta, implementation, sdfg_name):
    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    B, B_arr = sdfg.add_array("B", B_shape, dtype)
    C, C_arr = sdfg.add_array("C", C_shape, dtype)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_write("C")

    libnode = BatchedMatMul('_BatchedMatMul_')
    libnode.alpha = alpha
    libnode.beta = beta
    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_b', dace.Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_c', wC, None, dace.Memlet.from_array(C, C_arr))

    return sdfg


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float32),
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.float32, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack),
])
def test_batchmm_alpha(implementation: str, dtype):
    """alpha != 1 must scale A @ B: every CPU expansion used to hard-code alpha=1 and drop node.alpha."""
    b, m, n, k = 3, 32, 31, 30
    alpha = 2.5

    sdfg = create_bmm_sdfg(dtype, [b, m, k], [b, k, n], [b, m, n], alpha, 0.0, implementation,
                           f"bmm_alpha_{implementation}_{dtype.to_string()}")

    rng = np.random.default_rng(0)
    x = rng.random((b, m, k)).astype(dtype.as_numpy_dtype())
    y = rng.random((b, k, n)).astype(dtype.as_numpy_dtype())
    z = np.zeros((b, m, n), dtype=dtype.as_numpy_dtype())

    sdfg(A=x, B=y, C=z)

    ref = alpha * (x @ y)
    np.testing.assert_allclose(z, ref, rtol=1e-5)


@pytest.mark.parametrize("implementation, dtype, rtol", [
    pytest.param("pure", dace.complex64, 1e-5),
    pytest.param("pure", dace.complex128, 1e-12),
    pytest.param("MKL", dace.complex64, 1e-5, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.complex128, 1e-12, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.complex64, 1e-5, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.complex128, 1e-12, marks=pytest.mark.lapack),
])
def test_batchmm_alpha_complex(implementation: str, dtype, rtol: float):
    """A complex alpha != 1 must scale A @ B; MKL additionally routes it through MKL_Complex8/16."""
    b, m, n, k = 3, 32, 30, 31
    alpha = complex(1.5, -0.5)

    sdfg = create_bmm_sdfg(dtype, [b, m, k], [b, k, n], [b, m, n], alpha, 0.0, implementation,
                           f"bmm_alpha_complex_{implementation}_{dtype.to_string()}")

    rng = np.random.default_rng(0)
    npdtype = dtype.as_numpy_dtype()
    x = (rng.standard_normal((b, m, k)) + 1j * rng.standard_normal((b, m, k))).astype(npdtype)
    y = (rng.standard_normal((b, k, n)) + 1j * rng.standard_normal((b, k, n))).astype(npdtype)
    z = np.zeros((b, m, n), dtype=npdtype)

    sdfg(A=x, B=y, C=z)

    ref = alpha * (x @ y)
    np.testing.assert_allclose(z, ref, rtol=rtol, atol=rtol * np.abs(ref).max())


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack),
    pytest.param("rocBLAS", dace.float64, marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("beta", [0.0, 1.0, 0.5])
def test_batchmm_beta(implementation: str, dtype, beta: float):
    """beta scales the PRIOR value of C in place: BatchedMatMul carries no _cin connector (see
    BatchedMatMul.__init__), so every expansion reads and writes C through its sole "_c" connector,
    mirroring Gemm(cin=False). Result must equal alpha * A @ B + beta * C0 for beta in {0, 1, 0.5}."""
    b, m, n, k = 2, 4, 5, 3
    alpha = 1.5

    beta_tag = str(beta).replace('.', 'p')
    sdfg = create_bmm_sdfg(dtype, [b, m, k], [b, k, n], [b, m, n], alpha, beta, implementation,
                           f"bmm_beta_{implementation}_{dtype.to_string()}_{beta_tag}")

    rng = np.random.default_rng(0)
    x = rng.random((b, m, k)).astype(dtype.as_numpy_dtype())
    y = rng.random((b, k, n)).astype(dtype.as_numpy_dtype())
    c0 = rng.random((b, m, n)).astype(dtype.as_numpy_dtype())
    z = c0.copy()

    sdfg(A=x, B=y, C=z)

    ref = alpha * (x @ y) + beta * c0
    np.testing.assert_allclose(z, ref, rtol=1e-5)


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
