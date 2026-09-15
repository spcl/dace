# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.libraries.linalg.nodes.cholesky import GPU_SOLVERS
from dace import Memlet
from dace.libraries.linalg import Cholesky
import numpy as np
import pytest


def generate_matrix(size, dtype):
    from numpy.random import default_rng
    rng = default_rng(42)
    A = rng.random((size, size), dtype=dtype)
    return (0.5 * A @ A.T).copy()


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n", dace.int64)

    sdfg = dace.SDFG("linalg_cholesky_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    inp = sdfg.add_array("xin", [n, n], dtype)
    out = sdfg.add_array("xout", [n, n], dtype)

    xin = state.add_read("xin")
    xout = state.add_write("xout")

    chlsky_node = Cholesky("cholesky", lower=True)
    chlsky_node.implementation = implementation

    state.add_memlet_path(xin, chlsky_node, dst_conn="_a", memlet=Memlet.from_array(*inp))
    state.add_memlet_path(chlsky_node, xout, src_conn="_b", memlet=Memlet.from_array(*out))

    return sdfg


def test_cholesky_pure_refuses_a_slice_of_a_higher_rank_array():
    """A connector that is a slice of a bigger array is refused, not read as if it were contiguous.

    ``Cholesky.validate`` returns the caller's whole descriptor alongside a SQUEEZED shape, so the
    slice's own strides are not recoverable in the expansion. No implementation supports this (the
    vendor path fails with a bare ``TypeError`` about stride length); the pure one at least says
    which connector and why.
    """
    size = 5
    sdfg = dace.SDFG('linalg_cholesky_pure_sliced')
    sdfg.add_array('xin', [2, size, size], dace.float64)
    sdfg.add_array('xout', [2, size, size], dace.float64)
    state = sdfg.add_state('dataflow')
    node = Cholesky('cholesky', lower=True)
    node.implementation = 'pure'
    subset = '1, 0:%d, 0:%d' % (size, size)
    state.add_memlet_path(state.add_read('xin'), node, dst_conn='_a', memlet=Memlet.simple('xin', subset))
    state.add_memlet_path(node, state.add_write('xout'), src_conn='_b', memlet=Memlet.simple('xout', subset))

    with pytest.raises(NotImplementedError, match='rank-2 slice of a rank-3'):
        sdfg.expand_library_nodes()


@pytest.mark.parametrize("lower", [True, False])
def test_cholesky_pure_conjugates_a_complex_hermitian_factor(lower):
    """A complex Hermitian matrix factors as ``L L^H`` (upper ``U = L^H``); the pure expansion multiplied
    ``L[i,k] * L[j,k]`` without the conjugate, which left an imaginary diagonal and broke cegterg's
    generalized eigen-solve."""
    size = 4
    sdfg = dace.SDFG(f"linalg_cholesky_pure_complex_{'lower' if lower else 'upper'}")
    sdfg.add_array("xin", [size, size], dace.complex128)
    sdfg.add_array("xout", [size, size], dace.complex128)
    state = sdfg.add_state("dataflow")
    node = Cholesky("cholesky", lower=lower)
    node.implementation = "pure"
    state.add_memlet_path(state.add_read("xin"), node, dst_conn="_a", memlet=Memlet.simple("xin", "0:4, 0:4"))
    state.add_memlet_path(node, state.add_write("xout"), src_conn="_b", memlet=Memlet.simple("xout", "0:4, 0:4"))

    rng = np.random.default_rng(5)
    X = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    A = X @ X.conj().T + size * np.eye(size)
    B = np.zeros((size, size), dtype=np.complex128)
    sdfg(xin=A.copy(), xout=B)

    L = np.linalg.cholesky(A)
    np.testing.assert_allclose(B, L if lower else L.conj().T, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.imag(np.diag(B)), 0.0)


@pytest.mark.parametrize("implementation, dtype, storage", [
    pytest.param("pure", dace.float32, dace.StorageType.Default),
    pytest.param("pure", dace.float64, dace.StorageType.Default),
    pytest.param("MKL", dace.float32, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.float32, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("cuSolverDn", dace.float32, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("cuSolverDn", dace.float64, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("rocSOLVER", dace.float32, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("rocSOLVER", dace.float64, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
])
def test_cholesky(implementation, dtype, storage):
    sdfg = make_sdfg(implementation, dtype, storage)
    if implementation in GPU_SOLVERS:
        sdfg.apply_gpu_transformations()
        sdfg.simplify()
    np_dtype = getattr(np, dtype.to_string())
    cholesky_sdfg = sdfg.compile()

    size = 4
    A = generate_matrix(size, np_dtype)
    B = np.zeros([size, size], dtype=np_dtype)
    cholesky_ref = np.linalg.cholesky(A)

    # the x is input AND output, the "result" argument gives the lapack status!
    cholesky_sdfg(xin=A, xout=B, n=size)

    if dtype == dace.float32:
        rtol = 1e-6
    elif dtype == dace.float64:
        rtol = 1e-12
    else:
        raise NotImplementedError
    assert (np.linalg.norm(cholesky_ref - B) / np.linalg.norm(cholesky_ref)) < rtol


###############################################################################

if __name__ == "__main__":
    test_cholesky("MKL", dace.float32, dace.StorageType.Default)
    test_cholesky("MKL", dace.float64, dace.StorageType.Default)
    test_cholesky("cuSolverDn", dace.float32, dace.StorageType.GPU_Global)
    test_cholesky("cuSolverDn", dace.float64, dace.StorageType.GPU_Global)
