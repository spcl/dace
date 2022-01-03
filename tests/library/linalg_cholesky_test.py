# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
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

    state.add_memlet_path(xin,
                          chlsky_node,
                          dst_conn="_a",
                          memlet=Memlet.from_array(*inp))
    state.add_memlet_path(chlsky_node,
                          xout,
                          src_conn="_b",
                          memlet=Memlet.from_array(*out))

    return sdfg


@pytest.mark.parametrize("implementation, dtype, storage", [
    pytest.param(
        "MKL", dace.float32, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param(
        "MKL", dace.float64, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS",
                 dace.float32,
                 dace.StorageType.Default,
                 marks=pytest.mark.lapack),
    pytest.param("OpenBLAS",
                 dace.float64,
                 dace.StorageType.Default,
                 marks=pytest.mark.lapack),
    pytest.param("cuSolverDn",
                 dace.float32,
                 dace.StorageType.GPU_Global,
                 marks=pytest.mark.gpu),
    pytest.param("cuSolverDn",
                 dace.float64,
                 dace.StorageType.GPU_Global,
                 marks=pytest.mark.gpu),
])
def test_cholesky(implementation, dtype, storage):
    sdfg = make_sdfg(implementation, dtype, storage)
    if implementation == 'cuSolverDn':
        sdfg.apply_gpu_transformations()
        sdfg.coarsen_dataflow()
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
    assert (np.linalg.norm(cholesky_ref - B) /
            np.linalg.norm(cholesky_ref)) < rtol


###############################################################################

if __name__ == "__main__":
    test_cholesky("MKL", dace.float32, dace.StorageType.Default)
    test_cholesky("MKL", dace.float64, dace.StorageType.Default)
    test_cholesky("cuSolverDn", dace.float32, dace.StorageType.GPU_Global)
    test_cholesky("cuSolverDn", dace.float64, dace.StorageType.GPU_Global)
