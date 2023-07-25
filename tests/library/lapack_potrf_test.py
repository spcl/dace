# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.libraries.lapack as lapack
import dace.libraries.standard as std
import numpy as np
import pytest

from dace.memlet import Memlet

###############################################################################


def generate_matrix(size, dtype):
    from numpy.random import default_rng
    rng = default_rng(42)
    A = rng.random((size, size), dtype=dtype)
    return (0.5 * A @ A.T).copy()


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("matrix_choleskyfact_potrf_{}_{}".format(implementation, str(dtype)))
    state = sdfg.add_state("dataflow")

    xhost_arr = sdfg.add_array("x", [n, n], dtype, storage=dace.StorageType.Default)

    if transient:
        x_arr = sdfg.add_array("x" + suffix, [n, n], dtype, storage=storage, transient=transient)
        xt_arr = sdfg.add_array('xt' + suffix, [n, n], dtype, storage=storage, transient=transient)
    sdfg.add_array("result" + suffix, [1], dace.dtypes.int32, storage=storage, transient=transient)

    if transient:
        xhi = state.add_read("x")
        xho = state.add_write("x")
        xi = state.add_access("x" + suffix)
        xo = state.add_access("x" + suffix)
        xin = state.add_access("xt" + suffix)
        xout = state.add_access("xt" + suffix)
        transpose_in = std.Transpose("transpose_in", dtype=dtype)
        transpose_in.implementation = "cuBLAS"
        transpose_out = std.Transpose("transpose_out", dtype=dtype)
        transpose_out.implementation = "cuBLAS"
        state.add_nedge(xhi, xi, Memlet.from_array(*xhost_arr))
        state.add_nedge(xo, xho, Memlet.from_array(*xhost_arr))
        state.add_memlet_path(xi, transpose_in, dst_conn='_inp', memlet=Memlet.from_array(*x_arr))
        state.add_memlet_path(transpose_in, xin, src_conn='_out', memlet=Memlet.from_array(*xt_arr))
        state.add_memlet_path(xout, transpose_out, dst_conn='_inp', memlet=Memlet.from_array(*xt_arr))
        state.add_memlet_path(transpose_out, xo, src_conn='_out', memlet=Memlet.from_array(*x_arr))
    else:
        xin = state.add_access("x" + suffix)
        xout = state.add_access("x" + suffix)
    result = state.add_access("result" + suffix)

    potrf_node = lapack.Potrf("potrf")
    potrf_node.implementation = implementation

    state.add_memlet_path(xin, potrf_node, dst_conn="_xin", memlet=Memlet.simple(xin, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(potrf_node, result, src_conn="_res", memlet=Memlet.simple(result, "0", num_accesses=1))
    state.add_memlet_path(potrf_node,
                          xout,
                          src_conn="_xout",
                          memlet=Memlet.simple(xout, "0:n, 0:n", num_accesses=n * n))

    return sdfg


###############################################################################


@pytest.mark.parametrize("implementation, dtype, storage", [
    pytest.param("MKL", dace.float32, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.float32, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("cuSolverDn", dace.float32, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("cuSolverDn", dace.float64, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
])
def test_potrf(implementation, dtype, storage):
    sdfg = make_sdfg(implementation, dtype, storage)
    potrf_sdfg = sdfg.compile()
    np_dtype = getattr(np, dtype.to_string())

    size = 4
    lapack_status = np.array([-1], dtype=np.int32)
    A = generate_matrix(4, np_dtype)
    cholesky_ref = np.linalg.cholesky(A)

    # the x is input AND output, the "result" argument gives the lapack status!
    potrf_sdfg(x=A, result=lapack_status, n=size)

    if dtype == dace.float32:
        rtol = 1e-6
    elif dtype == dace.float64:
        rtol = 1e-12
    else:
        raise NotImplementedError
    assert (np.linalg.norm(cholesky_ref - np.tril(A)) / np.linalg.norm(cholesky_ref)) < rtol


###############################################################################

if __name__ == "__main__":
    test_potrf("MKL", dace.float32, dace.StorageType.Default)
    test_potrf("MKL", dace.float64, dace.StorageType.Default)
    test_potrf("cuSolverDn", dace.float32, dace.StorageType.GPU_Global)
    test_potrf("cuSolverDn", dace.float64, dace.StorageType.GPU_Global)
###############################################################################
