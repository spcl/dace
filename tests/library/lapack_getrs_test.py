# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.libraries.lapack as lapack
import dace.libraries.standard as std
import numpy as np
import pytest

from dace.memlet import Memlet

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("matrix_solve_getrf_getrs_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    Ahost_arr = sdfg.add_array("A", [n, n], dtype, storage=dace.StorageType.Default)
    Bhost_arr = sdfg.add_array("B", [n], dtype, storage=dace.StorageType.Default)

    if transient:
        A_arr = sdfg.add_array("A" + suffix, [n, n], dtype, storage=storage, transient=transient)
        AT_arr = sdfg.add_array('AT' + suffix, [n, n], dtype, storage=storage, transient=transient)
        B_arr = sdfg.add_array("B" + suffix, [n], dtype, storage=storage, transient=transient)
    sdfg.add_array("pivots" + suffix, [n], dace.dtypes.int32, storage=storage, transient=transient)
    sdfg.add_array("result_getrf" + suffix, [1], dace.dtypes.int32, storage=storage, transient=transient)
    sdfg.add_array("result_getrs" + suffix, [1], dace.dtypes.int32, storage=storage, transient=transient)

    if transient:
        Ahi = state.add_read("A")
        Ai = state.add_access("A" + suffix)
        Ain = state.add_access("AT" + suffix)
        Aout = state.add_access("AT" + suffix)
        Bhi = state.add_read("B")
        Bho = state.add_read("B")
        Bin = state.add_access("B" + suffix)
        Bout = state.add_access("B" + suffix)
        transpose_in = std.Transpose("transpose_in", dtype=dtype)
        transpose_in.implementation = "cuBLAS"
        state.add_nedge(Ahi, Ai, Memlet.from_array(*Ahost_arr))
        state.add_nedge(Bhi, Bin, Memlet.from_array(*Bhost_arr))
        state.add_nedge(Bout, Bho, Memlet.from_array(*Bhost_arr))
        state.add_memlet_path(Ai, transpose_in, dst_conn='_inp', memlet=Memlet.from_array(*A_arr))
        state.add_memlet_path(transpose_in, Ain, src_conn='_out', memlet=Memlet.from_array(*AT_arr))
    else:
        Ain = state.add_access("A" + suffix)
        Aout = state.add_access("A" + suffix)
        Bin = state.add_access("B" + suffix)
        Bout = state.add_access("B" + suffix)
    pivots = state.add_access("pivots" + suffix)
    res_getrf = state.add_access("result_getrf" + suffix)
    res_getrs = state.add_access("result_getrs" + suffix)

    getrf_node = lapack.Getrf("getrf")
    getrf_node.implementation = implementation
    getrs_node = lapack.Getrs("getrs")
    getrs_node.implementation = implementation

    state.add_memlet_path(Ain, getrf_node, dst_conn="_xin", memlet=Memlet.simple(Ain, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(getrf_node, res_getrf, src_conn="_res", memlet=Memlet.simple(res_getrf, "0", num_accesses=1))
    state.add_memlet_path(getrs_node, res_getrs, src_conn="_res", memlet=Memlet.simple(res_getrs, "0", num_accesses=1))
    state.add_memlet_path(getrf_node, pivots, src_conn="_ipiv", memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(pivots, getrs_node, dst_conn="_ipiv", memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(getrf_node,
                          Aout,
                          src_conn="_xout",
                          memlet=Memlet.simple(Aout, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(Aout, getrs_node, dst_conn="_a", memlet=Memlet.simple(Aout, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(Bin, getrs_node, dst_conn="_rhs_in", memlet=Memlet.simple(Bin, "0:n", num_accesses=n))
    state.add_memlet_path(getrs_node, Bout, src_conn="_rhs_out", memlet=Memlet.simple(Bout, "0:n", num_accesses=n))

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
def test_getrs(implementation, dtype, storage):
    sdfg = make_sdfg(implementation, dtype, storage)
    solve_sdfg = sdfg.compile()
    np_dtype = getattr(np, dtype.to_string())

    # this is what we are trying to do, using getrf (LU factorize the matrix a) and getrs (solve the system for b as rhs)
    a1 = np.array([[1, 2], [3, 5]], dtype=np_dtype)
    b1 = np.array([1, 2], dtype=np_dtype)
    x = np.linalg.solve(a1, b1)

    # verify if it works in numpy :)
    if not np.allclose(np.dot(a1, x), b1):
        raise ValueError("NumPy solve returned wrong result")

    lapack_status1 = np.array([-1], dtype=np.int32)
    lapack_status2 = np.array([-1], dtype=np.int32)
    a2 = np.copy(a1)  # a input will be overwritten by its lu factorization (by getrf)
    b2 = np.copy(b1)  # rhs input will be overwritten by the solution (by getrs)
    solve_sdfg(A=a2,
               B=b2,
               result_getrf=lapack_status1,
               result_getrs=lapack_status2,
               pivots=np.ndarray([0, 0], dtype=np.int32),
               n=2)

    if np.allclose(np.dot(a1, b2), b1):
        print("Test ran successfully for {}.".format(implementation))
    else:
        print(b2)
        print(np.dot(a1, b2) - b1)
        raise ValueError("Validation error!")


###############################################################################

if __name__ == "__main__":
    test_getrs("MKL", dace.float32)
    test_getrs("MKL", dace.float64)
    test_getrs("cuSolverDn", dace.float32, dace.StorageType.GPU_Global)
    test_getrs("cuSolverDn", dace.float64, dace.StorageType.GPU_Global)
###############################################################################
