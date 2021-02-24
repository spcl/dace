# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
import dace.libraries.lapack as lapack
import numpy as np
import sys
import warnings

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("matrix_solve_getrf_getrs_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("A" + suffix, [n,n],
                   dtype,
                   storage=storage,
                   transient=False)
    sdfg.add_array("B" + suffix, [n],
                   dtype,
                   storage=storage,
                   transient=False)
    sdfg.add_array("pivots" + suffix, [n],
                   dace.dtypes.int32,
                   storage=storage,
                   transient=True)
    sdfg.add_array("result_getrf" + suffix, [1],
                   dace.dtypes.int32,
                   storage=storage,
                   transient=False)
    sdfg.add_array("result_getrs" + suffix, [1],
                   dace.dtypes.int32,
                   storage=storage,
                   transient=False)
    
    ain = state.add_access("A" + suffix)
    aout_getrf = state.add_access("A" + suffix)
    rhs_in_getrs = state.add_access("B" + suffix)
    rhs_out_getrs = state.add_access("B" + suffix)
    pivots = state.add_access("pivots" + suffix)
    res_getrf = state.add_access("result_getrf" + suffix)
    res_getrs = state.add_access("result_getrs" + suffix)

    getrf_node = lapack.nodes.getrf.Getrf("getrf")
    getrf_node.implementation = implementation
    getrs_node = lapack.nodes.getrs.Getrs("getrs")
    getrs_node.implementation = implementation


    state.add_memlet_path(ain,
                          getrf_node,
                          dst_conn="_xin",
                          memlet=Memlet.simple(ain, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(getrf_node,
                          res_getrf,
                          src_conn="_res",
                          memlet=Memlet.simple(res_getrf, "0", num_accesses=1))
    state.add_memlet_path(getrs_node,
                          res_getrs,
                          src_conn="_res",
                          memlet=Memlet.simple(res_getrs, "0", num_accesses=1))
    state.add_memlet_path(getrf_node,
                          pivots,
                          src_conn="_ipiv",
                          memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(pivots,
                          getrs_node,
                          dst_conn="_ipiv",
                          memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(getrf_node,
                          aout_getrf,
                          src_conn="_xout",
                          memlet=Memlet.simple(aout_getrf, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(aout_getrf,
                          getrs_node,
                          dst_conn="_a",
                          memlet=Memlet.simple(aout_getrf, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(rhs_in_getrs,
                          getrs_node,
                          dst_conn="_rhs_in",
                          memlet=Memlet.simple(rhs_in_getrs, "0:n", num_accesses=n))
    state.add_memlet_path(getrs_node,
                          rhs_out_getrs,
                          src_conn="_rhs_out",
                          memlet=Memlet.simple(rhs_out_getrs, "0:n", num_accesses=n))

    return sdfg


###############################################################################


def _test_getrs(implementation, dtype, sdfg):
    solve_sdfg = sdfg.compile()
    
    # this is what we are trying to do, using getrf (LU factorize the matrix a) and getrs (solve the system for b as rhs)
    a1 = np.array([[1, 2], [3, 5]], dtype=dtype)
    b1 = np.array([1, 2], dtype=dtype)
    x = np.linalg.solve(a1, b1)

    # verify if it works in numpy :)
    if not np.allclose(np.dot(a1, x), b1):
        raise ValueError("NumPy solve returned wrong result o_O")
  
    lapack_status1 = np.array([-1], dtype=np.int32)
    lapack_status2 = np.array([-1], dtype=np.int32)
    a2 = np.copy(a1)  # a input will be overwritten by its lu factorization (by getrf) 
    b2 = np.copy(b1)  # rhs input will be overwritten by the solution (by getrs)
    solve_sdfg(A=a2, B=b2, result_getrf=lapack_status1, result_getrs=lapack_status2, pivots=np.ndarray([0,0], dtype=np.int32), n=2)

    if np.allclose(np.dot(a1, b2), b1):
        print("Test ran successfully for {}.".format(implementation))
    else:
        print(A1-A3)
        raise ValueError("Validation error!")


def test_getrs():
    _test_getrs("32-bit MKL", np.float32, make_sdfg("MKL", dace.float32))
    _test_getrs("64-bit MKL", np.float64, make_sdfg("MKL", dace.float64))

###############################################################################

if __name__ == "__main__":
    test_getrs()
###############################################################################
