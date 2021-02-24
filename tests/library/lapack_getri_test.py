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

    sdfg = dace.SDFG("matrix_inv_getrf_getri_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x" + suffix, [n,n],
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
    sdfg.add_array("result_getri" + suffix, [1],
                   dace.dtypes.int32,
                   storage=storage,
                   transient=False)

    xin = state.add_access("x" + suffix)
    xout_getrf = state.add_access("x" + suffix)
    xout_getri = state.add_access("x" + suffix)
    pivots = state.add_access("pivots" + suffix)
    res_getrf = state.add_access("result_getrf" + suffix)
    res_getri = state.add_access("result_getri" + suffix)

    getrf_node = lapack.nodes.getrf.Getrf("getrf")
    getrf_node.implementation = implementation
    getri_node = lapack.nodes.getri.Getri("getri")
    getri_node.implementation = implementation


    state.add_memlet_path(xin,
                          getrf_node,
                          dst_conn="_xin",
                          memlet=Memlet.simple(xin, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(getrf_node,
                          res_getrf,
                          src_conn="_res",
                          memlet=Memlet.simple(res_getrf, "0", num_accesses=1))
    state.add_memlet_path(getri_node,
                          res_getri,
                          src_conn="_res",
                          memlet=Memlet.simple(res_getri, "0", num_accesses=1))
    state.add_memlet_path(getrf_node,
                          pivots,
                          src_conn="_ipiv",
                          memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(pivots,
                          getri_node,
                          dst_conn="_ipiv",
                          memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(getrf_node,
                          xout_getrf,
                          src_conn="_xout",
                          memlet=Memlet.simple(xout_getrf, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(xout_getrf,
                          getri_node,
                          dst_conn="_xin",
                          memlet=Memlet.simple(xout_getrf, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(getri_node,
                          xout_getri,
                          src_conn="_xout",
                          memlet=Memlet.simple(xout_getri, "0:n, 0:n", num_accesses=n*n))

    return sdfg


###############################################################################


def _test_getri(implementation, dtype, sdfg):
    inv_sdfg = sdfg.compile()
    
    size = 4
    lapack_status1 = np.array([-1], dtype=np.int32)
    lapack_status2 = np.array([-1], dtype=np.int32)
    A1 = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype=dtype)
    A2 = np.copy(A1)
    A3 = np.linalg.inv(A2)
  
    # the x is input AND output, the "result" argument gives the lapack status!
    inv_sdfg(x=A1, result_getrf=lapack_status1, result_getri=lapack_status2, pivots=np.ndarray([0,0,0,0], dtype=np.int32), n=size)

    if np.allclose(A1, A3):
        print("Test ran successfully for {}.".format(implementation))
    else:
        print(A1-A3)
        raise ValueError("Validation error!")


def test_getri():
    _test_getri("32-bit MKL", np.float32, make_sdfg("MKL", dace.float32))
    _test_getri("64-bit MKL", np.float64, make_sdfg("MKL", dace.float64))

###############################################################################

if __name__ == "__main__":
    test_getri()
###############################################################################
