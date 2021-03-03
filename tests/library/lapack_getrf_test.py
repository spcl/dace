# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
import dace.libraries.lapack as lapack
import numpy as np
import sys
import warnings
import pytest

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("matrix_lufact_getrf_{}_{}".format(implementation, str(dtype)))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x" + suffix, [n,n],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("pivots" + suffix, [n],
                   dace.dtypes.int32,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("result" + suffix, [1],
                   dace.dtypes.int32,
                   storage=storage,
                   transient=transient)

    xin = state.add_access("x" + suffix)
    xout = state.add_access("x" + suffix)
    pivots = state.add_access("pivots" + suffix)
    result = state.add_access("result" + suffix)

    getrf_node = lapack.nodes.getrf.Getrf("getrf")
    getrf_node.implementation = implementation

    state.add_memlet_path(xin,
                          getrf_node,
                          dst_conn="_xin",
                          memlet=Memlet.simple(xin, "0:n, 0:n", num_accesses=n*n))
    state.add_memlet_path(getrf_node,
                          result,
                          src_conn="_res",
                          memlet=Memlet.simple(result, "0", num_accesses=1))
    state.add_memlet_path(getrf_node,
                          pivots,
                          src_conn="_ipiv",
                          memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(getrf_node,
                          xout,
                          src_conn="_xout",
                          memlet=Memlet.simple(xout, "0:n, 0:n", num_accesses=n*n))



    return sdfg


###############################################################################

@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl)
])

def test_getrf(implementation, dtype):
    sdfg = make_sdfg(implementation, dtype)
    getrf_sdfg = sdfg.compile()
    np_dtype = getattr(np, dtype.to_string())
    
    from scipy.linalg import lu_factor
    size = 4
    lapack_status = np.array([-1], dtype=np.int32)
    A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype=np_dtype)
    lu_ref, _ = lu_factor(A)
    pivots = np.ndarray([0,0,0,0], dtype=np.int32)
  
    # the x is input AND output, the "result" argument gives the lapack status!
    getrf_sdfg(x=A, result=lapack_status, pivots=pivots, n=size)

    if np.allclose(A, lu_ref):
        print("Test ran successfully for {}.".format(implementation))
    else:
        raise ValueError("Validation error!")

###############################################################################

if __name__ == "__main__":
    test_getrf("MKL", dace.float32)
    test_getrf("MKL", dace.float64)
###############################################################################
