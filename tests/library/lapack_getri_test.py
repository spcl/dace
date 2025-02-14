# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.lapack as lapack
import numpy as np
import pytest

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    sdfg = dace.SDFG("matrix_inv_getrf_getri_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x", [n, n], dtype, storage=storage, transient=False)
    sdfg.add_array("pivots", [n], dace.dtypes.int32, storage=storage, transient=True)
    sdfg.add_array("result_getrf", [1], dace.dtypes.int32, storage=storage, transient=False)
    sdfg.add_array("result_getri", [1], dace.dtypes.int32, storage=storage, transient=False)

    xin = state.add_access("x")
    xout_getrf = state.add_access("x")
    xout_getri = state.add_access("x")
    pivots = state.add_access("pivots")
    res_getrf = state.add_access("result_getrf")
    res_getri = state.add_access("result_getri")

    getrf_node = lapack.nodes.getrf.Getrf("getrf")
    getrf_node.implementation = implementation
    getri_node = lapack.nodes.getri.Getri("getri")
    getri_node.implementation = implementation

    state.add_memlet_path(xin, getrf_node, dst_conn="_xin", memlet=Memlet.simple(xin, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(getrf_node, res_getrf, src_conn="_res", memlet=Memlet.simple(res_getrf, "0", num_accesses=1))
    state.add_memlet_path(getri_node, res_getri, src_conn="_res", memlet=Memlet.simple(res_getri, "0", num_accesses=1))
    state.add_memlet_path(getrf_node, pivots, src_conn="_ipiv", memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(pivots, getri_node, dst_conn="_ipiv", memlet=Memlet.simple(pivots, "0:n", num_accesses=n))
    state.add_memlet_path(getrf_node,
                          xout_getrf,
                          src_conn="_xout",
                          memlet=Memlet.simple(xout_getrf, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(xout_getrf,
                          getri_node,
                          dst_conn="_xin",
                          memlet=Memlet.simple(xout_getrf, "0:n, 0:n", num_accesses=n * n))
    state.add_memlet_path(getri_node,
                          xout_getri,
                          src_conn="_xout",
                          memlet=Memlet.simple(xout_getri, "0:n, 0:n", num_accesses=n * n))

    return sdfg


###############################################################################


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.float32, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, marks=pytest.mark.lapack)
])
def test_getri(implementation, dtype):
    sdfg = make_sdfg(implementation, dtype)
    inv_sdfg = sdfg.compile()
    np_dtype = getattr(np, dtype.to_string())

    size = 4
    lapack_status1 = np.array([-1], dtype=np.int32)
    lapack_status2 = np.array([-1], dtype=np.int32)
    A1 = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype=np_dtype)
    A2 = np.copy(A1)
    A3 = np.linalg.inv(A2)
    pivots = np.ndarray([0, 0, 0, 0], dtype=np.int32)

    # the x is input AND output, the "result" argument gives the lapack status!
    inv_sdfg(x=A1, result_getrf=lapack_status1, result_getri=lapack_status2, pivots=pivots, n=size)

    if np.allclose(A1, A3):
        print("Test ran successfully for {}.".format(implementation))
    else:
        print(A1 - A3)
        raise ValueError("Validation error!")


###############################################################################

if __name__ == "__main__":
    test_getri("MKL", dace.float32)
    test_getri("MKL", dace.float64)

###############################################################################
