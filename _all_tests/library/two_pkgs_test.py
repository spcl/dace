# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
from dace.libraries.linalg import Inv
import numpy as np
import warnings

n = dace.symbol("n", dace.int64)


def generate_matrix(size, dtype):
    if dtype == np.float32:
        tol = 1e-7
    elif dtype == np.float64:
        tol = 1e-14
    else:
        raise NotImplementedError
    while True:
        A = np.random.randn(size, size).astype(dtype)
        B = A @ A.T
        err = np.absolute(B @ np.linalg.inv(B) - np.eye(size))
        if np.all(err < tol):
            break
    return A


def make_sdfg(implementation,
              dtype,
              id=0,
              in_shape=[n, n],
              out_shape=[n, n],
              in_subset="0:n, 0:n",
              out_subset="0:n, 0:n",
              overwrite=False,
              getri=True):

    sdfg = dace.SDFG("linalg_inv_{}_{}_{}".format(implementation, dtype.__name__, id))
    sdfg.add_symbol("n", dace.int64)
    state = sdfg.add_state("dataflow")

    sdfg.add_array("xin", in_shape, dtype)
    if not overwrite:
        sdfg.add_array("xout", out_shape, dtype)

    xin = state.add_read("xin")
    if overwrite:
        xout = state.add_write("xin")
    else:
        xout = state.add_write("xout")

    inv_node = Inv("inv", overwrite_a=overwrite, use_getri=getri)
    inv_node.implementation = implementation

    state.add_memlet_path(xin, inv_node, dst_conn="_ain", memlet=Memlet.simple(xin, in_subset, num_accesses=n * n))
    state.add_memlet_path(inv_node, xout, src_conn="_aout", memlet=Memlet.simple(xout, out_subset, num_accesses=n * n))

    return sdfg


def _test_inv(implementation,
              dtype,
              id=0,
              size=4,
              in_shape=[4, 4],
              out_shape=[4, 4],
              in_offset=[0, 0],
              out_offset=[0, 0],
              in_dims=[0, 1],
              out_dims=[0, 1],
              overwrite=False,
              getri=True):

    assert np.all(np.array(in_shape)[in_dims] >= size)
    assert np.all(np.array(out_shape)[out_dims] >= size)
    assert np.all(np.array(in_offset) < size)
    assert np.all(np.array(out_offset) < size)
    assert np.all(np.array(in_offset)[in_dims] + size <= np.array(in_shape)[in_dims])
    assert np.all(np.array(out_offset)[out_dims] + size <= np.array(out_shape)[out_dims])

    in_subset = tuple([slice(o, o + size) if i in in_dims else o for i, o in enumerate(in_offset)])
    if overwrite:
        out_subset = in_subset
    else:
        out_subset = tuple([slice(o, o + size) if i in out_dims else o for i, o in enumerate(out_offset)])

    in_subset_str = ','.join(
        ["{b}:{e}".format(b=o, e=o + size) if i in in_dims else str(o) for i, o in enumerate(in_offset)])
    if overwrite:
        out_subset_str = in_subset_str
    else:
        out_subset_str = ','.join(
            ["{b}:{e}".format(b=o, e=o + size) if i in out_dims else str(o) for i, o in enumerate(out_offset)])

    sdfg = make_sdfg(implementation, dtype, id, in_shape, out_shape, in_subset_str, out_subset_str, overwrite, getri)
    inv_sdfg = sdfg.compile()
    A0 = np.zeros(in_shape, dtype=dtype)
    A0[in_subset] = generate_matrix(size, dtype)
    A1 = np.copy(A0)
    if overwrite:
        A2 = A1
    else:
        A2 = np.zeros(out_shape, dtype=dtype)
    A3 = np.linalg.inv(A0[in_subset])

    inv_sdfg(xin=A1, xout=A2, n=size)

    if dtype == np.float32:
        rtol = 1e-7
        atol = 1e-7
    elif dtype == np.float64:
        rtol = 1e-14
        atol = 1e-14
    else:
        raise NotImplementedError

    assert np.allclose(A2[out_subset], A3, rtol=rtol, atol=atol)
    if overwrite:
        assert not np.array_equal(A0, A1)


if __name__ == "__main__":
    _test_inv('OpenBLAS', np.float32, id=0)
