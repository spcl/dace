# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
from dace.libraries.linalg import Inv
import numpy as np
import warnings
import pytest

n = dace.symbol("n", dace.int64)
id = -1


def generate_matrix(size, dtype):
    if dtype == np.float32:
        tol = 1e-7
    elif dtype == np.float64:
        tol = 1e-14
    else:
        raise NotImplementedError
    from numpy.random import default_rng
    rng = default_rng(42)
    while True:
        A = rng.random((size, size), dtype=dtype)
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


@pytest.mark.parametrize("implementation, dtype, size, shape, overwrite, getri", [
    pytest.param(
        'MKL', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, True, marks=pytest.mark.mkl),
    pytest.param(
        'MKL', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, True, marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 True,
                 marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 True,
                 marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 True,
                 marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 True,
                 marks=pytest.mark.mkl),
    pytest.param(
        'MKL', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, False, marks=pytest.mark.mkl),
    pytest.param(
        'MKL', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, False, marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 False,
                 marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 False,
                 marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 False,
                 marks=pytest.mark.mkl),
    pytest.param('MKL',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 False,
                 marks=pytest.mark.mkl),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]],
                 False,
                 True,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]],
                 False,
                 True,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 True,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 True,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 True,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 True,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]],
                 False,
                 False,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]],
                 False,
                 False,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 False,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 False,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 False,
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 False,
                 marks=pytest.mark.lapack),
    pytest.param('cuSolverDn',
                 np.float32,
                 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]],
                 False,
                 False,
                 marks=pytest.mark.gpu),
    pytest.param('cuSolverDn',
                 np.float64,
                 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]],
                 False,
                 False,
                 marks=pytest.mark.gpu),
    pytest.param('cuSolverDn',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 False,
                 marks=pytest.mark.gpu),
    pytest.param('cuSolverDn',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 False,
                 False,
                 marks=pytest.mark.gpu),
    pytest.param('cuSolverDn',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 False,
                 marks=pytest.mark.gpu),
    pytest.param('cuSolverDn',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 True,
                 False,
                 marks=pytest.mark.gpu)
])
def test_inv(implementation, dtype, size, shape, overwrite, getri):
    global id
    id += 1

    in_shape = shape[0]
    out_shape = shape[1]
    in_offset = shape[2]
    out_offset = shape[3]
    in_dims = shape[4]
    out_dims = shape[5]

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
    if implementation == 'cuSolverDn':
        sdfg.apply_gpu_transformations()
        sdfg.simplify()
    try:
        inv_sdfg = sdfg.compile()
    except (CompilerConfigurationError, CompilationError):
        warnings.warn('Configuration/compilation failed, library missing or '
                      'misconfigured, skipping test for {}.'.format(implementation))
        return

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


###############################################################################

if __name__ == "__main__":
    test_inv('MKL', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, True)
    test_inv('MKL', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, True)
    test_inv('MKL', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], False, True)
    test_inv('MKL', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], False, True)
    test_inv('MKL', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], True, True)
    test_inv('MKL', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], True, True)
    test_inv('MKL', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, False)
    test_inv('MKL', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, False)
    test_inv('MKL', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], False, False)
    test_inv('MKL', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], False, False)
    test_inv('MKL', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], True, False)
    test_inv('MKL', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], True, False)
    test_inv('cuSolverDn', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, False)
    test_inv('cuSolverDn', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], False, False)
    test_inv('cuSolverDn', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], False, False)
    test_inv('cuSolverDn', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], False, False)
    test_inv('cuSolverDn', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], True, False)
    test_inv('cuSolverDn', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], True, False)
