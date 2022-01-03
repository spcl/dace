# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import Memlet
from dace.libraries.linalg import Solve
import numpy as np
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
              out_subset="0:n, 0:n"):

    sdfg = dace.SDFG("linalg_solve_{}_{}_{}".format(implementation, dtype.__name__, id))
    sdfg.add_symbol("n", dace.int64)
    state = sdfg.add_state("dataflow")

    sdfg.add_array("ain", in_shape, dtype)
    sdfg.add_array("bin", out_shape, dtype)
    sdfg.add_array("bout", out_shape, dtype)

    ain = state.add_read("ain")
    bin = state.add_read("bin")
    bout = state.add_write("bout")

    solve_node = Solve("solve")
    solve_node.implementation = implementation

    state.add_memlet_path(ain, solve_node, dst_conn="_ain", memlet=Memlet.simple(ain, in_subset, num_accesses=n * n))
    state.add_memlet_path(bin, solve_node, dst_conn="_bin", memlet=Memlet.simple(bin, out_subset, num_accesses=n * n))
    state.add_memlet_path(solve_node,
                          bout,
                          src_conn="_bout",
                          memlet=Memlet.simple(bout, out_subset, num_accesses=n * n))

    return sdfg


@pytest.mark.parametrize("implementation, dtype, size, shape", [
    pytest.param('MKL', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], marks=pytest.mark.mkl),
    pytest.param('MKL', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], marks=pytest.mark.mkl),
    pytest.param(
        'MKL', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], marks=pytest.mark.mkl),
    pytest.param(
        'MKL', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]], marks=pytest.mark.mkl),
    pytest.param('OpenBLAS', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], marks=pytest.mark.lapack),
    pytest.param('OpenBLAS', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float32,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 marks=pytest.mark.lapack),
    pytest.param('OpenBLAS',
                 np.float64,
                 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]],
                 marks=pytest.mark.lapack),
    pytest.param('cuSolverDn', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], marks=pytest.mark.gpu),
    pytest.param('cuSolverDn', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]], marks=pytest.mark.gpu)
])
def test_solve(implementation, dtype, size, shape):
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
    out_subset = tuple([slice(o, o + size) if i in out_dims else o for i, o in enumerate(out_offset)])

    in_subset_str = ','.join(
        ["{b}:{e}".format(b=o, e=o + size) if i in in_dims else str(o) for i, o in enumerate(in_offset)])
    out_subset_str = ','.join(
        ["{b}:{e}".format(b=o, e=o + size) if i in out_dims else str(o) for i, o in enumerate(out_offset)])

    sdfg = make_sdfg(implementation, dtype, id, in_shape, out_shape, in_subset_str, out_subset_str)
    if implementation == 'cuSolverDn':
        sdfg.apply_gpu_transformations()
        sdfg.coarsen_dataflow()
    solve_sdfg = sdfg.compile()

    A0 = np.zeros(in_shape, dtype=dtype)
    A0[in_subset] = generate_matrix(size, dtype)
    B0 = np.zeros(out_shape, dtype=dtype)
    B0[out_subset] = generate_matrix(size, dtype)
    A1 = np.copy(A0)
    B1 = np.copy(B0)
    B2 = np.zeros(out_shape, dtype=dtype)
    ref = np.linalg.solve(A0[in_subset], B0[out_subset])

    solve_sdfg(ain=A1, bin=B1, bout=B2, n=size)

    if dtype == np.float32:
        rtol = 1e-6
    elif dtype == np.float64:
        rtol = 1e-12
    else:
        raise NotImplementedError

    assert (np.linalg.norm(ref - B2[out_subset]) / np.linalg.norm(ref)) < rtol


###############################################################################

if __name__ == "__main__":
    test_solve('MKL', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]])
    test_solve('MKL', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]])
    test_solve('MKL', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]])
    test_solve('MKL', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]])
    test_solve('cuSolverDn', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]])
    test_solve('cuSolverDn', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]])
    test_solve('cuSolverDn', np.float32, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]])
    test_solve('cuSolverDn', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]])
