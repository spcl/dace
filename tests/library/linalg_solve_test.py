# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import Memlet
from dace.libraries.lapack import Getrf, Getrs
from dace.libraries.linalg import Solve
from dace.libraries.linalg.nodes.transpose import Transpose
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
    pytest.param('pure', np.float32, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]]),
    pytest.param('pure', np.float64, 4, [[4, 4], [4, 4], [0, 0], [0, 0], [0, 1], [0, 1]]),
    pytest.param('pure', np.float64, 4, [[5, 5, 5], [5, 5, 5], [1, 3, 0], [2, 0, 1], [0, 2], [1, 2]]),
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
        sdfg.simplify()
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

# A single right-hand side reaches the library node RANK-1 whichever way the caller spelled it:
# ``Solve.validate`` squeezes the memlet subsets, so an ``(n, 1)`` container arrives with the same
# squeezed shape as a plain vector. The vendor path therefore has to serve both spellings.
SINGLE_RHS_SHAPES = {'vector': [n], 'column': [n, 1]}


def make_rhs_sdfg(implementation, dtype, rhs_shape, uid):
    """``bout = solve(ain, bin)`` with a right-hand side of the given shape."""

    sdfg = dace.SDFG("linalg_solve_rhs_{}_{}_{}".format(implementation, dtype.__name__, uid))
    sdfg.add_symbol("n", dace.int64)
    state = sdfg.add_state("dataflow")

    sdfg.add_array("ain", [n, n], dtype)
    sdfg.add_array("bin", rhs_shape, dtype)
    sdfg.add_array("bout", rhs_shape, dtype)

    subset = "0:n" if len(rhs_shape) == 1 else "0:n, 0:{}".format(rhs_shape[1])
    solve_node = Solve("solve")
    solve_node.implementation = implementation

    state.add_memlet_path(state.add_read("ain"), solve_node, dst_conn="_ain", memlet=Memlet.simple("ain", "0:n, 0:n"))
    state.add_memlet_path(state.add_read("bin"), solve_node, dst_conn="_bin", memlet=Memlet.simple("bin", subset))
    state.add_memlet_path(solve_node, state.add_write("bout"), src_conn="_bout", memlet=Memlet.simple("bout", subset))

    return sdfg


def expansion_report(sdfg):
    """Expand ``Solve`` one level and report WHICH expansion ran, not just that one did.

    :return: ``(lapack node names, staged right-hand-side shape, transpose count)``. The shape is
             stringified because a symbol carries its dtype into equality. A pure expansion has no
             LAPACK node and no staging array at all.
    """
    sdfg.expand_library_nodes(recursive=False)
    lapack, binout, transposes = [], None, 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (Getrf, Getrs)):
            lapack.append(type(node).__name__)
        elif isinstance(node, Transpose):
            transposes += 1
        elif isinstance(node, dace.sdfg.nodes.NestedSDFG) and '_binout' in node.sdfg.arrays:
            binout = [str(d) for d in node.sdfg.arrays['_binout'].shape]
    return sorted(lapack), binout, transposes


@pytest.mark.parametrize("implementation", ['OpenBLAS', 'MKL'])
@pytest.mark.parametrize("spelling", sorted(SINGLE_RHS_SHAPES))
def test_single_rhs_expands_to_vendor_getrs(implementation, spelling):
    sdfg = make_rhs_sdfg(implementation, np.float64, SINGLE_RHS_SHAPES[spelling], 100)
    lapack, binout, transposes = expansion_report(sdfg)

    assert lapack == ['Getrf', 'Getrs']
    assert binout == ['n']
    assert transposes == 0
    assert 'LAPACKE_dgetrs' in ''.join(c.clean_code for c in sdfg.generate_code())


@pytest.mark.parametrize("implementation", ['OpenBLAS', 'MKL'])
def test_multi_rhs_expands_to_vendor_getrs(implementation):
    sdfg = make_rhs_sdfg(implementation, np.float64, [n, 3], 101)
    lapack, binout, transposes = expansion_report(sdfg)

    assert lapack == ['Getrf', 'Getrs']
    assert binout == ['n', '3']
    assert transposes == 0


@pytest.mark.parametrize("spelling", sorted(SINGLE_RHS_SHAPES))
def test_single_rhs_gpu_skips_the_rhs_transposes(spelling):
    sdfg = make_rhs_sdfg('cuSolverDn', np.float64, SINGLE_RHS_SHAPES[spelling], 102)
    lapack, binout, transposes = expansion_report(sdfg)

    # One column is contiguous in both layouts, so only A is transposed; the multi-RHS graph below
    # needs three transposes for the same solve.
    assert lapack == ['Getrf', 'Getrs']
    assert binout == ['n']
    assert transposes == 1


def test_multi_rhs_gpu_keeps_the_rhs_transposes():
    sdfg = make_rhs_sdfg('cuSolverDn', np.float64, [n, 3], 103)
    lapack, binout, transposes = expansion_report(sdfg)

    assert lapack == ['Getrf', 'Getrs']
    assert binout == ['3', 'n']
    assert transposes == 3


@pytest.mark.parametrize("spelling", sorted(SINGLE_RHS_SHAPES))
def test_single_rhs_pure_stays_pure(spelling):
    sdfg = make_rhs_sdfg('pure', np.float64, SINGLE_RHS_SHAPES[spelling], 104)
    lapack, binout, transposes = expansion_report(sdfg)

    assert lapack == []
    assert binout is None
    assert transposes == 0


@pytest.mark.parametrize("implementation, rhs_shape", [
    pytest.param('pure', [n]),
    pytest.param('pure', [n, 1]),
    pytest.param('pure', [n, 3]),
    pytest.param('OpenBLAS', [n], marks=pytest.mark.lapack),
    pytest.param('OpenBLAS', [n, 1], marks=pytest.mark.lapack),
    pytest.param('OpenBLAS', [n, 3], marks=pytest.mark.lapack),
    pytest.param('MKL', [n], marks=pytest.mark.mkl),
    pytest.param('MKL', [n, 1], marks=pytest.mark.mkl),
    pytest.param('MKL', [n, 3], marks=pytest.mark.mkl),
    pytest.param('cuSolverDn', [n], marks=pytest.mark.gpu),
    pytest.param('cuSolverDn', [n, 1], marks=pytest.mark.gpu),
    pytest.param('cuSolverDn', [n, 3], marks=pytest.mark.gpu),
])
def test_rhs_shape_values(implementation, rhs_shape):
    global id
    id += 1
    size = 6
    dtype = np.float64
    sdfg = make_rhs_sdfg(implementation, dtype, rhs_shape, id)
    if implementation == 'cuSolverDn':
        sdfg.apply_gpu_transformations()
        sdfg.simplify()

    rng = np.random.default_rng(42)
    A = (generate_matrix(size, dtype) + size * np.eye(size)).astype(dtype)
    shape = (size, ) if len(rhs_shape) == 1 else (size, rhs_shape[1])
    B = rng.random(shape).astype(dtype)
    out = np.zeros(shape, dtype=dtype)

    sdfg.compile()(ain=A.copy(), bin=B.copy(), bout=out, n=size)

    ref = np.linalg.solve(A, B)
    assert np.linalg.norm(ref - out) / np.linalg.norm(ref) < 1e-12


@pytest.mark.parametrize("implementation", [
    pytest.param('pure'),
    pytest.param('OpenBLAS', marks=pytest.mark.lapack),
    pytest.param('MKL', marks=pytest.mark.mkl),
])
def test_single_rhs_strided_slice_values(implementation):
    """A single right-hand side that is a strided column of a bigger container."""

    global id
    id += 1
    size = 6
    sdfg = dace.SDFG("linalg_solve_strided_{}_{}".format(implementation, id))
    sdfg.add_symbol("n", dace.int64)
    state = sdfg.add_state("dataflow")
    sdfg.add_array("ain", [n, n], np.float64)
    sdfg.add_array("bin", [5, n, 5], np.float64)
    sdfg.add_array("bout", [5, n, 5], np.float64)

    solve_node = Solve("solve")
    solve_node.implementation = implementation
    state.add_memlet_path(state.add_read("ain"), solve_node, dst_conn="_ain", memlet=Memlet.simple("ain", "0:n, 0:n"))
    state.add_memlet_path(state.add_read("bin"), solve_node, dst_conn="_bin", memlet=Memlet.simple("bin", "1, 0:n, 2"))
    state.add_memlet_path(solve_node,
                          state.add_write("bout"),
                          src_conn="_bout",
                          memlet=Memlet.simple("bout", "3, 0:n, 4"))

    rng = np.random.default_rng(3)
    A = generate_matrix(size, np.float64) + size * np.eye(size)
    B = rng.random((5, size, 5))
    out = np.zeros((5, size, 5))

    sdfg.compile()(ain=A.copy(), bin=B.copy(), bout=out, n=size)

    ref = np.linalg.solve(A, B[1, :, 2])
    assert np.linalg.norm(ref - out[3, :, 4]) / np.linalg.norm(ref) < 1e-12
    # Nothing outside the destination slice may be touched.
    untouched = np.ones((5, size, 5), dtype=bool)
    untouched[3, :, 4] = False
    assert np.all(out[untouched] == 0)


@pytest.mark.lapack
def test_frontend_single_rhs_is_not_refused():
    """``np.linalg.solve(A, vector)`` -- the spelling that used to raise NotImplementedError."""

    N = dace.symbol('N')

    @dace.program
    def solve_vector(A: dace.float64[N, N], b: dace.float64[N], out: dace.float64[N]):
        out[:] = np.linalg.solve(A, b)

    size = 6
    sdfg = solve_vector.to_sdfg(simplify=True)
    sdfg.expand_library_nodes(recursive=False)
    assert 'LAPACKE_dgetrs' in ''.join(c.clean_code for c in sdfg.generate_code())

    rng = np.random.default_rng(7)
    A = rng.random((size, size)) + size * np.eye(size)
    b = rng.random(size)
    out = np.zeros(size)
    sdfg.compile()(A=A.copy(), b=b.copy(), out=out, N=size)

    assert np.allclose(out, np.linalg.solve(A, b))


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
