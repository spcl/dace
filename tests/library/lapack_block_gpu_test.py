# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LAPACK and dense ``linalg`` nodes inside a GPU kernel lower to ONE thread block per matrix.

A ``GPU_Device`` map over a batch of small matrices holds the node; the in-kernel pick is ``'CUDA (block)'``, a
``GPU_ThreadBlock`` lane map around one tasklet calling ``dace/cuda/block_lapack.cuh``. The numeric tests compare
every matrix of the batch with NumPy / SciPy.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.lapack import Getrf, GetrfNoPivot, Getri, Getrs, Potrf, Syevd
from dace.libraries.lapack.block import BLOCK_LAPACK
from dace.libraries.linalg import Cholesky, Eigh, Inv, Solve
from dace.libraries.standard.block_reduce import (
    gpu_block_implementation,
    in_kernel_block_implementation,
    release_kernel_block_size,
)
from dace.sdfg import nodes
from dace.transformation.passes.offloading.taskloop import is_kernel_refused_libnode

BATCH = 6
N = 13
RHS = 3


def batched_kernel(
    library_node: nodes.LibraryNode,
    arrays: dict[str, tuple[list, dace.typeclass]],
    inputs: dict[str, str],
    outputs: dict[str, str],
):
    """``library_node`` in a ``GPU_Device`` map over ``b``, wired to the global ``arrays`` by the given memlets."""
    sdfg = dace.SDFG(f"batched_{library_node.label}")
    for name, (shape, dtype) in arrays.items():
        sdfg.add_array(name, [BATCH] + shape, dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("batch", {"b": f"0:{BATCH}"}, schedule=dtypes.ScheduleType.GPU_Device)
    state.add_node(library_node)
    for conn, subset in inputs.items():
        state.add_memlet_path(
            state.add_read(subset.split("[")[0]), entry, library_node, dst_conn=conn, memlet=dace.Memlet(subset)
        )
    for conn, subset in outputs.items():
        state.add_memlet_path(
            library_node, exit_node, state.add_write(subset.split("[")[0]), src_conn=conn, memlet=dace.Memlet(subset)
        )
    return sdfg, state


def mat(name: str, n: int = N) -> str:
    return f"{name}[b, 0:{n}, 0:{n}]"


def vec(name: str, n: int = N) -> str:
    return f"{name}[b, 0:{n}]"


def lower_in_kernel(sdfg: dace.SDFG, state: dace.SDFGState, node: nodes.LibraryNode) -> str:
    """Take the in-kernel pick, expand, and return the block tasklet's code."""
    assert not is_kernel_refused_libnode(node, state), "a node with a block lowering may sit in a kernel"
    assert in_kernel_block_implementation(node, state) == BLOCK_LAPACK
    node.implementation = BLOCK_LAPACK
    release_kernel_block_size(node, state)
    sdfg.expand_library_nodes()
    lanes = [
        n
        for n in (pair[0] for pair in sdfg.all_nodes_recursive())
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert [lane.map.range.size()[0] for lane in lanes] == [256], "one lane map of the block's width"
    (tasklet,) = [n for n in (pair[0] for pair in sdfg.all_nodes_recursive()) if isinstance(n, nodes.Tasklet)]
    return tasklet.code.as_string


def kernel_getrf(dtype: dace.typeclass):
    node = Getrf("getrf")
    return node, batched_kernel(
        node,
        {"A": ([N, N], dtype), "piv": ([N], dace.int32), "info": ([], dace.int32)},
        {"_xin": mat("A")},
        {"_xout": mat("A"), "_ipiv": vec("piv"), "_res": "info[b]"},
    )


def kernel_getrf_nopivot(dtype: dace.typeclass):
    node = GetrfNoPivot("getrf_nopivot")
    return node, batched_kernel(node, {"A": ([N, N], dtype)}, {"_x": mat("A")}, {"_x": mat("A")})


def kernel_getrs(dtype: dace.typeclass):
    node = Getrs("getrs")
    return node, batched_kernel(
        node,
        {"LU": ([N, N], dtype), "piv": ([N], dace.int32), "B": ([N, RHS], dtype), "info": ([], dace.int32)},
        {"_a": mat("LU"), "_ipiv": vec("piv"), "_rhs_in": f"B[b, 0:{N}, 0:{RHS}]"},
        {"_rhs_out": f"B[b, 0:{N}, 0:{RHS}]", "_res": "info[b]"},
    )


def kernel_getri(dtype: dace.typeclass):
    node = Getri("getri")
    return node, batched_kernel(
        node,
        {"A": ([N, N], dtype), "piv": ([N], dace.int32), "info": ([], dace.int32)},
        {"_xin": mat("A"), "_ipiv": vec("piv")},
        {"_xout": mat("A"), "_res": "info[b]"},
    )


def kernel_potrf(dtype: dace.typeclass, lower: bool = True):
    node = Potrf("potrf", lower=lower)
    return node, batched_kernel(
        node,
        {"A": ([N, N], dtype), "info": ([], dace.int32)},
        {"_xin": mat("A")},
        {"_xout": mat("A"), "_res": "info[b]"},
    )


def kernel_syevd(dtype: dace.typeclass):
    node = Syevd("syevd")
    return node, batched_kernel(
        node,
        {"A": ([N, N], dtype), "w": ([N], dtype), "info": ([], dace.int32)},
        {"_xin": mat("A")},
        {"_xout": mat("A"), "_evals": vec("w"), "_res": "info[b]"},
    )


def kernel_cholesky(dtype: dace.typeclass, lower: bool = True):
    node = Cholesky("cholesky", lower=lower)
    return node, batched_kernel(node, {"A": ([N, N], dtype), "L": ([N, N], dtype)}, {"_a": mat("A")}, {"_b": mat("L")})


def kernel_inv(dtype: dace.typeclass):
    node = Inv("inv")
    return node, batched_kernel(
        node, {"A": ([N, N], dtype), "X": ([N, N], dtype)}, {"_ain": mat("A")}, {"_aout": mat("X")}
    )


def kernel_solve(dtype: dace.typeclass):
    node = Solve("solve")
    return node, batched_kernel(
        node,
        {"A": ([N, N], dtype), "B": ([N, RHS], dtype), "X": ([N, RHS], dtype)},
        {"_ain": mat("A"), "_bin": f"B[b, 0:{N}, 0:{RHS}]"},
        {"_bout": f"X[b, 0:{N}, 0:{RHS}]"},
    )


def kernel_eigh(dtype: dace.typeclass):
    node = Eigh("eigh")
    return node, batched_kernel(
        node,
        {"A": ([N, N], dtype), "w": ([N], dtype), "V": ([N, N], dtype)},
        {"_a": mat("A")},
        {"_w": vec("w"), "_v": mat("V")},
    )


KERNELS = {
    "getrf": (kernel_getrf, "dace::block_lapack::getrf("),
    "getrf_nopivot": (kernel_getrf_nopivot, "dace::block_lapack::getrf_nopivot("),
    "getrs": (kernel_getrs, "dace::block_lapack::getrs("),
    "getri": (kernel_getri, "dace::block_lapack::getri("),
    "potrf": (kernel_potrf, "dace::block_lapack::potrf("),
    "syevd": (kernel_syevd, "dace::block_lapack::syevj("),
    "cholesky": (kernel_cholesky, "dace::block_lapack::potrf("),
    "inv": (kernel_inv, "dace::block_lapack::getri("),
    "solve": (kernel_solve, "dace::block_lapack::getrs("),
    "eigh": (kernel_eigh, "dace::block_lapack::syevj("),
}


@pytest.mark.parametrize("name", list(KERNELS))
def test_an_in_kernel_lapack_node_lowers_to_one_thread_block(name: str):
    build, call = KERNELS[name]
    node, (sdfg, state) = build(dace.float64)
    code = lower_in_kernel(sdfg, state, node)
    assert call in code
    sdfg.validate()


def test_a_symbolic_order_refuses_the_shared_scratch_lowerings():
    n = dace.symbol("n")
    square = ([n, n], dace.float64)
    eigh = Eigh("eigh")
    sdfg, state = batched_kernel(
        eigh,
        {"A": square, "V": square, "w": ([n], dace.float64)},
        {"_a": "A[b, 0:n, 0:n]"},
        {"_w": "w[b, 0:n]", "_v": "V[b, 0:n, 0:n]"},
    )
    assert gpu_block_implementation(eigh, state, sdfg) is None
    assert is_kernel_refused_libnode(eigh, state)
    inv = Inv("inv")
    sdfg, state = batched_kernel(
        inv, {"A": square, "X": square}, {"_ain": "A[b, 0:n, 0:n]"}, {"_aout": "X[b, 0:n, 0:n]"}
    )
    assert gpu_block_implementation(inv, state, sdfg) is None
    # A factorization works on its operands alone, so any order takes the block
    getrf = Getrf("getrf")
    sdfg, state = batched_kernel(
        getrf,
        {"A": square, "piv": ([n], dace.int32), "info": ([], dace.int32)},
        {"_xin": "A[b, 0:n, 0:n]"},
        {"_xout": "A[b, 0:n, 0:n]", "_ipiv": "piv[b, 0:n]", "_res": "info[b]"},
    )
    assert gpu_block_implementation(getrf, state, sdfg) == BLOCK_LAPACK


def test_a_complex_matrix_refuses_the_block_lowering():
    node, (sdfg, state) = kernel_potrf(dace.complex128)
    assert gpu_block_implementation(node, state, sdfg) is None


def getri_on_the_device(dtype: dace.typeclass) -> dace.SDFG:
    """A host-issued ``Getri`` on GPU-resident operands, through the vendor solver."""
    sdfg = dace.SDFG("getri_device")
    sdfg.add_array("A", [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("piv", [N], dace.int32, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("info", [1], dace.int32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Getri("getri")
    node.implementation = "cuSolverDn"
    state.add_edge(state.add_read("A"), None, node, "_xin", dace.Memlet(f"A[0:{N}, 0:{N}]"))
    state.add_edge(state.add_read("piv"), None, node, "_ipiv", dace.Memlet(f"piv[0:{N}]"))
    state.add_edge(node, "_xout", state.add_write("A"), None, dace.Memlet(f"A[0:{N}, 0:{N}]"))
    state.add_edge(node, "_res", state.add_write("info"), None, dace.Memlet("info[0]"))
    return sdfg


def test_getri_on_the_device_solves_against_an_identity():
    sdfg = getri_on_the_device(dace.float64)
    sdfg.expand_library_nodes(recursive=False)
    (nested,) = [n for n in sdfg.start_state.nodes() if isinstance(n, nodes.NestedSDFG)]
    (solve,) = [n for n in nested.sdfg.start_state.nodes() if isinstance(n, Getrs)]
    assert solve.implementation == "cuSolverDn"
    assert nested.sdfg.arrays["_eye"].transient


def random_batch(rng: np.random.Generator, shape: tuple, dtype) -> np.ndarray:
    return rng.standard_normal((BATCH,) + shape).astype(dtype)


def spd_batch(rng: np.random.Generator, dtype) -> np.ndarray:
    a = random_batch(rng, (N, N), np.float64)
    return (a @ a.transpose(0, 2, 1) + N * np.eye(N)).astype(dtype)


def symmetric_batch(rng: np.random.Generator, dtype) -> np.ndarray:
    a = random_batch(rng, (N, N), np.float64)
    return (a + a.transpose(0, 2, 1)).astype(dtype)


def tolerance(dtype) -> dict:
    return {"rtol": 2e-3, "atol": 2e-3} if dtype == np.float32 else {"rtol": 1e-9, "atol": 1e-9}


def run(name: str, dtype, **arrays: np.ndarray) -> dict[str, np.ndarray]:
    """Build kernel ``name`` at ``dtype``, lower it in-kernel, run it on device copies of ``arrays``."""
    import cupy

    build = KERNELS[name][0]
    node, (sdfg, state) = build(dace.dtype_to_typeclass(dtype))
    lower_in_kernel(sdfg, state, node)
    device = {k: cupy.asarray(v) for k, v in arrays.items()}
    sdfg(**device)
    return {k: cupy.asnumpy(v) for k, v in device.items()}


def lu_factors(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import scipy.linalg

    factors = [scipy.linalg.lu_factor(m) for m in a]
    return np.stack([f[0] for f in factors]), np.stack([f[1] + 1 for f in factors]).astype(np.int32)


def sign_normalized(v: np.ndarray) -> np.ndarray:
    """Each eigenvector column scaled so its largest-magnitude entry is positive."""
    picks = np.take_along_axis(v, np.abs(v).argmax(axis=-2)[..., None, :], axis=-2)
    return v * np.sign(picks)


DTYPES = [np.float64, np.float32]


@pytest.mark.gpu
def test_getri_on_the_device_matches_numpy_inv():
    import cupy
    import scipy.linalg

    a = np.random.default_rng(11).standard_normal((N, N))
    # cuSolverDn reads the buffer column-major, so it holds the factors of A^T transposed
    lu, piv = scipy.linalg.lu_factor(a.T)
    device = {"A": cupy.asarray(np.ascontiguousarray(lu.T)), "piv": cupy.asarray(piv.astype(np.int32) + 1)}
    device["info"] = cupy.ones(1, np.int32)
    getri_on_the_device(dace.float64)(**device)
    np.testing.assert_allclose(cupy.asnumpy(device["A"]), np.linalg.inv(a), rtol=1e-9, atol=1e-9)
    assert int(device["info"][0]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_getrf_in_a_kernel_matches_scipy(dtype):
    a = random_batch(np.random.default_rng(0), (N, N), dtype)
    out = run("getrf", dtype, A=a.copy(), piv=np.zeros((BATCH, N), np.int32), info=np.ones(BATCH, np.int32))
    lu, piv = lu_factors(a.astype(np.float64))
    np.testing.assert_array_equal(out["piv"], piv)
    np.testing.assert_allclose(out["A"], lu, **tolerance(dtype))
    np.testing.assert_array_equal(out["info"], 0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_getrf_nopivot_in_a_kernel_reconstructs_the_matrix(dtype):
    a = spd_batch(np.random.default_rng(1), dtype)
    out = run("getrf_nopivot", dtype, A=a.copy())["A"].astype(np.float64)
    lower = np.tril(out, -1) + np.eye(N)
    np.testing.assert_allclose(lower @ np.triu(out), a, **tolerance(dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_getrs_in_a_kernel_matches_numpy_solve(dtype):
    rng = np.random.default_rng(2)
    a, b = random_batch(rng, (N, N), np.float64), random_batch(rng, (N, RHS), dtype)
    lu, piv = lu_factors(a)
    out = run("getrs", dtype, LU=lu.astype(dtype), piv=piv, B=b.copy(), info=np.ones(BATCH, np.int32))
    np.testing.assert_allclose(out["B"], np.linalg.solve(a, b.astype(np.float64)), **tolerance(dtype))
    np.testing.assert_array_equal(out["info"], 0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_getri_in_a_kernel_matches_numpy_inv(dtype):
    a = random_batch(np.random.default_rng(3), (N, N), np.float64)
    lu, piv = lu_factors(a)
    out = run("getri", dtype, A=lu.astype(dtype), piv=piv, info=np.ones(BATCH, np.int32))
    np.testing.assert_allclose(out["A"], np.linalg.inv(a), **tolerance(dtype))
    np.testing.assert_array_equal(out["info"], 0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_potrf_in_a_kernel_factors_the_lower_triangle_only(dtype):
    a = spd_batch(np.random.default_rng(4), dtype)
    out = run("potrf", dtype, A=a.copy(), info=np.ones(BATCH, np.int32))
    np.testing.assert_allclose(np.tril(out["A"]), np.linalg.cholesky(a.astype(np.float64)), **tolerance(dtype))
    np.testing.assert_array_equal(np.triu(out["A"], 1), np.triu(a, 1))
    np.testing.assert_array_equal(out["info"], 0)


@pytest.mark.gpu
def test_potrf_in_a_kernel_reports_the_first_non_positive_pivot():
    a = spd_batch(np.random.default_rng(5), np.float64)
    a[:, 4, 4] = -1.0
    out = run("potrf", np.float64, A=a.copy(), info=np.zeros(BATCH, np.int32))
    np.testing.assert_array_equal(out["info"], 5)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_syevd_in_a_kernel_matches_numpy_eigh(dtype):
    a = symmetric_batch(np.random.default_rng(6), dtype)
    out = run("syevd", dtype, A=a.copy(), w=np.zeros((BATCH, N), dtype), info=np.ones(BATCH, np.int32))
    w, v = np.linalg.eigh(a.astype(np.float64))
    np.testing.assert_allclose(out["w"], w, **tolerance(dtype))
    np.testing.assert_allclose(sign_normalized(out["A"]), sign_normalized(v), **tolerance(dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_cholesky_in_a_kernel_matches_numpy(dtype):
    a = spd_batch(np.random.default_rng(7), dtype)
    out = run("cholesky", dtype, A=a, L=np.full((BATCH, N, N), 7, dtype))
    np.testing.assert_allclose(out["L"], np.linalg.cholesky(a.astype(np.float64)), **tolerance(dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_inv_in_a_kernel_matches_numpy(dtype):
    a = random_batch(np.random.default_rng(8), (N, N), dtype)
    out = run("inv", dtype, A=a, X=np.zeros_like(a))
    np.testing.assert_allclose(out["X"], np.linalg.inv(a.astype(np.float64)), **tolerance(dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_solve_in_a_kernel_matches_numpy(dtype):
    rng = np.random.default_rng(9)
    a, b = random_batch(rng, (N, N), dtype), random_batch(rng, (N, RHS), dtype)
    out = run("solve", dtype, A=a, B=b, X=np.zeros_like(b))
    np.testing.assert_allclose(
        out["X"], np.linalg.solve(a.astype(np.float64), b.astype(np.float64)), **tolerance(dtype)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_eigh_in_a_kernel_matches_numpy(dtype):
    a = symmetric_batch(np.random.default_rng(10), dtype)
    out = run("eigh", dtype, A=a, w=np.zeros((BATCH, N), dtype), V=np.zeros_like(a))
    w, v = np.linalg.eigh(a.astype(np.float64))
    np.testing.assert_allclose(out["w"], w, **tolerance(dtype))
    np.testing.assert_allclose(sign_normalized(out["V"]), sign_normalized(v), **tolerance(dtype))


if __name__ == "__main__":
    for kernel in KERNELS:
        test_an_in_kernel_lapack_node_lowers_to_one_thread_block(kernel)
    test_a_symbolic_order_refuses_the_shared_scratch_lowerings()
    test_a_complex_matrix_refuses_the_block_lowering()
    test_getri_on_the_device_solves_against_an_identity()
    test_getri_on_the_device_matches_numpy_inv()
    for precision in DTYPES:
        test_getrf_in_a_kernel_matches_scipy(precision)
        test_getrf_nopivot_in_a_kernel_reconstructs_the_matrix(precision)
        test_getrs_in_a_kernel_matches_numpy_solve(precision)
        test_getri_in_a_kernel_matches_numpy_inv(precision)
        test_potrf_in_a_kernel_factors_the_lower_triangle_only(precision)
        test_syevd_in_a_kernel_matches_numpy_eigh(precision)
        test_cholesky_in_a_kernel_matches_numpy(precision)
        test_inv_in_a_kernel_matches_numpy(precision)
        test_solve_in_a_kernel_matches_numpy(precision)
        test_eigh_in_a_kernel_matches_numpy(precision)
    test_potrf_in_a_kernel_reports_the_first_non_positive_pivot()
