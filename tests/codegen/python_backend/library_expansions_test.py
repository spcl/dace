# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``CuPy`` expansions run the common library nodes as one array call: NumPy on the host, CuPy on the GPU."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes.axpy import Axpy
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.ger import Ger
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.fill.node import FillLibraryNode

N = dace.symbol("N")


@contextlib.contextmanager
def python_backend() -> Iterator[None]:
    with dace.config.set_temporary("compiler", "backend", value="python"):
        yield


def function_source(sdfg: dace.SDFG) -> str:
    with python_backend():
        compiled = sdfg.compile()
    source = compiled.path.read_text()
    return source[source.index(f"def {sdfg.name}(") :]


def rng_matrix(*shape: int) -> np.ndarray:
    return np.random.default_rng(7).random(shape)


@dace.program
def matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    C[:] = A @ B


def test_a_matrix_product_is_one_numpy_matmul():
    a, b, c = rng_matrix(4, 4), rng_matrix(4, 4).T.copy(), np.zeros((4, 4))

    with python_backend():
        matmul(a, b, c)

    assert np.allclose(c, a @ b)
    source = function_source(matmul.to_sdfg())
    assert source.count(" @ ") == 1 and "for " not in source


@dace.program
def matvec(A: dace.float64[N, N], x: dace.float64[N], y: dace.float64[N]):
    y[:] = A @ x


def test_a_matrix_vector_product_matches_numpy():
    a, x, y = rng_matrix(5, 5), rng_matrix(5), np.zeros(5)

    with python_backend():
        matvec(a, x, y)

    assert np.allclose(y, a @ x)


@dace.program
def inner(x: dace.float64[N], y: dace.float64[N], r: dace.float64[1]):
    r[0] = np.dot(x, y)


def test_a_dot_product_matches_numpy():
    x, y, r = rng_matrix(6), rng_matrix(6)[::-1].copy(), np.zeros(1)

    with python_backend():
        inner(x, y, r)

    assert np.isclose(r[0], x @ y)


@dace.program
def row_and_total_sums(A: dace.float64[N, 4], rows: dace.float64[N], total: dace.float64[1]):
    rows[:] = np.sum(A, axis=1)
    total[0] = np.sum(A)


def test_reductions_along_an_axis_and_over_everything_match_numpy():
    a, rows, total = rng_matrix(3, 4), np.zeros(3), np.zeros(1)

    with python_backend():
        row_and_total_sums(a, rows, total)

    assert np.allclose(rows, a.sum(axis=1)) and np.isclose(total[0], a.sum())
    assert "np.sum(" in function_source(row_and_total_sums.to_sdfg())


def einsum_sdfg() -> dace.SDFG:
    """One ``Einsum`` node, ``C = einsum("ijk,kj->i", A, B)``."""
    sdfg = dace.SDFG("contraction")
    sdfg.add_array("A", [3, 4, 5], dace.float64)
    sdfg.add_array("B", [5, 4], dace.float64)
    sdfg.add_array("C", [3], dace.float64)
    state = sdfg.add_state()
    node = Einsum("einsum")
    node.einsum_str = "ijk,kj->i"
    state.add_node(node)
    for name in "AB":
        node.add_in_connector(f"_{name.lower()}")
        state.add_edge(
            state.add_read(name), None, node, f"_{name.lower()}", dace.Memlet.from_array(name, sdfg.arrays[name])
        )
    node.add_out_connector("_out")
    state.add_edge(node, "_out", state.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def test_an_einsum_is_one_numpy_einsum():
    a, b, c = rng_matrix(3, 4, 5), rng_matrix(5, 4), np.zeros(3)

    with python_backend():
        einsum_sdfg()(A=a, B=b, C=c)

    assert np.allclose(c, np.einsum("ijk,kj->i", a, b))
    assert "np.einsum('ijk,kj->i', _a, _b)" in function_source(einsum_sdfg())


@dace.program
def factor_and_solve(A: dace.float64[4, 4], b: dace.float64[4], L: dace.float64[4, 4], x: dace.float64[4]):
    L[:] = np.linalg.cholesky(A)
    x[:] = np.linalg.solve(A, b)


def test_a_cholesky_factor_and_a_solve_match_numpy():
    m = rng_matrix(4, 4)
    a, b = m @ m.T + 4 * np.eye(4), rng_matrix(4)
    lower, x = np.zeros((4, 4)), np.zeros(4)

    with python_backend():
        factor_and_solve(a, b, lower, x)

    assert np.allclose(lower, np.linalg.cholesky(a)) and np.allclose(a @ x, b)


def gpu_gemm_sdfg() -> dace.SDFG:
    """``C = 2 * A @ B.T + 0.5 * C`` with every operand in GPU global memory."""
    sdfg = dace.SDFG("gpu_gemm")
    for name in "ABC":
        sdfg.add_array(name, [8, 8], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Gemm("gemm", transB=True, alpha=2.0, beta=0.5)
    state.add_node(node)
    for name, conn in (("A", "_a"), ("B", "_b"), ("C", "_c")):
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


@pytest.mark.gpu
def test_a_gemm_on_gpu_arrays_runs_in_cupy():
    import cupy

    a, b, c = rng_matrix(8, 8), rng_matrix(8, 8), rng_matrix(8, 8)
    c_gpu = cupy.asarray(c)

    with python_backend():
        gpu_gemm_sdfg()(A=cupy.asarray(a), B=cupy.asarray(b), C=c_gpu)

    assert np.allclose(c_gpu.get(), 2 * a @ b.T + 0.5 * c)
    assert "cupy.copyto(_c, " in function_source(gpu_gemm_sdfg())


def round_trip_sdfg() -> dace.SDFG:
    """Host ``A`` to a GPU transient, doubled there by a ``Gemm`` with the identity, and back to host ``B``."""
    sdfg = dace.SDFG("round_trip")
    sdfg.add_array("A", [4, 4], dace.float64)
    sdfg.add_array("I", [4, 4], dace.float64)
    sdfg.add_array("B", [4, 4], dace.float64)
    for name in ("gA", "gI", "gB"):
        sdfg.add_array(name, [4, 4], dace.float64, storage=dace.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state()
    node = Gemm("gemm", alpha=2.0)
    state.add_node(node)
    for host, device, conn in (("A", "gA", "_a"), ("I", "gI", "_b")):
        on_device = state.add_access(device)
        state.add_edge(state.add_read(host), None, on_device, None, dace.Memlet.from_array(host, sdfg.arrays[host]))
        state.add_edge(on_device, None, node, conn, dace.Memlet.from_array(device, sdfg.arrays[device]))
    result = state.add_access("gB")
    state.add_edge(node, "_c", result, None, dace.Memlet.from_array("gB", sdfg.arrays["gB"]))
    state.add_edge(result, None, state.add_write("B"), None, dace.Memlet.from_array("gB", sdfg.arrays["gB"]))
    return sdfg


@pytest.mark.gpu
def test_host_arrays_cross_to_the_gpu_and_back_around_a_cupy_call():
    a, b = rng_matrix(4, 4), np.zeros((4, 4))

    with python_backend():
        round_trip_sdfg()(A=a, I=np.eye(4), B=b)

    assert np.allclose(b, 2 * a)
    source = function_source(round_trip_sdfg())
    assert "cupy.asarray(A[" in source and "cupy.asnumpy(gB[" in source and "cupy.copyto(" in source


@dace.program
def batched(A: dace.float64[2, 3, 4], B: dace.float64[2, 4, 5], C: dace.float64[2, 3, 5]):
    C[:] = A @ B


def test_a_batched_matrix_product_matches_numpy():
    a, b, c = rng_matrix(2, 3, 4), rng_matrix(2, 4, 5), np.zeros((2, 3, 5))

    with python_backend():
        batched(a, b, c)

    assert np.allclose(c, a @ b)


def single_node_sdfg(node: dace.nodes.LibraryNode, inputs: dict, outputs: dict) -> dace.SDFG:
    """One library node between host arrays: ``inputs`` and ``outputs`` map a connector to ``(array, shape)``."""
    sdfg = dace.SDFG(f"single_{node.label}")
    state = sdfg.add_state()
    state.add_node(node)
    for conn, (name, shape) in inputs.items():
        sdfg.add_array(name, shape, dace.float64)
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    for conn, (name, shape) in outputs.items():
        if name not in sdfg.arrays:
            sdfg.add_array(name, shape, dace.float64)
        state.add_edge(node, conn, state.add_write(name), None, dace.Memlet.from_array(name, sdfg.arrays[name]))
    return sdfg


def test_an_axpy_matches_numpy():
    node = Axpy("axpy", a=3.0, n=4)
    sdfg = single_node_sdfg(node, {"_x": ("x", [4]), "_y": ("y", [4])}, {"_res": ("r", [4])})
    x, y, r = rng_matrix(4), rng_matrix(4)[::-1].copy(), np.zeros(4)

    with python_backend():
        sdfg(x=x, y=y, r=r)

    assert np.allclose(r, 3 * x + y)


def test_a_rank_one_update_matches_numpy():
    node = Ger("ger", n=4, m=3, alpha=2.0)
    sdfg = single_node_sdfg(node, {"_x": ("x", [3]), "_y": ("y", [4]), "_A": ("A", [3, 4])}, {"_res": ("R", [3, 4])})
    x, y, a, r = rng_matrix(3), rng_matrix(4), rng_matrix(3, 4), np.zeros((3, 4))

    with python_backend():
        sdfg(x=x, y=y, A=a, R=r)

    assert np.allclose(r, 2 * np.outer(x, y) + a)


def test_a_copy_node_reshapes_into_its_destination():
    node = CopyLibraryNode("copy")
    sdfg = single_node_sdfg(node, {node.INPUT_CONNECTOR_NAME: ("A", [2, 6])}, {node.OUTPUT_CONNECTOR_NAME: ("B", [12])})
    a, b = rng_matrix(2, 6), np.zeros(12)

    with python_backend():
        sdfg(A=a, B=b)

    assert np.array_equal(b, a.ravel())


def test_a_fill_node_writes_its_constant_everywhere():
    node = FillLibraryNode("fill", value=2.5)
    sdfg = single_node_sdfg(node, {}, {node.OUTPUT_CONNECTOR_NAME: ("B", [3, 3])})
    b = np.zeros((3, 3))

    with python_backend():
        sdfg(B=b)

    assert np.array_equal(b, np.full((3, 3), 2.5))


@pytest.mark.gpu
def test_a_gpu_return_value_comes_back_as_a_cupy_array():
    import cupy

    sdfg = dace.SDFG("gpu_return")
    sdfg.add_array("__return", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = FillLibraryNode("fill", value=3.0)
    state.add_node(node)
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, state.add_write("__return"), None, dace.Memlet("__return[0:4]"))

    with python_backend():
        result = sdfg()

    assert isinstance(result, cupy.ndarray) and np.array_equal(result.get(), np.full(4, 3.0))
