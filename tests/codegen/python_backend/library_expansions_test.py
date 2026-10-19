# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``CuPy`` expansions run the common library nodes as one array call: NumPy on the host, CuPy on the GPU."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest
import sympy

import dace
from dace.codegen.python.emit import generate
from dace.libraries.blas.nodes.axpy import Axpy
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.ger import Ger
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.libraries.python_expansions import Operands, register
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.libraries.standard.nodes.find_first import FindFirst
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.nodes.scan import Scan, ScanOp

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


@pytest.mark.gpu
def test_a_dot_product_of_gpu_vectors_lands_in_a_one_element_gpu_array():
    import cupy

    sdfg = dace.SDFG("gpu_dot")
    for name, size in (("x", 8), ("y", 8), ("r", 1)):
        sdfg.add_array(name, [size], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Dot("dot", n=8)
    state.add_node(node)
    for name, conn in (("x", "_x"), ("y", "_y")):
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    state.add_edge(node, "_result", state.add_write("r"), None, dace.Memlet("r[0]"))
    x, y, r = rng_matrix(8), rng_matrix(8)[::-1].copy(), cupy.zeros(1)

    with python_backend():
        sdfg(x=cupy.asarray(x), y=cupy.asarray(y), r=r)

    assert np.isclose(r.get()[0], np.dot(x, y))


def test_a_tensor_transpose_permutes_and_scales_like_numpy():
    node = TensorTranspose("transpose", axes=[2, 0, 1], alpha=2.0)
    sdfg = single_node_sdfg(node, {"_inp_tensor": ("A", [2, 3, 4])}, {"_out_tensor": ("B", [4, 2, 3])})
    a, b = rng_matrix(2, 3, 4), np.zeros((4, 2, 3))

    with python_backend():
        sdfg(A=a, B=b)

    assert np.array_equal(b, 2 * np.transpose(a, (2, 0, 1)))


def conflict_check_sdfg(storage: dace.StorageType) -> dace.SDFG:
    """``flag[0]`` is 1 when ``idx`` repeats a value; ``idx`` lives in ``storage``, the flag on the host."""
    sdfg = dace.SDFG(f"conflict_check_{storage.name}")
    sdfg.add_array("idx", [6], dace.int64, storage=storage)
    sdfg.add_array("flag", [1], dace.int64)
    state = sdfg.add_state()
    node = ScatterConflictCheck("check")
    state.add_node(node)
    state.add_edge(state.add_read("idx"), None, node, node.INPUT_CONNECTOR_NAME, dace.Memlet("idx[0:6]"))
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, state.add_write("flag"), None, dace.Memlet("flag[0]"))
    return sdfg


@pytest.mark.parametrize("indices, expected", [([3, 1, 4, 0, 5, 2], 0), ([3, 1, 4, 1, 5, 2], 1)])
def test_a_conflict_check_flags_exactly_the_index_arrays_that_repeat_a_value(indices: list[int], expected: int):
    flag = np.full(1, -1, dtype=np.int64)

    with python_backend():
        conflict_check_sdfg(dace.StorageType.Default)(idx=np.array(indices, dtype=np.int64), flag=flag)

    assert flag[0] == expected


@pytest.mark.gpu
def test_a_conflict_check_of_gpu_indices_leaves_its_flag_on_the_host():
    import cupy

    flag = np.full(1, -1, dtype=np.int64)

    with python_backend():
        conflict_check_sdfg(dace.StorageType.GPU_Global)(idx=cupy.array([0, 2, 2, 3, 4, 5]), flag=flag)

    assert flag[0] == 1


def test_a_reduction_without_identity_into_one_element_folds_in_what_it_holds():
    node = Reduce("reduce", wcr="lambda a, b: max(a, b)", axes=None, identity=None)
    sdfg = single_node_sdfg(node, {node.INPUT_CONNECTOR_NAME: ("A", [3, 4])}, {})
    sdfg.add_array("m", [1], dace.float64)
    state = next(iter(sdfg.states()))
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, state.add_write("m"), None, dace.Memlet("m[0]"))
    a = rng_matrix(3, 4)
    above, below = np.array([a.max() + 1.0]), np.array([a.min() - 1.0])

    with python_backend():
        sdfg(A=a, m=above)
        sdfg(A=a, m=below)

    assert above[0] == a.max() + 1.0 and below[0] == a.max()


def wired_sdfg(
    node: dace.nodes.LibraryNode,
    arrays: dict[str, tuple[list, dace.typeclass]],
    reads: dict[str, str],
    writes: dict[str, str],
    storage: dace.StorageType = dace.StorageType.Default,
) -> dace.SDFG:
    """``node`` alone in a state: ``reads`` / ``writes`` map a connector to its memlet over ``arrays``.

    A one-element array stays on the host, where the library nodes leave their answers."""
    sdfg = dace.SDFG(f"wired_{node.label}_{storage.name}")
    for name, (shape, dtype) in arrays.items():
        sdfg.add_array(name, shape, dtype, storage=storage if shape != [1] else dace.StorageType.Default)
    state = sdfg.add_state()
    state.add_node(node)
    for conn, memlet in reads.items():
        state.add_edge(state.add_read(memlet.split("[")[0]), None, node, conn, dace.Memlet(memlet))
    for conn, memlet in writes.items():
        state.add_edge(node, conn, state.add_write(memlet.split("[")[0]), None, dace.Memlet(memlet))
    return sdfg


def sequential_scan(values: np.ndarray, step, seed) -> np.ndarray:
    out, acc = np.empty_like(values), seed
    for k, value in enumerate(values):
        acc = value if acc is None else step(acc, value)
        out[k] = acc
    return out


SCAN_CASES = {
    "sum": (ScanOp.SUM, lambda acc, x: acc + x),
    "product": (ScanOp.PRODUCT, lambda acc, x: acc * x),
    "min": (ScanOp.MIN, min),
    "max": (ScanOp.MAX, max),
}


@pytest.mark.parametrize("name", SCAN_CASES)
def test_an_inclusive_scan_matches_the_sequential_loop(name: str):
    op, step = SCAN_CASES[name]
    node = Scan("scan", op=op)
    sdfg = wired_sdfg(
        node, {"x": ([13], dace.float64), "y": ([13], dace.float64)}, {"_scan_in": "x[0:13]"}, {"_scan_out": "y[0:13]"}
    )
    x, y = np.random.default_rng(3).random(13) + 0.5, np.zeros(13)

    with python_backend():
        sdfg(x=x, y=y)

    assert np.allclose(y, sequential_scan(x, step, None))


def test_an_exclusive_sum_scan_starts_from_zero():
    node = Scan("scan", op=ScanOp.SUM, exclusive=True)
    sdfg = wired_sdfg(
        node, {"x": ([5], dace.float64), "y": ([5], dace.float64)}, {"_scan_in": "x[0:5]"}, {"_scan_out": "y[0:5]"}
    )
    y = np.zeros(5)

    with python_backend():
        sdfg(x=np.arange(1.0, 6.0), y=y)

    assert y.tolist() == [0.0, 1.0, 3.0, 6.0, 10.0]


@pytest.mark.parametrize(
    "identity, dtype, values, expected",
    [
        (str(np.iinfo(np.int64).max), np.int64, [2**62, 5, 7], [np.iinfo(np.int64).max, 2**62, 5]),
        (sympy.oo, np.float64, [3.0, 1.0, 2.0], [np.inf, 3.0, 1.0]),
    ],
    ids=["int64_max_string", "sympy_infinity"],
)
def test_an_exclusive_min_scan_starts_from_its_identity_exactly(identity, dtype, values, expected):
    node = Scan("scan", op=ScanOp.MIN, exclusive=True, identity=identity)
    element = dace.dtypes.typeclass(dtype)
    sdfg = wired_sdfg(node, {"x": ([3], element), "y": ([3], element)}, {"_scan_in": "x[0:3]"}, {"_scan_out": "y[0:3]"})
    y = np.zeros(3, dtype)

    with python_backend():
        sdfg(x=np.array(values, dtype), y=y)

    assert y.tolist() == expected


def affine_scan_sdfg(storage: dace.StorageType) -> dace.SDFG:
    """``y[k] = c[k] * y[k-1] + d[k]`` from ``y[-1] = s[0]``."""
    node = Scan("scan", op=ScanOp.AFFINE)
    node.add_in_connector("_scan_init")
    arrays = {"c": ([37], dace.float64), "d": ([37], dace.float64), "y": ([37], dace.float64), "s": ([1], dace.float64)}
    reads = {"_scan_in": "d[0:37]", "_scan_coef": "c[0:37]", "_scan_init": "s[0]"}
    return wired_sdfg(node, arrays, reads, {"_scan_out": "y[0:37]"}, storage)


def affine_reference(c: np.ndarray, d: np.ndarray, seed: float) -> np.ndarray:
    return sequential_scan(np.arange(c.size, dtype=np.float64), lambda acc, k: c[int(k)] * acc + d[int(k)], seed)


def test_an_affine_scan_carries_its_seed_through_every_coefficient():
    rng = np.random.default_rng(4)
    c, d, y = rng.random(37) + 0.5, rng.random(37), np.zeros(37)

    with python_backend():
        affine_scan_sdfg(dace.StorageType.Default)(c=c, d=d, y=y, s=np.array([2.0]))

    assert np.allclose(y, affine_reference(c, d, 2.0))


@pytest.mark.gpu
def test_an_affine_scan_of_gpu_arrays_runs_in_cupy():
    import cupy

    rng = np.random.default_rng(4)
    c, d, y = rng.random(37) + 0.5, rng.random(37), cupy.zeros(37)

    with python_backend():
        affine_scan_sdfg(dace.StorageType.GPU_Global)(c=cupy.asarray(c), d=cupy.asarray(d), y=y, s=np.array([2.0]))

    assert np.allclose(y.get(), affine_reference(c, d, 2.0))


def argmax_sdfg(storage: dace.StorageType, transform: str = "") -> dace.SDFG:
    arrays = {"a": ([9], dace.float64), "v": ([1], dace.float64), "i": ([1], dace.int64)}
    return wired_sdfg(
        ArgReduce("argmax", op="max", transform=transform),
        arrays,
        {"_in": "a[0:9]"},
        {"_out_val": "v[0]", "_out_idx": "i[0]"},
        storage,
    )


def test_an_argmax_returns_the_first_of_tied_extremes_and_its_value():
    v, i = np.zeros(1), np.zeros(1, dtype=np.int64)

    with python_backend():
        argmax_sdfg(dace.StorageType.Default, transform="abs")(a=np.array([1.0, -7, 3, 7, -7, 0, 2, 1, 5]), v=v, i=i)

    assert (v[0], i[0]) == (7.0, 1)


@pytest.mark.gpu
def test_an_argmax_of_a_gpu_array_leaves_its_answers_on_the_host():
    import cupy

    v, i = np.zeros(1), np.zeros(1, dtype=np.int64)

    with python_backend():
        argmax_sdfg(dace.StorageType.GPU_Global)(a=cupy.array([1.0, 9, 3, 9, 0, 0, 2, 1, 5]), v=v, i=i)

    assert (v[0], i[0]) == (9.0, 1)


def find_first_sdfg(storage: dace.StorageType) -> dace.SDFG:
    search = FindFirst("first", predicate="_a[__i] > 0.5 and not _a[__i] > 2.0", begin=2, end=9)
    search.add_in_connector("_a")
    arrays = {"a": ([9], dace.float64), "i": ([1], dace.int64)}
    return wired_sdfg(search, arrays, {"_a": "a[0:9]"}, {"_out_idx": "i[0]"}, storage)


@pytest.mark.parametrize(
    "values, expected",
    [([1.0, 1, 0, 3, 0, 1, 1, 0, 0], 5), ([1.0, 1, 0, 0, 0, 0, 0, 0, 3], 9)],
    ids=["hit", "no_hit"],
)
def test_a_search_returns_the_first_index_in_range_where_the_predicate_holds(values: list[float], expected: int):
    i = np.zeros(1, dtype=np.int64)

    with python_backend():
        find_first_sdfg(dace.StorageType.Default)(a=np.array(values), i=i)

    assert i[0] == expected


@pytest.mark.parametrize(
    "predicate, expected",
    [("0.5 < _a[__i] < 2.0", 2), ("(_a[__i] > 1.0) if __i > 2 else False", 3)],
    ids=["chained_comparison", "conditional_expression"],
)
def test_a_search_predicate_holds_lane_by_lane(predicate: str, expected: int):
    search = FindFirst("lanes", predicate=predicate, begin=0, end=5)
    search.add_in_connector("_a")
    arrays = {"a": ([5], dace.float64), "i": ([1], dace.int64)}
    sdfg = wired_sdfg(search, arrays, {"_a": "a[0:5]"}, {"_out_idx": "i[0]"})
    i = np.zeros(1, dtype=np.int64)

    with python_backend():
        sdfg(a=np.array([0.1, 3.0, 1.0, 5.0, 1.0]), i=i)

    assert i[0] == expected


@pytest.mark.gpu
def test_a_search_over_a_gpu_array_leaves_its_answer_on_the_host():
    import cupy

    i = np.zeros(1, dtype=np.int64)

    with python_backend():
        find_first_sdfg(dace.StorageType.GPU_Global)(a=cupy.array([1.0, 1, 0, 3, 0, 1, 1, 0, 0]), i=i)

    assert i[0] == 5


def emitted(sdfg: dace.SDFG) -> dict:
    """The namespace of ``sdfg`` expanded to its CuPy tasklets and emitted as it is, with no connector inlined."""
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.LibraryNode):
            node.implementation = "CuPy"
    sdfg.expand_library_nodes()
    namespace: dict = {}
    exec(compile(generate(sdfg), sdfg.name, "exec"), namespace)
    return namespace


def test_a_reduction_without_identity_reads_what_its_output_holds_through_a_connector():
    node = Reduce("reduce", wcr="lambda a, b: max(a, b)", axes=None, identity=None)
    sdfg = single_node_sdfg(node, {node.INPUT_CONNECTOR_NAME: ("A", [3, 4])}, {})
    sdfg.add_array("m", [1], dace.float64)
    state = next(iter(sdfg.states()))
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, state.add_write("m"), None, dace.Memlet("m[0]"))
    a, m = rng_matrix(3, 4), np.array([100.0])

    emitted(sdfg)[sdfg.name](A=a, m=m)

    assert m[0] == 100.0


def single_search_sdfg(predicate: str, size: int) -> dace.SDFG:
    search = FindFirst("first", predicate=predicate, begin=0, end=size)
    search.add_in_connector("_a")
    arrays = {"a": ([size], dace.int64), "i": ([1], dace.int64)}
    return wired_sdfg(search, arrays, {"_a": f"a[0:{size}]"}, {"_out_idx": "i[0]"})


@pytest.mark.parametrize(
    "predicate, values, expected",
    [("_a[__i]", [0, 1, 5], 1), ("not _a[__i]", [-1, 0, 4], 1)],
    ids=["truthy", "negated"],
)
def test_a_search_takes_the_truth_of_a_non_boolean_predicate(predicate: str, values: list[int], expected: int):
    i = np.zeros(1, dtype=np.int64)

    with python_backend():
        single_search_sdfg(predicate, len(values))(a=np.array(values), i=i)

    assert i[0] == expected


def test_a_guard_in_a_search_predicate_still_protects_the_read_it_guards():
    """At the last index ``_a[__i + 1]`` is past the end; the sequential search never reads it."""
    i = np.zeros(1, dtype=np.int64)

    with python_backend():
        single_search_sdfg("__i + 1 < 6 and _a[__i + 1] > 0", 6)(a=np.zeros(6, dtype=np.int64), i=i)

    assert i[0] == 6


@dace.library.node
class InPlaceProbe(dace.nodes.LibraryNode):
    """A node reading and writing ``_buf``, whose CuPy body refuses."""

    implementations: dict = {}
    default_implementation = None

    def __init__(self, name: str):
        super().__init__(name, inputs={"_buf"}, outputs={"_buf"})


def refuse(o) -> str:
    raise NotImplementedError("refused")


register(InPlaceProbe, refuse, Operands)


def test_a_refused_expansion_leaves_the_node_wired_as_it_was():
    node = InPlaceProbe("probe")
    sdfg = wired_sdfg(node, {"b": ([4], dace.float64)}, {"_buf": "b[0:4]"}, {"_buf": "b[0:4]"})
    state = next(iter(sdfg.states()))
    node.implementation = "CuPy"

    with pytest.raises(NotImplementedError):
        node.expand(state)

    assert [e.dst_conn for e in state.in_edges(node)] == ["_buf"]


def test_a_transpose_of_an_operand_with_an_extent_one_dim_permutes_every_named_axis():
    node = TensorTranspose("transpose", axes=[0, 2, 1])
    arrays = {"a": ([2, 3, 4], dace.float64), "b": ([1, 4, 3], dace.float64)}
    sdfg = wired_sdfg(node, arrays, {"_inp_tensor": "a[1:2, 0:3, 0:4]"}, {"_out_tensor": "b[0:1, 0:4, 0:3]"})
    a, b = rng_matrix(2, 3, 4), np.zeros((1, 4, 3))

    with python_backend():
        sdfg(a=a, b=b)

    assert np.array_equal(b, a[1:2].transpose(0, 2, 1))


@pytest.mark.parametrize(
    "op, identity, expected",
    [(ScanOp.MIN, float("inf"), [np.inf, 3.0, 1.0, 1.0]), (ScanOp.AFFINE, 5.0, [3.0, 7.0, 15.0, 31.0])],
    ids=["exclusive_min_from_infinity", "affine_ignores_the_identity"],
)
def test_a_scan_starts_from_the_value_the_cpp_expansions_start_from(op: ScanOp, identity: float, expected: list):
    """An exclusive scan starts from its identity, ``inf`` included; an affine one from zero whatever it holds."""
    node = Scan("scan", op=op, exclusive=op is ScanOp.MIN, identity=identity)
    reads = {"_scan_in": "d[0:4]"} | ({"_scan_coef": "c[0:4]"} if op is ScanOp.AFFINE else {})
    arrays = {"d": ([4], dace.float64), "c": ([4], dace.float64), "y": ([4], dace.float64)}
    sdfg = wired_sdfg(node, arrays, reads, {"_scan_out": "y[0:4]"})
    y = np.zeros(4)

    with python_backend():
        sdfg(d=np.array([3.0, 1.0, 1.0, 1.0]), c=np.full(4, 2.0), y=y)

    assert y.tolist() == expected


def test_a_scan_of_one_element_is_that_element():
    node = Scan("scan", op=ScanOp.MAX)
    sdfg = wired_sdfg(
        node, {"x": ([1], dace.float64), "y": ([1], dace.float64)}, {"_scan_in": "x[0]"}, {"_scan_out": "y[0]"}
    )
    y = np.zeros(1)

    with python_backend():
        sdfg(x=np.array([4.0]), y=y)

    assert y[0] == 4.0
