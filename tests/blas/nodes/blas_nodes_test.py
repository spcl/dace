# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries import blas
from dace.libraries.blas.nodes import Axpy, Gemm, Gemv, MatMul
from dace.transformation.dataflow import RedundantSecondArray

M = dace.symbol("M")
N = dace.symbol("N")


@pytest.mark.parametrize(
    ("implementation",),
    [
        ("pure",),
        pytest.param("MKL", marks=pytest.mark.mkl),
        ("OpenBLAS",),
        pytest.param("cuBLAS", marks=pytest.mark.gpu),
    ],
)
def test_gemv_strided(implementation):

    @dace.program
    def gemv(A: dace.float64[M, N], x: dace.float64[N, N]):
        return A @ x[:, 1]

    A = np.random.rand(20, 30)
    x = np.random.rand(30, 30)
    reference = A @ x[:, 1]
    sdfg = gemv.to_sdfg()
    sdfg.name = f"{sdfg.name}_{implementation}"
    if implementation == "cuBLAS":
        sdfg.apply_gpu_transformations()
        sdfg.apply_transformations_repeated(RedundantSecondArray)

    blas.default_implementation = implementation
    daceres = sdfg(A=A, x=x, M=20, N=30)

    blas.default_implementation = None
    assert np.allclose(daceres, reference)


def test_dot_subset():

    @dace.program
    def dot(x: dace.float64[N, N], y: dace.float64[N, N]):
        return x[1, 1 : N - 1] @ y[1 : N - 1, 1]

    x = np.random.rand(30, 30)
    y = np.random.rand(30, 30)
    reference = x[1, 1:29] @ y[1:29, 1]
    sdfg = dot.to_sdfg()

    # Enforce one-dimensional memlets from two-dimensional arrays
    sdfg.apply_transformations_repeated(RedundantSecondArray)
    blas.default_implementation = "pure"
    daceres = sdfg(x=x, y=y, N=30)

    blas.default_implementation = None
    assert np.allclose(daceres, reference)


@pytest.mark.parametrize(
    ("implementation",),
    [
        ("pure",),
        pytest.param("MKL", marks=pytest.mark.mkl),
        ("OpenBLAS",),
        pytest.param("cuBLAS", marks=pytest.mark.gpu),
    ],
)
def test_dot_strided(implementation):

    @dace.program
    def dot(x: dace.float64[N, N], y: dace.float64[N, N]):
        return x[1, :] @ y[:, 1]

    x = np.random.rand(30, 30)
    y = np.random.rand(30, 30)
    reference = x[1, :] @ y[:, 1]
    sdfg = dot.to_sdfg()
    sdfg.name = f"{sdfg.name}_{implementation}"

    # Enforce one-dimensional memlets from two-dimensional arrays
    sdfg.apply_transformations_repeated(RedundantSecondArray)

    if implementation == "cuBLAS":
        sdfg.apply_gpu_transformations()

    blas.default_implementation = implementation
    daceres = sdfg(x=x, y=y, N=30)

    blas.default_implementation = None
    assert np.allclose(daceres, reference)


@pytest.mark.parametrize(("implementation",), [("pure",), pytest.param("MKL", marks=pytest.mark.mkl), ("OpenBLAS",)])
def test_transpose_strided_columns(implementation):
    """``pos[:, 0:2]`` is an (N, 2) block whose rows are 3 apart: a vendor transpose reads it by that row stride,
    not as 2 * N contiguous elements."""

    @dace.program
    def transpose_columns(pos: dace.float64[N, 3]):
        return pos[:, 0:2].T

    pos = np.random.rand(25, 3)
    sdfg = transpose_columns.to_sdfg()
    sdfg.name = f"{sdfg.name}_{implementation}"
    transposes = [node for node, _ in sdfg.all_nodes_recursive() if type(node).__name__ == "Transpose"]
    assert transposes, "the transpose did not become a Transpose library node"
    for node in transposes:
        node.implementation = implementation
    assert np.allclose(sdfg(pos=pos, N=25), pos[:, 0:2].T)


def wired_sdfg(name: str, node, inputs: dict[str, str], outputs: dict[str, str], arrays: dict) -> dace.SDFG:
    """``node`` between one access node per memlet of ``inputs`` and ``outputs``."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    for array_name, (shape, dtype, strides) in arrays.items():
        sdfg.add_array(array_name, shape, dtype, strides=strides)
    state.add_node(node)
    for connector, text in inputs.items():
        memlet = dace.Memlet(text)
        state.add_edge(state.add_read(memlet.data), None, node, connector, memlet)
    for connector, text in outputs.items():
        memlet = dace.Memlet(text)
        state.add_edge(node, connector, state.add_write(memlet.data), None, memlet)
    return sdfg


MATRIX = ([2, 2], dace.float64, None)
VECTOR = ([2], dace.float64, None)


def test_gemv_scales_by_the_whole_expression():
    """``alpha * A`` without parentheses bound as ``s + (1 * A)``."""
    node = Gemv("gemv", alpha="s + 1", beta="s - 3")
    node.implementation = "pure"
    arrays = {"A": MATRIX, "x": VECTOR, "y": VECTOR}
    sdfg = wired_sdfg("gemv_expression", node, {"_A": "A", "_x": "x", "_y": "y"}, {"_y": "y"}, arrays)
    sdfg.add_symbol("s", dace.float64)

    a, x, y = np.full((2, 2), 5.0), np.full(2, 7.0), np.full(2, 11.0)
    sdfg(A=a, x=x, y=y, s=2.0)

    np.testing.assert_allclose(y, 3.0 * a @ x - 11.0)


def test_axpy_scales_by_the_whole_expression():
    node = Axpy("axpy", a="s + 1", n=2)
    node.implementation = "pure"
    arrays = {"x": VECTOR, "y": VECTOR, "res": VECTOR}
    sdfg = wired_sdfg("axpy_expression", node, {"_x": "x", "_y": "y"}, {"_res": "res"}, arrays)
    sdfg.add_symbol("s", dace.float64)

    x, y, res = np.full(2, 2.0), np.ones(2), np.zeros(2)
    sdfg(x=x, y=y, res=res, s=2.0)

    np.testing.assert_allclose(res, 3.0 * x + y)


def test_openblas_gemv_prints_a_symbolic_extent_as_cpp():
    """``N**2`` in the call is not C++."""
    extent = dace.symbol("E", dace.int64)
    node = Gemv("gemv")
    node.implementation = "OpenBLAS"
    arrays = {
        "A": ([extent**2, extent], dace.float64, None),
        "x": ([extent], dace.float64, None),
        "y": ([extent**2], dace.float64, None),
    }
    sdfg = wired_sdfg("gemv_power_extent", node, {"_A": "A", "_x": "x"}, {"_y": "y"}, arrays)

    a, x, y = np.arange(8.0).reshape(4, 2).copy(), np.ones(2), np.zeros(4)
    sdfg(A=a, x=x, y=y, E=2)

    np.testing.assert_allclose(y, a @ x)


def test_openblas_gemm_of_integers_falls_back_to_pure():
    """BLAS has no integer routine, and Gemv, Dot and the new nodes all route such a type to pure."""
    node = Gemm("gemm")
    node.implementation = "OpenBLAS"
    arrays = {name: ([2, 2], dace.int32, None) for name in ("a", "b", "c")}
    sdfg = wired_sdfg("gemm_integers", node, {"_a": "a", "_b": "b"}, {"_c": "c"}, arrays)

    a, c = np.arange(4, dtype=np.int32).reshape(2, 2).copy(), np.zeros((2, 2), dtype=np.int32)
    sdfg(a=a, b=a, c=c)

    np.testing.assert_array_equal(c, a @ a)


def test_a_scaled_matmul_of_two_vectors_is_refused():
    """Dot has no scale, so ``alpha`` vanished from the result."""
    arrays = {"x": VECTOR, "y": VECTOR, "out": ([1], dace.float64, None)}
    sdfg = wired_sdfg(
        "matmul_scaled_dot", MatMul("matmul", alpha=2.0), {"_a": "x", "_b": "y"}, {"_c": "out[0]"}, arrays
    )
    with pytest.raises(NotImplementedError, match="alpha"):
        sdfg.expand_library_nodes()


if __name__ == "__main__":
    implementations = ["pure", "MKL", "cuBLAS"]
    for implementation in implementations:
        test_gemv_strided(implementation)
    test_dot_subset()
    for implementation in implementations:
        test_dot_strided(implementation)
    for implementation in ["pure", "MKL", "OpenBLAS"]:
        test_transpose_strided_columns(implementation)
    test_gemv_scales_by_the_whole_expression()
    test_axpy_scales_by_the_whole_expression()
    test_openblas_gemv_prints_a_symbolic_extent_as_cpp()
    test_openblas_gemm_of_integers_falls_back_to_pure()
    test_a_scaled_matmul_of_two_vectors_is_refused()
