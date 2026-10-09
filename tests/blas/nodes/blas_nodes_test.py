# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
import dace.libraries.blas as blas
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
def test_transpose_strided_column(implementation):
    """``pos[:, 0:1]`` is an (N, 1) column whose rows are 3 apart: a vendor transpose reads it by that stride,
    not as N contiguous elements (nbody's pairwise separations under OpenBLAS)."""

    @dace.program
    def transpose_column(pos: dace.float64[N, 3]):
        return pos[:, 0:1].T

    pos = np.random.rand(25, 3)
    sdfg = transpose_column.to_sdfg()
    sdfg.name = f"{sdfg.name}_{implementation}"
    transposes = [node for node, parent in sdfg.all_nodes_recursive() if type(node).__name__ == "Transpose"]
    assert transposes, "the column transpose did not become a Transpose library node"
    for node in transposes:
        node.implementation = implementation
    assert np.allclose(sdfg(pos=pos, N=25), pos[:, 0:1].T)


if __name__ == "__main__":
    implementations = ["pure", "MKL", "cuBLAS"]
    for implementation in implementations:
        test_gemv_strided(implementation)
    test_dot_subset()
    for implementation in implementations:
        test_dot_strided(implementation)
    for implementation in ["pure", "MKL", "OpenBLAS"]:
        test_transpose_strided_column(implementation)
