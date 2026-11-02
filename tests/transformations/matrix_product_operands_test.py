# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Gathers and transpositions of matrix-product operands, fused into the product."""

import numpy as np

import dace
from dace.libraries.blas import MatMul
from dace.libraries.linalg import Transpose
from dace.libraries.standard import Gather
from dace.transformation.dataflow import GatherIntoMatrixProduct, TransposeIntoMatrixProduct

R, K, N, M, B = 12, 8, 6, 5, 3
rng = np.random.default_rng(0)
FUSIONS = [GatherIntoMatrixProduct, TransposeIntoMatrixProduct]


def count(sdfg: dace.SDFG, kind) -> int:
    return sum(isinstance(n, kind) for n, _ in sdfg.all_nodes_recursive())


@dace.program
def gathered_rows_times_transposed(x: dace.float32[R, K], idx: dace.int32[M], w: dace.float32[N, K]):
    return x[idx] @ w.T


@dace.program
def transposed_times_matrix(w: dace.float32[K, N], x: dace.float32[K, M]):
    return w.T @ x


@dace.program
def gathered_columns(x: dace.float32[K, R], idx: dace.int32[M], w: dace.float32[M, N]):
    return x[:, idx] @ w


@dace.program
def gathered_and_kept(x: dace.float32[R, K], idx: dace.int32[M], w: dace.float32[K, N], kept: dace.float32[M, K]):
    rows = x[idx]
    kept[:] = rows
    return rows @ w


@dace.program
def batched_times_transposed(x: dace.float32[B, M, K], w: dace.float32[N, K]):
    return x @ w.T


def test_gathered_rows_times_transposed():
    sdfg = gathered_rows_times_transposed.to_sdfg()
    assert sdfg.apply_transformations_repeated(FUSIONS) == 2
    assert count(sdfg, Gather) == 0 and count(sdfg, Transpose) == 0
    (product,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MatMul)]
    assert "_a_index" in product.in_connectors and product.transB and not product.transA
    x, w = rng.random((R, K), np.float32), rng.random((N, K), np.float32)
    idx = np.array([11, 0, 3, 3, 6], np.int32)
    assert np.allclose(sdfg(x=x, idx=idx, w=w), x[idx] @ w.T, rtol=1e-5)


def test_transposed_first_operand():
    sdfg = transposed_times_matrix.to_sdfg()
    assert sdfg.apply_transformations_repeated(FUSIONS) == 1
    (product,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MatMul)]
    assert product.transA and not product.transB
    w, x = rng.random((K, N), np.float32), rng.random((K, M), np.float32)
    assert np.allclose(sdfg(w=w, x=x), w.T @ x, rtol=1e-5)


def test_gather_of_another_dimension_stays():
    sdfg = gathered_columns.to_sdfg()
    assert sdfg.apply_transformations_repeated(FUSIONS) == 0
    x, w = rng.random((K, R), np.float32), rng.random((M, N), np.float32)
    idx = np.array([1, 2, 3, 5, 8], np.int32)
    assert np.allclose(sdfg(x=x, idx=idx, w=w), x[:, idx] @ w, rtol=1e-5)


def test_gather_used_elsewhere_stays():
    sdfg = gathered_and_kept.to_sdfg()
    assert sdfg.apply_transformations_repeated(FUSIONS) == 0
    assert count(sdfg, Gather) == 1


def test_batched_product_keeps_its_transposition():
    """A product with a batched operand specializes to BatchedMatMul, which does not take transB."""
    sdfg = batched_times_transposed.to_sdfg()
    assert sdfg.apply_transformations_repeated(FUSIONS) == 0
    x, w = rng.random((B, M, K), np.float32), rng.random((N, K), np.float32)
    assert np.allclose(sdfg(x=x, w=w), x @ w.T, rtol=1e-5)


if __name__ == "__main__":
    test_gathered_rows_times_transposed()
    test_transposed_first_operand()
    test_gather_of_another_dimension_stays()
    test_gather_used_elsewhere_stays()
    test_batched_product_keeps_its_transposition()
