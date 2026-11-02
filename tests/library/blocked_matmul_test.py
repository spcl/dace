# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Matrix-product operands stored in blocks of columns (``dace.libraries.blas.blocked``): the ``BlockMatrixOperands``
pass, the expansions that copy blocked operands into the usual layout, and the ``Pack`` library node."""

import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm, MatMul
from dace.transformation.dataflow import GatherIntoMatrixProduct, TransposeIntoMatrixProduct
from dace.transformation.passes import BlockMatrixOperands

M, N, K, E, R, BLOCK = 6, 5, 16, 3, 10, 4
rng = np.random.default_rng(0)


def pack(w: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """``w[..., R, C]`` in blocks of ``block`` columns: ``[..., C / block, R, block]``."""
    *batch, rows, columns = w.shape
    return w.reshape(*batch, rows, columns // block, block).swapaxes(-3, -2).copy()


def blocked(program, *names: str) -> dace.SDFG:
    sdfg = program.to_sdfg()
    sdfg.apply_transformations_repeated([GatherIntoMatrixProduct, TransposeIntoMatrixProduct])
    BlockMatrixOperands(names=set(names), block=BLOCK).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def products(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (MatMul, Gemm))]


@dace.program
def times_transposed(x: dace.float64[M, K], w: dace.float64[N, K]):
    return x @ w.T


@dace.program
def gathered_times_transposed(x: dace.float64[R, K], idx: dace.int32[M], w: dace.float64[N, K]):
    return x[idx] @ w.T


@dace.program
def times_matrix(x: dace.float64[M, K], w: dace.float64[K, N * BLOCK]):
    return x @ w


@dace.program
def experts(x: dace.float64[E, M, K], w: dace.float64[E, N, K], y: dace.float64[E, M, N]):
    for e in dace.map[0:E]:
        y[e] = x[e] @ w[e].T


@dace.program
def column_slices(x: dace.float64[M, K], w: dace.float64[N, K], parts: dace.float64[K // 8, M, N]):
    for ch in dace.map[0 : K // 8]:
        parts[ch] = x[:, ch * 8 : ch * 8 + 8] @ w[:, ch * 8 : ch * 8 + 8].T


@dace.program
def product_and_element(x: dace.float64[M, K], w: dace.float64[N, K], y: dace.float64[M, N], z: dace.float64[N]):
    y[:] = x @ w.T
    for i in dace.map[0:N]:
        z[i] = w[i, 6]


def test_times_transposed():
    sdfg = blocked(times_transposed, "w")
    (product,) = products(sdfg)
    assert product.blocked == {"_b"} and product.transB
    assert list(sdfg.arrays["w"].shape) == [K // BLOCK, N, BLOCK]
    x, w = rng.random((M, K)), rng.random((N, K))
    assert np.allclose(sdfg(x=x, w=pack(w)), x @ w.T)


def test_gathered_rows_times_transposed():
    sdfg = blocked(gathered_times_transposed, "w")
    (product,) = products(sdfg)
    assert product.blocked == {"_b"} and "_a_index" in product.in_connectors
    x, w = rng.random((R, K)), rng.random((N, K))
    idx = np.array([9, 0, 3, 3, 6, 1], np.int32)
    assert np.allclose(sdfg(x=x, idx=idx, w=pack(w)), x[idx] @ w.T)


def test_blocked_along_the_columns_of_the_product():
    """Blocks of the columns of ``b`` (not of the product's inner dimension), which only the copies read."""
    sdfg = blocked(times_matrix, "w")
    x, w = rng.random((M, K)), rng.random((K, N * BLOCK))
    assert np.allclose(sdfg(x=x, w=pack(w)), x @ w)


def test_batch_in_a_map():
    """The weights of every expert, through a map and a nested SDFG."""
    sdfg = blocked(experts, "w")
    assert list(sdfg.arrays["w"].shape) == [E, K // BLOCK, N, BLOCK]
    assert all(p.blocked == {"_b"} for p in products(sdfg))
    x, w, y = rng.random((E, M, K)), rng.random((E, N, K)), np.zeros((E, M, N))
    sdfg(x=x, w=pack(w), y=y)
    assert np.allclose(y, x @ w.swapaxes(1, 2))


def test_slices_of_whole_blocks():
    sdfg = blocked(column_slices, "w")
    x, w, parts = rng.random((M, K)), rng.random((N, K)), np.zeros((K // 8, M, N))
    sdfg(x=x, w=pack(w), parts=parts)
    expected = np.stack([x[:, c : c + 8] @ w[:, c : c + 8].T for c in range(0, K, 8)])
    assert np.allclose(parts, expected)


def test_elements_are_read_in_blocks():
    sdfg = blocked(product_and_element, "w")
    x, w, y, z = rng.random((M, K)), rng.random((N, K)), np.zeros((M, N)), np.zeros(N)
    sdfg(x=x, w=pack(w), y=y, z=z)
    assert np.allclose(y, x @ w.T) and np.allclose(z, w[:, 6])


@dace.program
def written(x: dace.float64[M, K], w: dace.float64[N, K]):
    w[0, 0] = 1
    return x @ w.T


@dace.program
def partial_block(x: dace.float64[M, 4], w: dace.float64[N, K]):
    return x @ w[:, 2:6].T


@pytest.mark.parametrize("program", [written, partial_block])
def test_rejected(program):
    sdfg = program.to_sdfg()
    sdfg.apply_transformations_repeated([TransposeIntoMatrixProduct])
    before = sdfg.to_json()
    with pytest.raises(ValueError, match="cannot be stored in blocks|not whole blocks"):
        BlockMatrixOperands(names={"w"}, block=BLOCK).apply_pass(sdfg, {})
    assert sdfg.to_json() == before  # Unchanged


@dace.program
def pack_program(w: dace.float64[E, N, K], out: dace.float64[E, K // BLOCK, N, BLOCK]):
    dace.libraries.blas.pack(w, out, block=BLOCK)


def test_pack_node():
    w, out = rng.random((E, N, K)), np.zeros((E, K // BLOCK, N, BLOCK))
    pack_program(w=w, out=out)
    assert np.array_equal(out, pack(w))


if __name__ == "__main__":
    test_times_transposed()
    test_gathered_rows_times_transposed()
    test_blocked_along_the_columns_of_the_product()
    test_batch_in_a_map()
    test_slices_of_whole_blocks()
    test_elements_are_read_in_blocks()
    test_rejected(written)
    test_rejected(partial_block)
    test_pack_node()
