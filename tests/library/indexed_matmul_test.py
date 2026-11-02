# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Index connectors of the matrix-product library nodes (see ``dace.libraries.blas.indexed``)."""

import numpy as np
import pytest

import dace
import dace.libraries.blas as blas
from dace.libraries.blas.indexed import index_connector


def product_sdfg(name: str, node, arrays: dict, inputs: dict, output: tuple, wcr: str | None = None) -> dace.SDFG:
    """An SDFG of one library node: ``arrays`` maps names to (shape, dtype), ``inputs`` connectors to arrays (index
    connectors included), ``output`` is (connector, array)."""
    sdfg = dace.SDFG(name)
    for array, (shape, dtype) in arrays.items():
        sdfg.add_array(array, shape, dtype)
    state = sdfg.add_state()
    state.add_node(node)
    for connector, array in inputs.items():
        if connector not in node.in_connectors:
            node.add_in_connector(connector)
        state.add_edge(state.add_read(array), None, node, connector, dace.Memlet.from_array(array, sdfg.arrays[array]))
    connector, array = output
    memlet = dace.Memlet.from_array(array, sdfg.arrays[array])
    memlet.wcr = wcr
    state.add_edge(node, connector, state.add_write(array), None, memlet)
    return sdfg


R, M, K, N = 20, 6, 8, 5
F32, I32 = dace.float32, dace.int32
rng = np.random.default_rng(0)


def test_gathered_rows():
    sdfg = product_sdfg(
        "indexed_gathered_rows",
        blas.MatMul("matmul"),
        {"x": ([R, K], F32), "rows": ([M], I32), "w": ([K, N], F32), "y": ([M, N], F32)},
        {"_a": "x", index_connector("_a"): "rows", "_b": "w"},
        ("_c", "y"),
    )
    x, w = rng.random((R, K), np.float32), rng.random((K, N), np.float32)
    rows = np.array([19, 0, 3, 3, 7, 12], np.int32)
    y = np.zeros((M, N), np.float32)
    sdfg(x=x, rows=rows, w=w, y=y)
    assert np.allclose(y, x[rows] @ w, rtol=1e-5)


def test_scatter_add_rows():
    sdfg = product_sdfg(
        "indexed_scatter_add_rows",
        blas.MatMul("matmul"),
        {"a": ([M, K], F32), "w": ([K, N], F32), "rows": ([M], I32), "y": ([R, N], F32)},
        {"_a": "a", "_b": "w", index_connector("_c"): "rows"},
        ("_c", "y"),
        wcr="lambda x, y: x + y",
    )
    a, w = rng.random((M, K), np.float32), rng.random((K, N), np.float32)
    rows = np.array([4, 4, 0, 19, 4, 1], np.int32)  # Repeated rows add up
    y = rng.random((R, N), np.float32)
    expected = y.copy()
    np.add.at(expected, rows, a @ w)
    sdfg(a=a, w=w, rows=rows, y=y)
    assert np.allclose(y, expected, rtol=1e-5)


def test_grouped_batched_product():
    G, E = 3, 4  # Tiles and groups
    sdfg = product_sdfg(
        "indexed_grouped_batched_product",
        blas.MatMul("matmul"),
        {"x": ([R, K], F32), "rows": ([G, M], I32), "w": ([E, K, N], F32), "group": ([G], I32), "y": ([G, M, N], F32)},
        {"_a": "x", index_connector("_a"): "rows", "_b": "w", index_connector("_b"): "group"},
        ("_c", "y"),
    )
    x, w = rng.random((R, K), np.float32), rng.random((E, K, N), np.float32)
    rows = rng.integers(0, R, (G, M), dtype=np.int32)
    group = np.array([3, 0, 3], np.int32)
    y = np.zeros((G, M, N), np.float32)
    sdfg(x=x, rows=rows, w=w, group=group, y=y)
    assert np.allclose(y, x[rows] @ w[group], rtol=1e-5)


def test_gathered_matrix_rows_times_vector():
    sdfg = product_sdfg(
        "indexed_gemv",
        blas.MatMul("matmul"),
        {"a": ([R, K], F32), "rows": ([M], I32), "v": ([K], F32), "y": ([M], F32)},
        {"_a": "a", index_connector("_a"): "rows", "_b": "v"},
        ("_c", "y"),
    )
    a, v = rng.random((R, K), np.float32), rng.random(K, np.float32)
    rows = np.array([2, 2, 5, 0, 19, 8], np.int32)
    y = np.zeros(M, np.float32)
    sdfg(a=a, rows=rows, v=v, y=y)
    assert np.allclose(y, a[rows] @ v, rtol=1e-5)


def test_gathered_dot_product():
    sdfg = product_sdfg(
        "indexed_dot",
        blas.MatMul("matmul"),
        {"u": ([R], F32), "i": ([M], I32), "v": ([R], F32), "j": ([M], I32), "y": ([1], F32)},
        {"_a": "u", index_connector("_a"): "i", "_b": "v", index_connector("_b"): "j"},
        ("_c", "y"),
    )
    u, v = rng.random(R, np.float32), rng.random(R, np.float32)
    i, j = np.array([0, 1, 2, 3, 4, 5], np.int32), np.array([19, 18, 0, 0, 1, 7], np.int32)
    y = np.zeros(1, np.float32)
    sdfg(u=u, i=i, v=v, j=j, y=y)
    assert np.allclose(y[0], u[i] @ v[j], rtol=1e-5)


def test_gemm_node_with_index():
    """A Gemm node created with an index connector, with each CPU implementation that is available."""
    environments = {"MKL": blas.environments.IntelMKL, "OpenBLAS": blas.environments.OpenBLAS}
    for implementation in ("pure", "MKL", "OpenBLAS"):
        if implementation in environments and not environments[implementation].is_installed():
            continue
        node = blas.Gemm("gemm")
        node.implementation = implementation
        sdfg = product_sdfg(
            f"indexed_gemm_{implementation}",
            node,
            {"x": ([R, K], F32), "rows": ([M], I32), "w": ([K, N], F32), "y": ([M, N], F32)},
            {"_a": "x", index_connector("_a"): "rows", "_b": "w"},
            ("_c", "y"),
        )
        x, w = rng.random((R, K), np.float32), rng.random((K, N), np.float32)
        rows = np.array([1, 2, 3, 4, 19, 0], np.int32)
        y = np.zeros((M, N), np.float32)
        sdfg(x=x, rows=rows, w=w, y=y)
        assert np.allclose(y, x[rows] @ w, rtol=1e-5), implementation


@pytest.mark.gpu
def test_gathered_rows_cublas():
    import cupy

    node = blas.MatMul("matmul")
    sdfg = product_sdfg(
        "indexed_gathered_rows_cublas",
        node,
        {"x": ([R, K], F32), "rows": ([M], I32), "w": ([K, N], F32), "y": ([R, N], F32)},
        {"_a": "x", index_connector("_a"): "rows", "_b": "w", index_connector("_c"): "rows"},
        ("_c", "y"),
        wcr="lambda x, y: x + y",
    )
    for desc in sdfg.arrays.values():
        desc.storage = dace.StorageType.GPU_Global
    blas.default_implementation = "cuBLAS"
    try:
        x, w = cupy.random.rand(R, K, dtype=np.float32), cupy.random.rand(K, N, dtype=np.float32)
        rows = cupy.array([3, 1, 4, 1, 5, 9], dtype=np.int32)
        y = cupy.zeros((R, N), dtype=np.float32)
        expected = cupy.asnumpy(y).copy()
        np.add.at(expected, cupy.asnumpy(rows), cupy.asnumpy(x[rows] @ w))
        sdfg(x=x, rows=rows, w=w, y=y)
        assert np.allclose(cupy.asnumpy(y), expected, rtol=1e-5)
    finally:
        blas.default_implementation = None


if __name__ == "__main__":
    test_gathered_rows()
    test_scatter_add_rows()
    test_grouped_batched_product()
    test_gathered_matrix_rows_times_vector()
    test_gathered_dot_product()
    test_gemm_node_with_index()
    test_gathered_rows_cublas()
