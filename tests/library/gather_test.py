# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``Gather`` library node, and the advanced indexing that the Python frontend lowers to it."""

import numpy as np
import pytest

import dace
from dace.libraries.standard import Gather

R, K, M = 10, 6, 4
rng = np.random.default_rng(0)


def gathers(sdfg: dace.SDFG) -> list[Gather]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Gather)]


@dace.program
def rows(x: dace.float32[R, K], idx: dace.int32[M], out: dace.float32[M, K]):
    out[:] = x[idx]


@dace.program
def columns_of_slice(x: dace.float32[R, K], idx: dace.int64[2, 3], out: dace.float32[R - 2, 2, 3]):
    out[:] = x[1 : R - 1, idx]


@dace.program
def element_of_rows(x: dace.float32[R, K], idx: dace.int32[M], out: dace.float32[M]):
    out[:] = x[idx, 2]


def test_rows():
    assert len(gathers(rows.to_sdfg())) == 1
    x = rng.random((R, K), np.float32)
    idx = np.array([9, 0, 0, 4], np.int32)
    out = np.zeros((M, K), np.float32)
    rows(x=x, idx=idx, out=out)
    assert np.array_equal(out, x[idx])


def test_multidimensional_index_of_a_slice():
    sdfg = columns_of_slice.to_sdfg()
    (node,) = gathers(sdfg)
    assert node.axis == 1
    x = rng.random((R, K), np.float32)
    idx = np.array([[5, 0, 1], [1, 1, 3]], np.int64)
    out = np.zeros((R - 2, 2, 3), np.float32)
    columns_of_slice(x=x, idx=idx, out=out)
    assert np.array_equal(out, x[1 : R - 1, idx])


def test_other_advanced_indexing_is_not_a_gather():
    """An index in another dimension drops it: the frontend keeps its general advanced indexing."""
    assert not gathers(element_of_rows.to_sdfg())
    x = rng.random((R, K), np.float32)
    idx = np.array([3, 1, 4, 1], np.int32)
    out = np.zeros(M, np.float32)
    element_of_rows(x=x, idx=idx, out=out)
    assert np.array_equal(out, x[idx, 2])


def test_gather_node():
    sdfg = dace.SDFG("gather_node")
    sdfg.add_array("x", [R, K, 3], dace.float64)
    sdfg.add_array("i", [M], dace.int32)
    sdfg.add_array("y", [R, M, 3], dace.float64)
    state = sdfg.add_state()
    node = Gather("gather", axis=1)
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_inp", dace.Memlet("x"))
    state.add_edge(state.add_read("i"), None, node, "_index", dace.Memlet("i"))
    state.add_edge(node, "_out", state.add_write("y"), None, dace.Memlet("y"))
    x = rng.random((R, K, 3))
    i = np.array([5, 5, 0, 2], np.int32)
    y = np.zeros((R, M, 3))
    sdfg(x=x, i=i, y=y)
    assert np.array_equal(y, np.take(x, i, axis=1))


@pytest.mark.gpu
def test_rows_on_gpu():
    sdfg = rows.to_sdfg()
    sdfg.apply_gpu_transformations()
    x = rng.random((R, K), np.float32)
    idx = np.array([2, 7, 7, 0], np.int32)
    out = np.zeros((M, K), np.float32)
    sdfg(x=x, idx=idx, out=out)
    assert np.array_equal(out, x[idx])


if __name__ == "__main__":
    test_rows()
    test_rows_on_gpu()
    test_multidimensional_index_of_a_slice()
    test_other_advanced_indexing_is_not_a_gather()
    test_gather_node()
