# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An array element passed as the output of a library call reaches the call as a copy; the result must be written back."""

import numpy as np

import dace
import dace.libraries.blas
import dace.libraries.standard

N = dace.symbol("N")


@dace.program
def norm2_rows(x: dace.float64[N, 8], out: dace.float64[N]):
    for i in dace.map[0:N]:
        dace.libraries.standard.norm2(x[i, :], out[i])


@dace.program
def argmax_into_element(x: dace.float64[8], idx: dace.int64[3]):
    dace.libraries.standard.argmax(x, idx[1])


@dace.program
def dot_rows(x: dace.float64[N, 8], y: dace.float64[8], out: dace.float64[N]):
    for i in dace.map[0:N]:
        dace.libraries.blas.dot(x[i, :], y, out[i])


@dace.program
def read_element(x: dace.float64[8], out: dace.float64[1]):
    out[0] = np.sqrt(x[2])


def written(sdfg: dace.SDFG, name: str) -> bool:
    """Whether some access node of ``name``, in ``sdfg`` or an SDFG nested in it, is written."""
    return any(
        node.data == name and state.in_degree(node) > 0
        for nested in sdfg.all_sdfgs_recursive()
        for state in nested.states()
        for node in state.data_nodes()
    )


def test_norm2_into_element_in_map():
    sdfg = norm2_rows.to_sdfg(simplify=False)
    assert written(sdfg, "out")
    x = np.random.default_rng(0).random((5, 8))
    out = np.zeros(5)
    norm2_rows(x, out, N=5)
    assert np.allclose(out, np.linalg.norm(x, axis=1))


def test_argmax_into_element():
    sdfg = argmax_into_element.to_sdfg(simplify=False)
    assert written(sdfg, "idx")
    x = np.random.default_rng(1).random(8)
    idx = np.full(3, -7, dtype=np.int64)
    argmax_into_element(x, idx)
    assert idx[0] == -7 and idx[2] == -7
    assert idx[1] == np.argmax(x) + 1  # one-based by default


def test_dot_into_element_in_map():
    sdfg = dot_rows.to_sdfg(simplify=False)
    assert written(sdfg, "out")
    rng = np.random.default_rng(2)
    x, y = rng.random((5, 8)), rng.random(8)
    out = np.zeros(5)
    dot_rows(x, y, out, N=5)
    assert np.allclose(out, x @ y)


def test_read_element_not_written_back():
    sdfg = read_element.to_sdfg(simplify=False)
    assert not written(sdfg, "x")


if __name__ == "__main__":
    test_norm2_into_element_in_map()
    test_argmax_into_element()
    test_dot_into_element_in_map()
    test_read_element_not_written_back()
