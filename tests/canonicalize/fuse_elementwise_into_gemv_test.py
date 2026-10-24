# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`FuseElementwiseIntoGemv`: a matrix-vector product over an elementwise map's private matrix
becomes a WCR map that fuses with the producer; GEMM and shared matrices keep their library node."""

import numpy as np

import dace
from dace.libraries.blas.nodes.matmul import MatMul
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize.fuse_elementwise_into_gemv import FuseElementwiseIntoGemv
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")


def _apply(sdfg) -> int:
    res = Pipeline([FuseElementwiseIntoGemv()]).apply_pass(sdfg, {})
    return (res or {}).get("FuseElementwiseIntoGemv", 0) or 0


def _square_transients(sdfg) -> list[str]:
    return [
        k
        for g in sdfg.all_sdfgs_recursive()
        for k, d in g.arrays.items()
        if d.transient and len(d.shape) == 2 and d.total_size != 1
    ]


@dace.program
def _pairwise_acc(x: dace.float64[N], mass: dace.float64[N], ax: dace.float64[N]):
    dx = np.reshape(x, (1, N)) - np.reshape(x, (N, 1))
    ax[:] = (dx * (dx * dx + 0.01) ** (-1.5)) @ mass


@dace.program
def _gemm(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N]):
    c[:] = (a * 2.0) @ b


@dace.program
def _matrix_read_twice(x: dace.float64[N], mass: dace.float64[N], ax: dace.float64[N], m: dace.float64[N, N]):
    dx = np.reshape(x, (1, N)) - np.reshape(x, (N, 1))
    ax[:] = dx @ mass
    m[:] = dx


def test_a_gemv_over_a_private_elementwise_matrix_fuses_away_the_matrix():
    sdfg = _pairwise_acc.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not any(isinstance(n, MatMul) for n, _ in sdfg.all_nodes_recursive())
    assert _square_transients(sdfg) == []
    n = 64
    x, mass, ax = np.random.rand(n), np.random.rand(n), np.zeros(n)
    sdfg(x=x, mass=mass, ax=ax, N=n)
    dx = x[None, :] - x[:, None]
    assert np.allclose(ax, (dx * (dx * dx + 0.01) ** (-1.5)) @ mass)


def test_a_gemm_keeps_its_library_node():
    assert _apply(_gemm.to_sdfg(simplify=True)) == 0


def test_a_matrix_with_another_reader_keeps_the_library_node():
    assert _apply(_matrix_read_twice.to_sdfg(simplify=True)) == 0


if __name__ == "__main__":
    test_a_gemv_over_a_private_elementwise_matrix_fuses_away_the_matrix()
    test_a_gemm_keeps_its_library_node()
    test_a_matrix_with_another_reader_keeps_the_library_node()
