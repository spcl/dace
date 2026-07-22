# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map that REDUCES (possibly multi-dim) into a SCALAR must still vectorize -- expand the
accumulator to a per-lane partial-sum tile and fold with a horizontal ``TileReduce``.

This is the shape ``PrivatizeSequentialMapReductionAccumulator`` leaves behind: a sequential
reduction's loop-carried scalar accumulator is privatized, yielding a map whose body reduces into
a scalar. The multi-dim tile vectorizer must lower that to lane-parallel partial accumulation +
one final ``TileReduce`` and stay bit-exact with the un-vectorized reference.

Covers:
* 1-D sum   ``s += a[i]``
* 2-D sum   ``s += a[i, j]``            (multi-dim reduction into one scalar)
* 1-D fused ``s += a[i] * b[i]``        (dot-product: reduce a compute, not a bare load)
"""
import numpy
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy, BranchMode
from tests.passes.vectorization.helpers.harness import N, X, Y, run_vectorization_test

pytestmark = pytest.mark.tile_nodes


@dace.program
def sum_1d(a: dace.float64[N], s: dace.float64[1]):
    for i, in dace.map[0:N:1]:
        s[0] += a[i]


@dace.program
def sum_2d(a: dace.float64[Y, X], s: dace.float64[1]):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        s[0] += a[i, j]


@dace.program
def dot_1d(a: dace.float64[N], b: dace.float64[N], s: dace.float64[1]):
    for i, in dace.map[0:N:1]:
        s[0] += a[i] * b[i]


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_sum_1d_reduction(branch_mode, remainder_strategy):
    """1-D reduction into a scalar -> per-lane partial sums + one horizontal TileReduce."""
    n = 60  # not a multiple of 8 -> remainder tile
    run_vectorization_test(
        dace_func=sum_1d,
        arrays={"a": numpy.random.random(n), "s": numpy.zeros(1)},
        params={"N": n},
        vector_width=8,
        sdfg_name="sum_1d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_sum_2d_reduction_into_scalar(branch_mode, remainder_strategy):
    """Multi-dim (2-D) reduction into ONE scalar -- the seq-map-privatized-accumulator shape."""
    yv, xv = 8, 60
    run_vectorization_test(
        dace_func=sum_2d,
        arrays={"a": numpy.random.random((yv, xv)), "s": numpy.zeros(1)},
        params={"Y": yv, "X": xv},
        vector_width=8,
        sdfg_name="sum_2d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )


def _vectorize_and_check_2d_reduction(widths):
    """Vectorize the 2-D reduction ``s[0] += a[i, j]`` at the given tile ``widths`` (K=1 tiles the
    innermost dim only; K=2 tiles both dims into an 8x8 partial-sum tile) and assert the compiled
    result equals ``a.sum()``. Isolated from :func:`run_vectorization_test` so the tile K is forced
    explicitly instead of auto-derived from the collapsed dimensionality."""
    import contextlib
    import io
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
    yv, xv = 16, 24
    a = numpy.random.random((yv, xv))
    ref = a.sum()
    sdfg = sum_2d.to_sdfg(simplify=True)
    sdfg.name = f"sum2d_k{len(widths)}"
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=widths, target_isa=ISA.AVX512, remainder_strategy=RemainderStrategy.MASKED_TAIL,
                        branch_mode=BranchMode.MERGE)).apply_pass(sdfg, {})
    sdfg.validate()
    for s in {str(x) for x in sdfg.free_symbols}:
        if s not in sdfg.symbols:
            sdfg.add_symbol(s, dace.int64)
    got = numpy.zeros(1)
    with contextlib.redirect_stdout(io.StringIO()):
        sdfg.compile()(a=a.copy(), s=got, Y=yv, X=xv)
    assert numpy.allclose(got[0], ref, rtol=1e-12, atol=1e-12), f"widths={widths}: {got[0]} vs {ref}"


def test_sum_2d_reduction_1d_tiling():
    """Multi-dim reduction into a scalar, tiled K=1 (innermost dim only) -> partial sums + TileReduce."""
    _vectorize_and_check_2d_reduction((8, ))


def test_sum_2d_reduction_2d_tiling():
    """Multi-dim reduction into a scalar, tiled K=2 (both dims -> 8x8 partial-sum tile) -> TileReduce."""
    _vectorize_and_check_2d_reduction((8, 8))


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_dot_1d_reduction(branch_mode, remainder_strategy):
    """Dot product ``s += a[i] * b[i]`` -- reduce a per-lane product, not a bare load."""
    n = 60
    run_vectorization_test(
        dace_func=dot_1d,
        arrays={"a": numpy.random.random(n), "b": numpy.random.random(n), "s": numpy.zeros(1)},
        params={"N": n},
        vector_width=8,
        sdfg_name="dot_1d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )
