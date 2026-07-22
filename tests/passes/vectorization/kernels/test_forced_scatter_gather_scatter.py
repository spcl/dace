# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Forced-parallel scatter / gather-scatter maps vectorize correctly (1-D and multi-dim).

The frontend ``dace.map`` makes each kernel a PARALLEL map up front -- the caller asserts the
index is injective (a permutation), so the map is data-parallel and needs NO ``LoopToMap`` with
the permissive flag to become one (the vectorizer must NOT rely on permissive ``LoopToMap`` to
parallelise a for-loop scatter). The tile path must then lower the per-lane index access to a
gather TileLoad / scatter TileStore and stay bit-exact with the un-vectorized reference.

Covers:
* pure scatter   ``a[idx[i]] = c[i]``
* gather-scatter ``a[idx[i]] = b[idx[i]] + c[i]``  (read AND write through the same per-lane idx)
* multi-dim gather-scatter ``a[idx[i, j]] = b[idx[i, j]] + c[i, j]`` (2-D parallel map, tiled).
"""
import numpy
import pytest

import dace
from tests.passes.vectorization.helpers.harness import N, X, Y, run_vectorization_test

pytestmark = pytest.mark.tile_nodes


@dace.program
def forced_scatter_store(a: dace.float64[N], idx: dace.int64[N], c: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        a[idx[i]] = c[i]


@dace.program
def forced_gather_scatter(a: dace.float64[N], b: dace.float64[N], idx: dace.int64[N], c: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        a[idx[i]] = b[idx[i]] + c[i]


@dace.program
def forced_gather_scatter_2d(a: dace.float64[Y, X], b: dace.float64[Y, X], idx: dace.int64[Y, X],
                             c: dace.float64[Y, X]):
    # Per-row gather-scatter: ``i`` is a direct tile dim, the inner index ``idx[i, j]`` scatters
    # along the X dim (each row of ``idx`` a permutation of ``0:X``), so the write is injective.
    for i, j in dace.map[0:Y:1, 0:X:1]:
        a[i, idx[i, j]] = b[i, idx[i, j]] + c[i, j]


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_forced_scatter_store(branch_mode, remainder_strategy):
    """Pure scatter through an injective (permutation) index -> scatter TileStore."""
    n = 60  # not a multiple of 8 -> exercises the remainder tile
    idx = numpy.random.permutation(n).astype(numpy.int64)
    run_vectorization_test(
        dace_func=forced_scatter_store,
        arrays={
            "a": numpy.zeros(n),
            "idx": idx,
            "c": numpy.random.random(n),
        },
        params={"N": n},
        vector_width=8,
        sdfg_name="forced_scatter_store",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_forced_gather_scatter(branch_mode, remainder_strategy):
    """Gather-scatter ``a[idx[i]] = b[idx[i]] + c[i]`` -> gather TileLoad + scatter TileStore."""
    n = 60  # not a multiple of 8 -> exercises the remainder tile
    idx = numpy.random.permutation(n).astype(numpy.int64)
    run_vectorization_test(
        dace_func=forced_gather_scatter,
        arrays={
            "a": numpy.zeros(n),
            "b": numpy.random.random(n),
            "idx": idx,
            "c": numpy.random.random(n),
        },
        params={"N": n},
        vector_width=8,
        sdfg_name="forced_gather_scatter",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_forced_gather_scatter_2d(branch_mode, remainder_strategy):
    """Multi-dim (2-D parallel map) per-row gather-scatter; both remainder strategies."""
    yv, xv = 8, 60  # xv not a multiple of 8 -> exercises the remainder tile
    idx = numpy.stack([numpy.random.permutation(xv) for _ in range(yv)]).astype(numpy.int64)
    run_vectorization_test(
        dace_func=forced_gather_scatter_2d,
        arrays={
            "a": numpy.zeros((yv, xv)),
            "b": numpy.random.random((yv, xv)),
            "idx": idx,
            "c": numpy.random.random((yv, xv)),
        },
        params={"Y": yv, "X": xv},
        vector_width=8,
        sdfg_name="forced_gather_scatter_2d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )
