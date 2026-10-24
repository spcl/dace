# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseMapsAtSharedAxes``: a producer map over ``(i, k)`` fuses into a consumer map over ``i`` that reads one
producer iteration row, as an inner map over ``k``; a consumer reading another row keeps the producer as it was."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.fuse_maps_at_shared_axes import FuseMapsAtSharedAxes

N = dace.symbol("N")


@dace.program
def row_then_columns(A: dace.float64[N, 8], B: dace.float64[N]):
    T = np.empty((N, 8), dtype=np.float64)
    for i, k in dace.map[0:N, 0:8]:
        T[i, k] = A[i, k] * 2.0
    for j in dace.map[0:N]:
        B[j] = T[j, 4] + T[j, 7]


@dace.program
def next_row(A: dace.float64[N, 8], B: dace.float64[N]):
    T = np.empty((N, 8), dtype=np.float64)
    for k, i in dace.map[0:8, 0:N]:
        T[i, k] = A[i, k] * 2.0
    for j in dace.map[0 : N - 1]:
        B[j] = T[j + 1, 4]


@dace.program
def two_refused_consumers(A: dace.float64[N, 8], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty((N, 8), dtype=np.float64)
    for i, k in dace.map[0:N, 0:8]:
        T[i, k] = A[i, k] * 2.0
    for j in dace.map[0:N]:
        B[j] = T[N - 1 - j, 4]
    for j in dace.map[0:N]:
        C[j] = T[N - 1 - j, 7]


def top_level_maps(sdfg: dace.SDFG) -> list[nodes.Map]:
    return [
        n.map
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None
    ]


def test_a_producer_over_more_axes_fuses_as_an_inner_map():
    sdfg = row_then_columns.to_sdfg(simplify=True)
    assert len(top_level_maps(sdfg)) == 2
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (outer,) = top_level_maps(sdfg)
    assert len(outer.params) == 1
    # The row the fused iteration produces replaces the full-extent intermediate.
    assert "T" not in sdfg.arrays and sdfg.arrays["__map_fusion_T"].total_size == 8
    a = np.random.default_rng(0).random((10, 8))
    b = np.zeros(10)
    sdfg(A=a, B=b, N=10)
    assert np.allclose(b, 2.0 * (a[:, 4] + a[:, 7]))


def test_a_consumer_reading_another_iteration_keeps_the_producer():
    sdfg = next_row.to_sdfg(simplify=True)
    before = [(list(m.params), str(m.range)) for m in top_level_maps(sdfg)]
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    sdfg.validate()
    assert [(list(m.params), str(m.range)) for m in top_level_maps(sdfg)] == before
    a = np.random.default_rng(1).random((10, 8))
    b = np.zeros(10)
    sdfg(A=a, B=b, N=10)
    assert np.allclose(b[:9], 2.0 * a[1:, 4])


def test_each_refused_consumer_sees_the_rebuilt_producer():
    """A refused try merges the producer back as a new map entry; the next consumer of the same producer must be
    matched against that entry, not the one the first try split (cp2k: ``KeyError`` on the stale entry)."""
    sdfg = two_refused_consumers.to_sdfg(simplify=True)
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    sdfg.validate()
    a = np.random.default_rng(2).random((10, 8))
    b, c = np.zeros(10), np.zeros(10)
    sdfg(A=a, B=b, C=c, N=10)
    assert np.allclose(b, 2.0 * a[::-1, 4]) and np.allclose(c, 2.0 * a[::-1, 7])


if __name__ == "__main__":
    test_a_producer_over_more_axes_fuses_as_an_inner_map()
    test_a_consumer_reading_another_iteration_keeps_the_producer()
    test_each_refused_consumer_sees_the_rebuilt_producer()
