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


@dace.program
def column_of_a_square(A: dace.float64[8, 8], B: dace.float64[8]):
    T = np.empty((8, 8), dtype=np.float64)
    for i, k in dace.map[0:8, 0:8]:
        T[i, k] = A[i, k] * 2.0
    for j in dace.map[0:8]:
        B[j] = T[3, j] + T[5, j]


M = dace.symbol("M")


@dace.program
def per_particle_then_taps(x: dace.float64[N], out: dace.float64[4, N]):
    w = np.empty(N, dtype=np.float64)
    for p in dace.map[0:N]:
        w[p] = x[p] * 2.0
    for t, p in dace.map[0:4, 0:N]:
        out[t, p] = w[p] + t


@dace.program
def outer_axis_of_equal_rank(A: dace.float64[N, 8], C: dace.float64[8], B: dace.float64[N, 8]):
    T = np.empty((N, 8), dtype=np.float64)
    for i, k in dace.map[0:N, 0:8]:
        T[i, k] = A[i, k] * 2.0
    for i, j in dace.map[0:N, 0:8]:
        B[i, j] = T[i, 0] * C[j]


@dace.program
def row_sums_then_scale(A: dace.float64[N, M], B: dace.float64[N]):
    T = np.zeros(N, dtype=np.float64)
    for i, k in dace.map[0:N, 0:M]:
        T[i] += A[i, k]
    for i in dace.map[0:N]:
        B[i] = T[i] * 2.0


@dace.program
def total_sum_then_scale(A: dace.float64[N, M], B: dace.float64[N]):
    T = np.zeros(1, dtype=np.float64)
    for i, k in dace.map[0:N, 0:M]:
        T[0] += A[i, k]
    for i in dace.map[0:N]:
        B[i] = T[0] * 2.0


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


def test_the_fused_map_drops_a_thread_block_sized_for_the_split_axes():
    """A block chosen for the producer's two axes is stale for the fused map over one; the kernel's block is
    chosen again, and a refused split keeps its block."""
    sdfg = row_then_columns.to_sdfg(simplify=True)
    producer = next(m for m in top_level_maps(sdfg) if len(m.params) == 2)
    producer.gpu_block_size = [32, 8, 1]
    FuseMapsAtSharedAxes().apply_pass(sdfg, {})
    (outer,) = top_level_maps(sdfg)
    assert outer.gpu_block_size is None

    sdfg = next_row.to_sdfg(simplify=True)
    producer = next(m for m in top_level_maps(sdfg) if len(m.params) == 2)
    producer.gpu_block_size = [32, 8, 1]
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    assert next(m for m in top_level_maps(sdfg) if len(m.params) == 2).gpu_block_size == [32, 8, 1]


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


def test_the_split_axis_is_the_one_the_consumer_indexes_when_two_share_its_range():
    """Both producer axes iterate ``0:8``; the consumer reads ``T[3, j]``, so ``k`` (not the first equal range ``i``)
    is the shared axis."""
    sdfg = column_of_a_square.to_sdfg(simplify=True)
    assert len(top_level_maps(sdfg)) == 2
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (outer,) = top_level_maps(sdfg)
    assert len(outer.params) == 1
    assert "T" not in sdfg.arrays and sdfg.arrays["__map_fusion_T"].total_size == 8
    a = np.random.default_rng(3).random((8, 8))
    b = np.zeros(8)
    sdfg(A=a, B=b)
    assert np.allclose(b, 2.0 * (a[3, :] + a[5, :]))


def top_level_entries(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [
        n
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None
    ]


def test_a_consumer_sharing_an_inner_axis_keeps_its_order_and_both_maps_get_a_hint():
    """User ruling: canon does not interchange the consumer to fuse; it records the alternative on both maps."""
    sdfg = per_particle_then_taps.to_sdfg(simplify=True)
    before = [(list(m.params), str(m.range)) for m in top_level_maps(sdfg)]
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    sdfg.validate()
    assert [(list(m.params), str(m.range)) for m in top_level_maps(sdfg)] == before
    hints = [e.specialization_hint or "" for e in top_level_entries(sdfg)]
    assert len(hints) == 2 and all("fusion: shares (" in h and "at an inner axis" in h and "alt: " in h for h in hints)
    FuseMapsAtSharedAxes().apply_pass(sdfg, {})
    assert [e.specialization_hint for e in top_level_entries(sdfg)] == hints, "a rerun records the hint once"
    x = np.random.default_rng(4).random(10)
    out = np.zeros((4, 10))
    sdfg(x=x, out=out, N=10)
    assert np.allclose(out, 2.0 * x[None, :] + np.arange(4.0)[:, None])


def test_an_existing_hint_keeps_its_line_and_the_fusion_note_follows():
    sdfg = per_particle_then_taps.to_sdfg(simplify=True)
    consumer = next(e for e in top_level_entries(sdfg) if len(e.map.params) == 2)
    consumer.specialization_hint = "parallel"
    FuseMapsAtSharedAxes().apply_pass(sdfg, {})
    first, second = consumer.specialization_hint.split("\n")
    assert first == "parallel" and second.startswith("fusion: shares (")


def test_maps_of_equal_rank_sharing_only_an_outer_axis_stay_apart_without_a_hint():
    sdfg = outer_axis_of_equal_rank.to_sdfg(simplify=True)
    before = [(list(m.params), str(m.range)) for m in top_level_maps(sdfg)]
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    assert [(list(m.params), str(m.range)) for m in top_level_maps(sdfg)] == before
    assert all(not e.specialization_hint for e in top_level_entries(sdfg))


def test_a_reduction_over_the_unshared_axis_completes_inside_the_fused_iteration():
    """``T[i] += A[i, k]`` split at ``i``: each outer iteration reduces its own element over the inner ``k`` map and
    reads it after, in place in ``T`` (its zero start value lives there, not in a buffer)."""
    sdfg = row_sums_then_scale.to_sdfg(simplify=True)
    before = len(top_level_maps(sdfg))
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert len(top_level_maps(sdfg)) == before - 1
    assert "__map_fusion_T" not in sdfg.arrays
    a = np.random.default_rng(5).random((10, 6))
    b = np.zeros(10)
    sdfg(A=a, B=b, N=10, M=6)
    assert np.allclose(b, 2.0 * a.sum(axis=1))


def test_a_reduction_every_iteration_contributes_to_keeps_the_producer():
    """``T[0]`` collects all ``i``: no fused iteration holds the finished sum."""
    sdfg = total_sum_then_scale.to_sdfg(simplify=True)
    before = sorted((list(m.params), str(m.range)) for m in top_level_maps(sdfg))
    assert FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
    sdfg.validate()
    assert sorted((list(m.params), str(m.range)) for m in top_level_maps(sdfg)) == before
    a = np.random.default_rng(6).random((10, 6))
    b = np.zeros(10)
    sdfg(A=a, B=b, N=10, M=6)
    assert np.allclose(b, 2.0 * a.sum())


if __name__ == "__main__":
    test_a_producer_over_more_axes_fuses_as_an_inner_map()
    test_the_fused_map_drops_a_thread_block_sized_for_the_split_axes()
    test_a_consumer_reading_another_iteration_keeps_the_producer()
    test_each_refused_consumer_sees_the_rebuilt_producer()
    test_the_split_axis_is_the_one_the_consumer_indexes_when_two_share_its_range()
    test_a_consumer_sharing_an_inner_axis_keeps_its_order_and_both_maps_get_a_hint()
    test_an_existing_hint_keeps_its_line_and_the_fusion_note_follows()
    test_maps_of_equal_rank_sharing_only_an_outer_axis_stay_apart_without_a_hint()
    test_a_reduction_over_the_unshared_axis_completes_inside_the_fused_iteration()
    test_a_reduction_every_iteration_contributes_to_keeps_the_producer()
