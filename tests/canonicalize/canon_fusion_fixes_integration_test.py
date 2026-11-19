# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The fusion fixes of 2026-11-18 through the whole CPU recipe: ``canonicalize`` then ``finalize_for_target``.

Each fix has a kernel it fuses and the strongest kernel it must not fuse. Fused means the full-size intermediate is
gone (or, for a reduction, the reduction runs inside the map writing the result); every kernel is compared against
numpy whatever the verdict.
"""

import numpy as np

import dace
from dace import symbolic
from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFusionVertical
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.fuse_maps_at_shared_axes import FuseMapsAtSharedAxes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64)
M = dace.symbol("M", dace.int64)
H = dace.symbol("H", dace.int64)


def canon_cpu(program) -> dace.SDFG:
    return finalize_for_target(canonicalize(program.to_sdfg(simplify=True), target="cpu", validate=True), "cpu")


def canon_only(program) -> dace.SDFG:
    return canonicalize(program.to_sdfg(simplify=True), target="cpu", validate=True)


def refuses_both_fusions(program) -> bool:
    """Whether FuseMapsAtSharedAxes and MapFusionVertical both leave the simplified program unchanged."""
    sdfg = program.to_sdfg(simplify=True)
    return (
        FuseMapsAtSharedAxes().apply_pass(sdfg, {}) is None
        and sdfg.apply_transformations_repeated(MapFusionVertical) == 0
    )


def transients_of_size(sdfg: dace.SDFG, size) -> list[str]:
    return [
        name
        for sd in sdfg.all_sdfgs_recursive()
        for name, desc in sd.arrays.items()
        if desc.transient and isinstance(desc, dace.data.Array) and symbolic.equal(desc.total_size, size) is True
    ]


def top_level_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [
        n
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None
    ]


def reduces_inside_map_writing(sdfg: dace.SDFG, name: str) -> bool:
    """Whether a top-level map whose exit writes ``name`` holds a reduction (a WCR edge or a Reduce) in its scope."""
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry) or state.entry_node(node) is not None:
                continue
            exit_node = state.exit_node(node)
            writes = any(isinstance(e.dst, nodes.AccessNode) and e.dst.data == name for e in state.out_edges(exit_node))
            body = state.scope_subgraph(node, include_entry=False, include_exit=False)
            reduces = any(e.data.wcr is not None for e in body.edges()) or any(
                isinstance(n, Reduce) for n in body.nodes()
            )
            if writes and reduces:
                return True
    return False


# FuseMapsAtSharedAxes, axis read off the intermediate's index (4aa8b9969)


@dace.program
def canon_fusion_column_of_square(A: dace.float64[8, 8], B: dace.float64[8]):
    T = np.empty((8, 8), dtype=np.float64)
    for i, k in dace.map[0:8, 0:8]:
        T[i, k] = A[i, k] * 2.0
    for j in dace.map[0:8]:
        B[j] = T[3, j] + 1.0


@dace.program
def canon_fusion_row_and_column_of_square(A: dace.float64[8, 8], B: dace.float64[8]):
    T = np.empty((8, 8), dtype=np.float64)
    for i, k in dace.map[0:8, 0:8]:
        T[i, k] = A[i, k] * 2.0
    for j in dace.map[0:8]:
        B[j] = T[3, j] + T[j, 3]


def test_a_column_read_of_a_square_intermediate_fuses_through_the_pipeline():
    sdfg = canon_cpu(canon_fusion_column_of_square)
    assert transients_of_size(sdfg, 64) == []
    a = np.random.default_rng(0).random((8, 8))
    b = np.zeros(8)
    sdfg(A=a, B=b)
    assert np.allclose(b, 2.0 * a[3, :] + 1.0)


def test_a_row_and_column_read_keeps_the_square_intermediate():
    """No split axis makes both ``T[3, j]`` and ``T[j, 3]`` an element of one producer iteration: neither fusion
    applies and canonicalization keeps ``T``. Finalization may still recompute the elementwise ``T`` at each read."""
    assert refuses_both_fusions(canon_fusion_row_and_column_of_square)
    assert transients_of_size(canon_only(canon_fusion_row_and_column_of_square), 64) != []
    sdfg = canon_cpu(canon_fusion_row_and_column_of_square)
    a = np.random.default_rng(1).random((8, 8))
    b = np.zeros(8)
    sdfg(A=a, B=b)
    assert np.allclose(b, 2.0 * (a[3, :] + a[:, 3]))


# MapFusionVertical, parameters paired by the intermediate's index (40e531a57)


@dace.program
def canon_fusion_transposed_read(A: dace.float64[N, N], B: dace.float64[N, N]):
    T = np.empty((N, N), dtype=np.float64)
    for i, j in dace.map[0:N, 0:N]:
        T[i, j] = A[i, j] + 1.0
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = T[j, i] * 2.0


@dace.program
def canon_fusion_both_orientations(A: dace.float64[N, N], B: dace.float64[N, N]):
    T = np.empty((N, N), dtype=np.float64)
    for i, j in dace.map[0:N, 0:N]:
        T[i, j] = A[i, j] + 1.0
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = T[i, j] - T[j, i]


def test_a_transposed_read_of_a_square_intermediate_fuses_through_the_pipeline():
    sdfg = canon_cpu(canon_fusion_transposed_read)
    assert transients_of_size(sdfg, N * N) == []
    a = np.random.default_rng(2).random((7, 7))
    b = np.zeros((7, 7))
    sdfg(A=a, B=b, N=7)
    assert np.allclose(b, 2.0 * (a + 1.0).T)


def test_reading_both_orientations_keeps_the_square_intermediate():
    """``T[i, j]`` and ``T[j, i]`` in one iteration: neither fusion applies and canonicalization keeps ``T``.
    Finalization may still recompute the elementwise ``T`` at each read."""
    assert refuses_both_fusions(canon_fusion_both_orientations)
    assert transients_of_size(canon_only(canon_fusion_both_orientations), N * N) != []
    sdfg = canon_cpu(canon_fusion_both_orientations)
    a = np.random.default_rng(3).random((7, 7))
    b = np.zeros((7, 7))
    sdfg(A=a, B=b, N=7)
    assert np.allclose(b, (a + 1.0) - (a + 1.0).T)


# FuseMapsAtSharedAxes + MapFusionVertical, a reduction over the unshared axis fused in place (Task 1)


@dace.program
def canon_fusion_row_sums(A: dace.float64[N, M], B: dace.float64[N]):
    T = np.zeros(N, dtype=np.float64)
    for i, k in dace.map[0:N, 0:M]:
        T[i] += A[i, k]
    for i in dace.map[0:N]:
        B[i] = T[i] * 2.0


@dace.program
def canon_fusion_total_sum(A: dace.float64[N, M], B: dace.float64[N]):
    T = np.zeros(1, dtype=np.float64)
    for i, k in dace.map[0:N, 0:M]:
        T[0] += A[i, k]
    for i in dace.map[0:N]:
        B[i] = T[0] * 2.0


def test_row_sums_reduce_inside_the_map_that_scales_them():
    sdfg = canon_cpu(canon_fusion_row_sums)
    assert reduces_inside_map_writing(sdfg, "B")
    a = np.random.default_rng(4).random((10, 6))
    b = np.zeros(10)
    sdfg(A=a, B=b, N=10, M=6)
    assert np.allclose(b, 2.0 * a.sum(axis=1))


@dace.program
def canon_fusion_eight_row_sums(A: dace.float64[8, M], B: dace.float64[8]):
    T = np.zeros(8, dtype=np.float64)
    for i, k in dace.map[0:8, 0:M]:
        T[i] += A[i, k]
    for i in dace.map[0:8]:
        B[i] = T[i] * 2.0


def test_row_sums_over_a_small_kept_axis_stay_a_separate_reduction():
    """Eight rows: fused, the eight iterations cannot share out and each sum runs sequentially inside (19x slower on
    GH200), so the reduction keeps its own map."""
    sdfg = canon_cpu(canon_fusion_eight_row_sums)
    assert not reduces_inside_map_writing(sdfg, "B")
    a = np.random.default_rng(14).random((8, 50))
    b = np.zeros(8)
    sdfg(A=a, B=b, M=50)
    assert np.allclose(b, 2.0 * a.sum(axis=1))


def test_a_total_sum_completes_before_the_map_that_reads_it():
    sdfg = canon_cpu(canon_fusion_total_sum)
    assert not reduces_inside_map_writing(sdfg, "B")
    a = np.random.default_rng(5).random((10, 6))
    b = np.zeros(10)
    sdfg(A=a, B=b, N=10, M=6)
    assert np.allclose(b, 2.0 * a.sum())


# MapFusionVertical, an intermediate accessed again downstream (Task 1)


@dace.program
def canon_fusion_rewrite_after_reader(A: dace.float64[N], O: dace.float64[N], R: dace.float64[N]):
    T = A + 1.0
    O[:] = T * 2.0
    T += O
    R[:] = T


@dace.program
def canon_fusion_overwrite_after_reader(A: dace.float64[N], O: dace.float64[N], R: dace.float64[N]):
    T = A + 1.0
    O[:] = T * 2.0
    T[:] = O + 10.0
    R[:] = T


def test_an_intermediate_updated_after_its_reader_fuses_into_one_map():
    sdfg = canon_cpu(canon_fusion_rewrite_after_reader)
    assert len(top_level_maps(sdfg)) == 1
    a = np.random.default_rng(6).random(10)
    o, r = np.zeros(10), np.zeros(10)
    sdfg(A=a, O=o, R=r, N=10)
    assert np.allclose(o, 2.0 * (a + 1.0)) and np.allclose(r, 3.0 * (a + 1.0))


def test_an_intermediate_overwritten_after_its_reader_keeps_its_values():
    """The unit test pins the refusal; through the pipeline another fusion order may fuse it in place, so only the
    values are the contract here."""
    sdfg = canon_cpu(canon_fusion_overwrite_after_reader)
    a = np.random.default_rng(7).random(10)
    o, r = np.zeros(10), np.zeros(10)
    sdfg(A=a, O=o, R=r, N=10)
    assert np.allclose(o, 2.0 * (a + 1.0)) and np.allclose(r, 2.0 * (a + 1.0) + 10.0)


# find_parameter_remapping and the FuseMaps prefilter, one extent in two spellings (Task 1)


@dace.program
def canon_fusion_floor_spellings(A: dace.float64[(H - 1) // 2 + 1], B: dace.float64[(H - 1) // 2 + 1]):
    T = np.empty_like(A)
    T[:] = A + 1.0
    for i in dace.map[0 : (H - 1) // 2 + 1]:
        B[i] = T[i] * 2.0


@dace.program
def canon_fusion_other_floor(A: dace.float64[(H - 1) // 2 + 1], B: dace.float64[(H - 1) // 3 + 1]):
    T = np.empty_like(A)
    T[:] = A + 1.0
    for i in dace.map[0 : (H - 1) // 3 + 1]:
        B[i] = T[i] * 2.0


def test_one_extent_in_two_floor_spellings_fuses_through_the_pipeline():
    sdfg = canon_cpu(canon_fusion_floor_spellings)
    assert transients_of_size(sdfg, (H - 1) // 2 + 1) == []
    a = np.random.default_rng(8).random(5)
    b = np.zeros(5)
    sdfg(A=a, B=b, H=9)
    assert np.allclose(b, 2.0 * (a + 1.0))


def test_a_smaller_floor_keeps_the_intermediate():
    """The consumer runs over a third, the producer over a half: no renaming pairs the two ranges."""
    sdfg = canon_cpu(canon_fusion_other_floor)
    assert transients_of_size(sdfg, (H - 1) // 2 + 1) != []
    a = np.random.default_rng(9).random(5)
    b = np.zeros(3)
    sdfg(A=a, B=b, H=9)
    assert np.allclose(b, 2.0 * (a[:3] + 1.0))


if __name__ == "__main__":
    test_a_column_read_of_a_square_intermediate_fuses_through_the_pipeline()
    test_a_row_and_column_read_keeps_the_square_intermediate()
    test_a_transposed_read_of_a_square_intermediate_fuses_through_the_pipeline()
    test_reading_both_orientations_keeps_the_square_intermediate()
    test_row_sums_reduce_inside_the_map_that_scales_them()
    test_a_total_sum_completes_before_the_map_that_reads_it()
    test_an_intermediate_updated_after_its_reader_fuses_into_one_map()
    test_an_intermediate_overwritten_after_its_reader_keeps_its_values()
    test_one_extent_in_two_floor_spellings_fuses_through_the_pipeline()
    test_a_smaller_floor_keeps_the_intermediate()
