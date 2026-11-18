# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NumPy kernels through canonicalization and the CPU finalization, compared against NumPy: a transient the fixes
remove, shrink, forward or bound goes, and one a counterexample needs stays."""

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.sdfg.sdfg import interstate_bound_names
from dace.transformation.passes.bound_data_dependent_extents import BoundDataDependentExtents
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64)
K = dace.symbol("K", dace.int64)
R = dace.symbol("R", dace.int64)
W = dace.symbol("W", dace.int64)
T = dace.symbol("T", dace.int64)
#: ``N`` for which ``(N - 7) // 2 + 1`` is 8.
SIZE = 21


def canonical(prog, name: str, target: str = "cpu") -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target=target)
    return sdfg


def finalized(sdfg: dace.SDFG) -> dace.SDFG:
    return finalize_for_target(sdfg, "cpu", validate=True)


def transient_extents(sdfg: dace.SDFG) -> list[tuple[str, ...]]:
    """The shape of every transient array (no View, no single element) anywhere in ``sdfg``."""
    return [
        tuple(str(extent) for extent in desc.shape)
        for nested in sdfg.all_sdfgs_recursive()
        for desc in nested.arrays.values()
        if desc.transient
        and isinstance(desc, dt.Array)
        and not isinstance(desc, dt.View)
        and any(str(extent) != "1" for extent in desc.shape)
    ]


def data_sized(sdfg: dace.SDFG) -> list[str]:
    """The transients whose extent names a symbol an interstate edge of their SDFG assigns."""
    return [
        name
        for nested in sdfg.all_sdfgs_recursive()
        for name, desc in nested.arrays.items()
        if desc.transient
        and isinstance(desc, dt.Array)
        and not isinstance(desc, dt.View)
        and {str(s) for s in desc.free_symbols} & interstate_bound_names(nested)
    ]


# --- BoundDataDependentExtents: an in-bounds access of an argument bounds a data-read extent ----------------------


@dace.program
def rows_bounded_by_argument(counts: dace.int64[R], nl: dace.float64[R, W], out: dace.float64[R]):
    for l in range(R):
        count = counts[l]
        pointers = np.empty(1 + count)
        pointers[0] = 1.0
        pointers[1:] = nl[l, 0:count]
        out[l] = np.sum(pointers)


@dace.program
def rows_guarded_by_other_trip(
    counts: dace.int64[R], trips: dace.int64[R], nl: dace.float64[R, W], out: dace.float64[R]
):
    for l in range(R):
        count = counts[l]
        trip = trips[l]
        pointers = np.empty(1 + count)
        pad = np.empty(trip)
        pointers[:] = 1.0
        for j in range(trip):
            pad[j] = 0.0
            pointers[1:] = nl[l, 0:count]
        out[l] = np.sum(pointers) + np.sum(pad)


def bounded_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"counts": np.array([2, 3, 0, 1], dtype=np.int64), "nl": rng.random((4, 3)), "out": np.zeros(4)}


def test_an_access_of_an_argument_bounds_a_data_read_extent():
    sdfg = canonical(rows_bounded_by_argument, "canon_fix_rows_bounded_by_argument")
    assert data_sized(sdfg), "canonicalization kept no data-sized buffer to bound"
    BoundDataDependentExtents().apply_pass(sdfg, {})
    assert not data_sized(sdfg), data_sized(sdfg)
    finalized(sdfg)
    args = bounded_inputs()
    expected = np.array([1.0 + args["nl"][l, : args["counts"][l]].sum() for l in range(4)])
    sdfg(**args, R=4, W=3)
    np.testing.assert_allclose(args["out"], expected)


def test_an_access_under_another_buffers_trip_count_bounds_nothing():
    """Row 1 runs no trip and counts 5 > W: only the copy in the loop would have bounded ``count``."""
    sdfg = canonical(rows_guarded_by_other_trip, "canon_fix_rows_guarded_by_other_trip")
    before = data_sized(sdfg)
    BoundDataDependentExtents().apply_pass(sdfg, {})
    after = data_sized(sdfg)
    assert before and sorted(after) == sorted(before), (before, after)
    finalized(sdfg)
    rng = np.random.default_rng(1)
    counts, trips = np.array([2, 5, 3, 1], dtype=np.int64), np.array([2, 0, 1, 3], dtype=np.int64)
    nl, out = rng.random((4, 3)), np.zeros(4)
    expected = np.array(
        [1.0 + nl[l, : counts[l]].sum() if trips[l] > 0 else 1.0 + counts[l] for l in range(4)], dtype=np.float64
    )
    sdfg(counts=counts, trips=trips, nl=nl, out=out, R=4, W=3)
    np.testing.assert_allclose(out, expected)


@pytest.mark.gpu
def test_the_offload_bounds_a_data_read_extent_by_an_argument_extent():
    import cupy

    from dace.transformation.passes.canonicalize.finalize import offload_to_gpu

    sdfg = canonical(rows_bounded_by_argument, "canon_fix_rows_bounded_by_argument_gpu", target="gpu")
    offload_to_gpu(sdfg)
    assert not data_sized(sdfg), data_sized(sdfg)
    finalize_for_target(sdfg, "gpu", validate=True)
    args = bounded_inputs()
    expected = np.array([1.0 + args["nl"][l, : args["counts"][l]].sum() for l in range(4)])
    given = {k: cupy.asarray(v) for k, v in args.items()}
    sdfg(**given, R=4, W=3)
    np.testing.assert_allclose(cupy.asnumpy(given["out"]), expected)


# --- RedundantArray: an extent spelled floor (annotation) and int_floor (body) --------------------------------------


@dace.program
def halved_copy(a: dace.float64[(N - 7) // 2 + 1], b: dace.float64[(N - 7) // 2 + 1]):
    t = np.empty((N - 7) // 2 + 1)
    for i in dace.map[0 : (N - 7) // 2 + 1]:
        t[i] = a[i] * 2.0
    b[:] = t


@dace.program
def halved_snapshot(a: dace.float64[(N - 7) // 2 + 1], b: dace.float64[(N - 7) // 2 + 1]):
    t = np.empty((N - 7) // 2 + 1)
    for i in dace.map[0 : (N - 7) // 2 + 1]:
        t[i] = a[i]
    for i in dace.map[0 : (N - 7) // 2 + 1]:
        a[i] = 0.0
    for i in dace.map[0 : (N - 7) // 2 + 1]:
        b[i] = t[(N - 7) // 2 - i] + a[i]


def test_a_copy_spelled_two_ways_leaves_no_buffer():
    sdfg = finalized(canonical(halved_copy, "canon_fix_halved_copy"))
    assert not transient_extents(sdfg), transient_extents(sdfg)
    a, b = np.random.default_rng(2).random(8), np.zeros(8)
    sdfg(a=a, b=b, N=SIZE)
    np.testing.assert_allclose(b, 2.0 * a)


def test_a_snapshot_spelled_two_ways_keeps_its_buffer():
    """``a`` is zeroed before ``t`` is read reversed: ``t`` holds the values ``a`` had."""
    sdfg = finalized(canonical(halved_snapshot, "canon_fix_halved_snapshot"))
    assert transient_extents(sdfg), "the snapshot of a was folded into a"
    a, b = np.random.default_rng(3).random(8), np.zeros(8)
    expected = a[::-1].copy()
    sdfg(a=a, b=b, N=SIZE)
    np.testing.assert_allclose(b, expected)
    np.testing.assert_allclose(a, 0.0)


# --- ForwardOffsetCopies: a shifted copy read in a later loop or branch ---------------------------------------------


@dace.program
def shifted_reads_in_loop(q: dace.float64[N + 1], out: dace.float64[N]):
    t = q[1 : N + 1].copy()
    for k in range(T):
        for i in dace.map[0:N]:
            out[i] = out[i] + t[N - 1 - i] * 2.0


@dace.program
def shifted_reads_in_loop_writing_source(q: dace.float64[N + 1], out: dace.float64[N]):
    t = q[1 : N + 1].copy()
    for k in range(T):
        for i in dace.map[0:N]:
            out[i] = out[i] + t[N - 1 - i] * 2.0
        for i in dace.map[0:N]:
            q[i + 1] = q[i + 1] + 1.0


def test_a_shifted_copy_read_in_a_later_loop_is_forwarded():
    sdfg = finalized(canonical(shifted_reads_in_loop, "canon_fix_shifted_loop"))
    assert ("N",) not in transient_extents(sdfg), transient_extents(sdfg)
    q, out = np.random.default_rng(4).random(9), np.zeros(8)
    sdfg(q=q, out=out, N=8, T=3)
    np.testing.assert_allclose(out, 6.0 * q[1:][::-1])


def test_a_loop_writing_the_source_keeps_the_shifted_copy():
    sdfg = finalized(canonical(shifted_reads_in_loop_writing_source, "canon_fix_shifted_loop_writer"))
    assert ("N",) in transient_extents(sdfg), transient_extents(sdfg)
    q, out = np.random.default_rng(5).random(9), np.zeros(8)
    expected_out, expected_q = 6.0 * q[1:][::-1], q.copy()
    expected_q[1:] += 3.0
    sdfg(q=q, out=out, N=8, T=3)
    np.testing.assert_allclose(out, expected_out)
    np.testing.assert_allclose(q, expected_q)


@dace.program
def shifted_reads_in_branch(q: dace.float64[N + 1], out: dace.float64[N], flag: dace.int64):
    t = q[1 : N + 1].copy()
    if flag > 0:
        for i in dace.map[0:N]:
            out[i] = t[N - 1 - i] * 2.0
    else:
        for i in dace.map[0:N]:
            out[i] = t[N - 1 - i] * 3.0


@dace.program
def shifted_reads_in_branch_writing_source(q: dace.float64[N + 1], out: dace.float64[N], flag: dace.int64):
    t = q[1 : N + 1].copy()
    if flag > 0:
        for i in dace.map[0 : N + 1]:
            q[i] = 0.0
        for i in dace.map[0:N]:
            out[i] = t[N - 1 - i] * 2.0
    else:
        for i in dace.map[0:N]:
            out[i] = t[N - 1 - i] * 3.0


def test_a_shifted_copy_read_in_a_branch_is_forwarded():
    sdfg = finalized(canonical(shifted_reads_in_branch, "canon_fix_shifted_branch"))
    assert ("N",) not in transient_extents(sdfg), transient_extents(sdfg)
    for flag, factor in ((1, 2.0), (0, 3.0)):
        q, out = np.random.default_rng(6 + flag).random(9), np.zeros(8)
        sdfg(q=q, out=out, flag=np.int64(flag), N=8)
        np.testing.assert_allclose(out, factor * q[1:][::-1])


def test_a_branch_writing_the_source_keeps_the_shifted_copy():
    sdfg = finalized(canonical(shifted_reads_in_branch_writing_source, "canon_fix_shifted_branch_writer"))
    assert ("N",) in transient_extents(sdfg), transient_extents(sdfg)
    q, out = np.random.default_rng(8).random(9), np.zeros(8)
    expected = 2.0 * q[1:][::-1]
    sdfg(q=q, out=out, flag=np.int64(1), N=8)
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(q, 0.0)


# --- ShrinkMapLocalTransients: the dimensions every access in one map body shares ----------------------------------


@dace.program
def row_scratch(A: dace.float64[N, K], B: dace.float64[N, K]):
    t = np.empty((N, K))
    for i in dace.map[0:N]:
        for k in dace.map[0:K]:
            t[i, k] = A[i, k] * 2.0
        for k in dace.map[0:K]:
            B[i, k] = t[i, K - 1 - k] + 1.0


@dace.program
def row_scratch_read_after(A: dace.float64[N, K], B: dace.float64[N, K], C: dace.float64[N, K]):
    t = np.empty((N, K))
    for i in dace.map[0:N]:
        for k in dace.map[0:K]:
            t[i, k] = A[i, k] * 2.0
        for k in dace.map[0:K]:
            B[i, k] = t[i, K - 1 - k] + 1.0
    C[:] = t


def test_a_row_scratch_keeps_no_full_extent_buffer():
    sdfg = finalized(canonical(row_scratch, "canon_fix_row_scratch"))
    assert ("N", "K") not in transient_extents(sdfg), transient_extents(sdfg)
    A, B = np.random.default_rng(9).random((7, 5)), np.zeros((7, 5))
    sdfg(A=A, B=B, N=7, K=5)
    np.testing.assert_allclose(B, 2.0 * A[:, ::-1] + 1.0)


def test_a_row_scratch_read_after_the_map_keeps_every_row():
    """``C = t`` after the map reads every row at once: no buffer may be cut to one row."""
    sdfg = finalized(canonical(row_scratch_read_after, "canon_fix_row_scratch_read_after"))
    assert all(len(shape) != 2 or shape[0] != "1" for shape in transient_extents(sdfg)), transient_extents(sdfg)
    A, B, C = np.random.default_rng(10).random((7, 5)), np.zeros((7, 5)), np.zeros((7, 5))
    sdfg(A=A, B=B, C=C, N=7, K=5)
    np.testing.assert_allclose(B, 2.0 * A[:, ::-1] + 1.0)
    np.testing.assert_allclose(C, 2.0 * A)


# --- RemoveViews / RemoveSliceView: a whole-array view spelled int_floor next to a library node --------------------


@dace.program
def sum_of_whole_view(P: dace.float64[(N - 7) // 2 + 1, 2], out: dace.float64[1]):
    out[0] = np.sum(P[0 : (N - 7) // 2 + 1, 0:2])


@dace.program
def sum_of_partial_view(P: dace.float64[(N - 7) // 2 + 1, 2], out: dace.float64[1]):
    out[0] = np.sum(P[0 : (N - 7) // 2, 0:2])


def views(sdfg: dace.SDFG) -> list[str]:
    return [
        name
        for nested in sdfg.all_sdfgs_recursive()
        for state in nested.states()
        for name in (n.data for n in state.data_nodes())
        if isinstance(nested.arrays[name], dt.View)
    ]


def test_a_whole_view_spelled_int_floor_is_removed():
    sdfg = finalized(canonical(sum_of_whole_view, "canon_fix_sum_of_whole_view"))
    assert not views(sdfg), views(sdfg)
    P, out = np.random.default_rng(11).random((8, 2)), np.zeros(1)
    sdfg(P=P, out=out, N=SIZE)
    np.testing.assert_allclose(out[0], P.sum())


def test_a_view_one_row_short_reads_one_row_less():
    """``P[0:(N - 7) // 2]`` is one row short of ``P``: not the identity, whatever spells the extents."""
    sdfg = finalized(canonical(sum_of_partial_view, "canon_fix_sum_of_partial_view"))
    P, out = np.random.default_rng(12).random((8, 2)), np.zeros(1)
    sdfg(P=P, out=out, N=SIZE)
    np.testing.assert_allclose(out[0], P[:7].sum())


if __name__ == "__main__":
    test_an_access_of_an_argument_bounds_a_data_read_extent()
    test_an_access_under_another_buffers_trip_count_bounds_nothing()
    test_a_copy_spelled_two_ways_leaves_no_buffer()
    test_a_snapshot_spelled_two_ways_keeps_its_buffer()
    test_a_shifted_copy_read_in_a_later_loop_is_forwarded()
    test_a_loop_writing_the_source_keeps_the_shifted_copy()
    test_a_shifted_copy_read_in_a_branch_is_forwarded()
    test_a_branch_writing_the_source_keeps_the_shifted_copy()
    test_a_row_scratch_keeps_no_full_extent_buffer()
    test_a_row_scratch_read_after_the_map_keeps_every_row()
    test_a_whole_view_spelled_int_floor_is_removed()
    test_a_view_one_row_short_reads_one_row_less()
