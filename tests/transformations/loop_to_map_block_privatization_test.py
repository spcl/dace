# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` on CloudSC's block loop: every block iteration writes its scratch arrays before it reads them.

Four shapes kept the block loop sequential, each a reproducer below:

1. A level recurrence (``ZQXNM1``, the values at the level above): a branch false on the first ``JK`` iteration
   reads what the previous ``JK`` iteration stored, so within one block nothing comes from the block before.
2. An in-place map body (``ZLDIFDT(JL)=...; ZLDIFDT(JL)=c*ZLDIFDT(JL)``): the NestedSDFG writes each element before
   reading it, and passes the container in only because it also passes it out.
3. A map that stores a whole row and reads part of it (``ZQXFG(JL,:)`` filled, ``ZQXFG(JL,1:2)`` read).
4. A dynamic in-place read of a slice indexed by the block (``PLUDE(:,JK,IBL)``): its subset bounds every element
   it can touch, so the dependence test runs on it instead of giving up.

The two negative tests pin that a guard true on the first iteration, or a body reading before writing, stays carried.
"""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap, read_before_written

N, M, F = (dace.symbol(s) for s in "NMF")


@dace.program
def level_above(A: dace.float64[N, M, F], B: dace.float64[N, M, F]):
    for b in range(N):
        prev = np.ndarray((F,), dtype=np.float64)
        for k in range(M):
            if k > 0:
                for l in range(F):
                    B[b, k, l] = prev[l] + A[b, k, l]
            for l in dace.map[0:F]:
                prev[l] = A[b, k, l]


@dace.program
def level_above_in_a_map_body(A: dace.float64[N, M, F], B: dace.float64[N, M, F], C: dace.float64[N, M, F]):
    for b in range(N):
        prev = np.ndarray((F,), dtype=np.float64)
        for k in range(M):
            for l in dace.map[0:F]:
                if k > 0:
                    B[b, k, l] = prev[l] + A[b, k, l]
            for l in dace.map[0:F]:
                prev[l] = A[b, k, l]
            for l in range(F):
                C[b, k, l] = prev[l]


@dace.program
def level_above_from_the_first_level(A: dace.float64[N, M, F], B: dace.float64[N, M, F]):
    for b in range(N):
        prev = np.ndarray((F,), dtype=np.float64)
        for k in range(M):
            if k < M:
                for l in range(F):
                    B[b, k, l] = prev[l] + A[b, k, l]
            for l in dace.map[0:F]:
                prev[l] = A[b, k, l]


@dace.program
def rewritten_in_place(A: dace.float64[N, M], B: dace.float64[N, M]):
    for b in range(N):
        t = np.ndarray((M,), dtype=np.float64)
        for j in dace.map[0:M]:
            t[j] = A[b, j]
            if A[b, j] > 0.5:
                t[j] = 2.0 * t[j]
        for j in dace.map[0:M]:
            B[b, j] = t[j]


@dace.program
def updated_in_place(A: dace.float64[N, M], B: dace.float64[N, M]):
    for b in range(N):
        t = np.ndarray((M,), dtype=np.float64)
        for j in dace.map[0:M]:
            if A[b, j] > 0.5:
                t[j] = 2.0 * t[j]
            t[j] = t[j] + A[b, j]
        for j in dace.map[0:M]:
            B[b, j] = t[j]


@dace.program
def row_then_part(A: dace.float64[N, M, F], B: dace.float64[N, M]):
    for b in range(N):
        t = np.ndarray((M, F), dtype=np.float64)
        for j in dace.map[0:M]:
            for l in range(F):
                t[j, l] = A[b, j, l]
            B[b, j] = t[j, 0] + t[j, 1]


@dace.program
def in_place_block(P: dace.float64[M, F, N], A: dace.float64[M, N]):
    for b in range(N):
        for k in range(F):
            for j in dace.map[0:M]:
                if A[j, b] > 0.5:
                    P[j, k, b] = P[j, k, b] * A[j, b]


def block_loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.parent_graph is sdfg)


def transients(sdfg: dace.SDFG) -> set[str]:
    return {name for name, desc in sdfg.arrays.items() if desc.transient}


def lifts(sdfg: dace.SDFG) -> bool:
    loop = block_loop(sdfg)
    xform = LoopToMap()
    xform.loop = loop
    return xform.can_be_applied(loop.parent_graph, 0, sdfg)


def lift_and_compare(program, arrays: dict[str, np.ndarray], symbols: dict[str, int]) -> None:
    """Lift the block loop, run both SDFGs on copies of ``arrays``, and compare every array."""
    sequential = program.to_sdfg(simplify=True)
    lifted = program.to_sdfg(simplify=True)
    lifted.name = f"{lifted.name}_lifted"
    assert read_before_written(block_loop(lifted), transients(lifted)) == set()
    assert lifts(lifted)
    loop = block_loop(lifted)
    xform = LoopToMap()
    xform.loop = loop
    xform.apply(loop.parent_graph, lifted)
    want = {name: value.copy() for name, value in arrays.items()}
    got = {name: value.copy() for name, value in arrays.items()}
    sequential(**want, **symbols)
    lifted(**got, **symbols)
    for name in arrays:
        assert np.allclose(got[name], want[name]), name


def test_a_level_recurrence_under_a_first_iteration_guard_stays_inside_the_block():
    rng = np.random.default_rng(0)
    lift_and_compare(level_above, {"A": rng.random((4, 5, 3)), "B": np.zeros((4, 5, 3))}, {"N": 4, "M": 5, "F": 3})


def test_a_level_recurrence_read_in_a_map_body_stays_inside_the_block():
    rng = np.random.default_rng(0)
    lift_and_compare(
        level_above_in_a_map_body,
        {"A": rng.random((4, 5, 3)), "B": np.zeros((4, 5, 3)), "C": np.zeros((4, 5, 3))},
        {"N": 4, "M": 5, "F": 3},
    )


def test_a_guard_true_on_the_first_iteration_reads_the_previous_block():
    sdfg = level_above_from_the_first_level.to_sdfg(simplify=True)
    assert "prev" in read_before_written(block_loop(sdfg), transients(sdfg))
    assert not lifts(sdfg)


def test_an_in_place_map_body_that_writes_first_reads_nothing_from_outside():
    rng = np.random.default_rng(1)
    lift_and_compare(rewritten_in_place, {"A": rng.random((4, 6)), "B": np.zeros((4, 6))}, {"N": 4, "M": 6})


def test_an_in_place_map_body_that_reads_first_is_carried():
    sdfg = updated_in_place.to_sdfg(simplify=True)
    assert "t" in read_before_written(block_loop(sdfg), transients(sdfg))
    assert not lifts(sdfg)


def test_a_row_stored_in_a_map_covers_a_part_read_in_it():
    rng = np.random.default_rng(2)
    lift_and_compare(row_then_part, {"A": rng.random((4, 6, 3)), "B": np.zeros((4, 6))}, {"N": 4, "M": 6, "F": 3})


def test_a_dynamic_in_place_read_indexed_by_the_block_is_tested_on_its_subset():
    rng = np.random.default_rng(3)
    lift_and_compare(in_place_block, {"P": rng.random((6, 3, 4)), "A": rng.random((6, 4))}, {"N": 4, "M": 6, "F": 3})


if __name__ == "__main__":
    test_a_level_recurrence_under_a_first_iteration_guard_stays_inside_the_block()
    test_a_level_recurrence_read_in_a_map_body_stays_inside_the_block()
    test_a_guard_true_on_the_first_iteration_reads_the_previous_block()
    test_an_in_place_map_body_that_writes_first_reads_nothing_from_outside()
    test_an_in_place_map_body_that_reads_first_is_carried()
    test_a_row_stored_in_a_map_covers_a_part_read_in_it()
    test_a_dynamic_in_place_read_indexed_by_the_block_is_tested_on_its_subset()
