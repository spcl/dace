# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LASTPRIVATE of a slot at a loop-invariant symbolic index: ``PromoteConstantIndexAccess(last_write=True)`` expands
``a[i]`` of a ``j`` loop per iteration, copies the last iteration's value back, and ``LoopToMap`` maps the loop."""

import json

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess

N = dace.symbol("N")
M = dace.symbol("M")


def promote_and_lift(sdfg: dace.SDFG) -> tuple[set[str], set[str]]:
    """Promote, lift every loop LoopToMap accepts, and return the loop variables left and the map parameters."""
    PromoteConstantIndexAccess(last_write=True).apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.apply_transformations_repeated(LoopToMap())
    loops = {r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)}
    params = {p for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) for p in n.map.params}
    return loops, params


@dace.program
def invariant_slot_live_after(a: dace.float64[N], aa: dace.float64[M, N], bb: dace.float64[M, N]):
    for i in range(1, N):
        for j in range(M):
            a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


def invariant_slot_reference(a: np.ndarray, aa: np.ndarray, bb: np.ndarray) -> None:
    for i in range(1, a.shape[0]):
        for j in range(aa.shape[0]):
            a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


def run_invariant_slot(rows: int) -> tuple[set[str], set[str]]:
    sdfg = invariant_slot_live_after.to_sdfg(simplify=True)
    loops, params = promote_and_lift(sdfg)
    rng = np.random.default_rng(0)
    a, aa, bb = rng.random(9), rng.random((rows, 9)), rng.random((rows, 9))
    ref = (a.copy(), aa.copy(), bb.copy())
    invariant_slot_reference(*ref)
    sdfg(a=a, aa=aa, bb=bb, N=9, M=rows)
    assert np.allclose(a, ref[0]) and np.allclose(aa, ref[1])
    return loops, params


# Reproducer: LoopToMap refused NON_UNIQUE_WRITE on a[i] in TSVC s257's j loop (lastprivate of a[i], last j wins)
def test_invariant_slot_read_in_iteration_and_live_after_loop_maps():
    loops, params = run_invariant_slot(rows=7)
    assert "j" in params and "j" not in loops
    # a[i] = ... - a[i - 1] still carries over i
    assert "i" in loops


# Reproducer: the copy-out of a zero-trip loop must leave a[i] at its pre-loop value
def test_zero_trip_loop_keeps_slot_unchanged():
    loops, params = run_invariant_slot(rows=0)
    assert "j" in params


@dace.program
def partial_store(a: dace.float64[N], aa: dace.float64[M, N], bb: dace.float64[M, N]):
    for i in range(1, N):
        for j in range(M):
            if bb[j, i] > 0.5:
                a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


def partial_store_reference(a: np.ndarray, aa: np.ndarray, bb: np.ndarray) -> None:
    for i in range(1, a.shape[0]):
        for j in range(aa.shape[0]):
            if bb[j, i] > 0.5:
                a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


# Reproducer: a store some iterations skip leaves the next iteration reading the previous one's a[i]: no lastprivate
def test_conditional_store_not_covering_every_iteration_stays_sequential():
    sdfg = partial_store.to_sdfg(simplify=True)
    before = json.dumps(sdfg.to_json(), sort_keys=True, default=str)
    assert PromoteConstantIndexAccess(last_write=True).apply_pass(sdfg, {}) is None
    assert json.dumps(sdfg.to_json(), sort_keys=True, default=str) == before
    loops, params = promote_and_lift(sdfg)
    assert "j" in loops and "j" not in params
    rng = np.random.default_rng(1)
    a, aa, bb = rng.random(9), rng.random((7, 9)), rng.random((7, 9))
    ref = (a.copy(), aa.copy(), bb.copy())
    partial_store_reference(*ref)
    sdfg(a=a, aa=aa, bb=bb, N=9, M=7)
    assert np.allclose(a, ref[0]) and np.allclose(aa, ref[1])


if __name__ == "__main__":
    test_invariant_slot_read_in_iteration_and_live_after_loop_maps()
    test_zero_trip_loop_keeps_slot_unchanged()
    test_conditional_store_not_covering_every_iteration_stays_sequential()
