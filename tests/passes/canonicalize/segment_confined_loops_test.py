# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop over the components of a segmented graph lifts when every iteration touches only its own component
(nfa_frontier), behind an entry guard that checks the data facts the confinement rests on."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg import tasklet_utils as tutil
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.segment_confined_loops import SEGMENT_HINT

C = dace.symbol("C")
NS = dace.symbol("NS")
NE = dace.symbol("NE")
NSTART = dace.symbol("NSTART")
T = dace.symbol("T")


@dace.program
def frontier_walk(
    comp_ptr: dace.int64[C + 1],
    row_ptr: dace.int64[NS + 1],
    col_idx: dace.int64[NE],
    start_ptr: dace.int64[C + 1],
    start_idx: dace.int64[NSTART],
    visits: dace.int64[NS],
    sizes: dace.int64[C],
):
    """Per component: seed the frontier with its starts, then T times replace it by its children, deduplicated."""
    enabled = np.zeros(NS, dtype=np.int64)
    frontier = np.zeros(NS, dtype=np.int64)
    matched = np.zeros(NS, dtype=np.int64)
    for c in range(C):
        base = comp_ptr[c]
        n_front = 0
        for k in range(start_ptr[c], start_ptr[c + 1]):
            s = start_idx[k]
            if enabled[s] == 0:
                enabled[s] = 1
                frontier[base + n_front] = s
                n_front += 1
        for t in range(T):
            n_match = 0
            for k in range(n_front):
                s = frontier[base + k]
                matched[base + n_match] = s
                n_match += 1
                visits[s] += 1
                enabled[s] = 0
            n_front = 0
            for k in range(n_match):
                s = matched[base + k]
                for e in range(row_ptr[s], row_ptr[s + 1]):
                    child = col_idx[e]
                    if enabled[child] == 0:
                        enabled[child] = 1
                        frontier[base + n_front] = child
                        n_front += 1
        sizes[c] = n_front


@dace.program
def frontier_walk_neighbour_write(
    comp_ptr: dace.int64[C + 1],
    start_ptr: dace.int64[C + 1],
    start_idx: dace.int64[NSTART],
    visits: dace.int64[NS + 1],
):
    """Writes ``visits[s + 1]``: one past a member, which may be the next component's first state."""
    for c in range(C):
        for k in range(start_ptr[c], start_ptr[c + 1]):
            s = start_idx[k]
            visits[s + 1] += 1


def reference(comp_ptr, row_ptr, col_idx, start_ptr, start_idx, steps):
    ns = row_ptr.size - 1
    enabled, visits, sizes = np.zeros(ns, np.int64), np.zeros(ns, np.int64), np.zeros(comp_ptr.size - 1, np.int64)
    for c in range(comp_ptr.size - 1):
        frontier = []
        for k in range(start_ptr[c], start_ptr[c + 1]):
            s = start_idx[k]
            if enabled[s] == 0:
                enabled[s] = 1
                frontier.append(s)
        for step in range(steps):  # noqa: B007 -- the steps repeat the same rewrite
            matched = list(frontier)
            for s in matched:
                visits[s] += 1
                enabled[s] = 0
            frontier = []
            for s in matched:
                for e in range(row_ptr[s], row_ptr[s + 1]):
                    if enabled[col_idx[e]] == 0:
                        enabled[col_idx[e]] = 1
                        frontier.append(col_idx[e])
        sizes[c] = len(frontier)
    return visits, sizes


def graph():
    """Three components of 4, 3 and 5 states, rings with a chord, one or two starts each."""
    sizes = [4, 3, 5]
    comp_ptr = np.cumsum([0] + sizes).astype(np.int64)
    rows, starts, start_ptr = [], [], [0]
    for c, size in enumerate(sizes):
        first = comp_ptr[c]
        rows.extend([first + (i + 1) % size, first + (i + 2) % size] for i in range(size))
        starts += [first, first + 1] if size > 3 else [first]
        start_ptr.append(len(starts))
    row_ptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
    col_idx = np.array([v for r in rows for v in r], dtype=np.int64)
    return comp_ptr, row_ptr, col_idx, np.array(start_ptr, np.int64), np.array(starts, np.int64)


def canonical(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg, target="cpu")
    return sdfg


def segment_maps(sdfg: dace.SDFG) -> list:
    return [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and "segment-confined" in (getattr(node, "specialization_hint", "") or "")
    ]


def test_a_component_loop_lifts_behind_its_entry_guard_and_computes_the_reference():
    sdfg = canonical(frontier_walk)
    assert len(segment_maps(sdfg)) == 1
    guards = [n for n, parent in sdfg.all_nodes_recursive() if tutil.is_abort_guard(n) and n.label.startswith("check_")]
    assert {g.label for g in guards} >= {"check_monotone", "check_member"}
    finalize_for_target(sdfg, "cpu")
    comp_ptr, row_ptr, col_idx, start_ptr, start_idx = graph()
    visits, sizes = np.zeros(row_ptr.size - 1, np.int64), np.zeros(comp_ptr.size - 1, np.int64)
    sdfg(
        comp_ptr=comp_ptr,
        row_ptr=row_ptr,
        col_idx=col_idx,
        start_ptr=start_ptr,
        start_idx=start_idx,
        visits=visits,
        sizes=sizes,
        C=comp_ptr.size - 1,
        NS=row_ptr.size - 1,
        NE=col_idx.size,
        NSTART=start_idx.size,
        T=5,
    )
    want_visits, want_sizes = reference(comp_ptr, row_ptr, col_idx, start_ptr, start_idx, 5)
    assert np.array_equal(visits, want_visits)
    assert np.array_equal(sizes, want_sizes)


def test_a_write_one_past_a_member_keeps_the_loop():
    sdfg = canonical(frontier_walk_neighbour_write)
    assert not segment_maps(sdfg)


def test_the_hint_names_the_pointer_array():
    assert "{pointer}" in SEGMENT_HINT and "{var}" in SEGMENT_HINT


N = dace.symbol("N")
knob = dace.symbol("knob", dace.float64)


@dace.program
def halves_until_knob(a: dace.float64[N], b: dace.float64[N]):
    b[:] = a
    for _ in range(64):
        if np.max(b) <= knob:
            break
        b[:] = b * 0.5


def test_a_loop_whose_interstate_value_is_a_comparison_canonicalizes():
    """An interstate assignment can hold a comparison (``max_b <= knob``), which is no counter
    ``var + k``: reading it as one raised ``TypeError`` (HPCAgent-Bench's pinned-knob fixture)."""
    sdfg = halves_until_knob.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    a, b = np.arange(1.0, 9.0), np.zeros(8)
    sdfg(a=a, b=b, N=8, knob=0.25)
    ref = a.copy()
    while ref.max() > 0.25:
        ref *= 0.5
    assert np.allclose(b, ref)


if __name__ == "__main__":
    test_a_component_loop_lifts_behind_its_entry_guard_and_computes_the_reference()
    test_a_write_one_past_a_member_keeps_the_loop()
    test_the_hint_names_the_pointer_array()
    test_a_loop_whose_interstate_value_is_a_comparison_canonicalizes()
