# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop over the components of a segmented graph lifts when every iteration touches only its own component
(nfa_frontier), behind an entry guard that checks the data facts the confinement rests on."""

import re

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg import tasklet_utils as tutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.segment_confined_loops import SEGMENT_HINT, SegmentConfinedLoops

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


# Reproducer: Canon fixes and performance work (canon-wip WIP 1-32) (eabfcf9196)
def test_a_component_loop_lifts_behind_its_entry_guard_and_computes_the_reference():
    sdfg = canonical(frontier_walk)
    assert len(segment_maps(sdfg)) == 1
    guards = [n for n, parent in sdfg.all_nodes_recursive() if tutil.is_abort_guard(n) and n.label.startswith("check_")]
    assert {g.label for g in guards} >= {"check_monotone", "check_member"}
    finalize_for_target(sdfg, "cpu")
    # The proof is the entry guard, not the memlets: schedule inference must not take the region back (nfa_frontier
    # ran its 2190 components on one thread, 155 s against numba's 8 s).
    assert [entry.map.schedule for entry in segment_maps(sdfg)] == [dace.ScheduleType.CPU_Multicore]
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


# Reproducer: Canon fixes and performance work (canon-wip WIP 1-32) (eabfcf9196)
def test_a_write_one_past_a_member_keeps_the_loop():
    sdfg = canonical(frontier_walk_neighbour_write)
    assert not segment_maps(sdfg)


# Reproducer: Canon fixes and performance work (canon-wip WIP 1-32) (eabfcf9196)
def test_the_hint_names_the_pointer_array():
    assert "{pointer}" in SEGMENT_HINT and "{var}" in SEGMENT_HINT


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_the_hint_states_the_checks_its_proof_rests_on():
    """nfa_frontier: the hint said ``c touches only comp_ptr[c] .. comp_ptr[c + 1]``, though the loop reads other
    elements, and left out that the proof holds only behind the entry checks."""
    (entry,) = segment_maps(canonical(frontier_walk))
    hint = entry.specialization_hint.splitlines()[0]
    assert re.match(
        r"parallel -- segment-confined: iteration (\w+) accesses the arrays the loop writes only at \[\1\] or inside "
        r"\[comp_ptr\[\1\], comp_ptr\[\1 \+ 1\]\), and the checks before the loop abort unless "
        r"comp_ptr is nondecreasing",
        hint,
    ), hint


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


def store_symbol(state: dace.SDFGState, value: str, target: str) -> None:
    tasklet = state.add_tasklet(f"store_{value}", {}, {"o"}, f"o = {value}")
    state.add_edge(tasklet, "o", state.add_write(target.split("[")[0]), None, dace.Memlet(target))


def worklist_count_carried_across_components() -> dace.SDFG:
    """Per component ``c``: append the unseen starts to ``frontier[base + n]`` (``n`` reset per component), then copy
    the kept ones to ``out[base + n_out]``, with ``n_out`` reset only before the component loop."""
    sdfg = dace.SDFG("worklist_count_carried")
    for name in ("C", "NS", "NSTART", "base", "lo", "hi", "s", "y", "n", "n_out"):
        sdfg.add_symbol(name, dace.int64)
    for name, shape in (("comp_ptr", "C + 1"), ("start_ptr", "C + 1"), ("start_idx", "NSTART")):
        sdfg.add_array(name, [shape], dace.int64)
    for name, shape in (("enabled", "NS"), ("frontier", "NS"), ("out", "2 * NS")):
        sdfg.add_array(name, [shape], dace.int64)
    sdfg.add_scalar("flag", dace.int64, transient=True)
    init = sdfg.add_state("init", is_start_block=True)
    components = LoopRegion("components", "c < C", "c", "c = 0", "c = c + 1", sdfg=sdfg)
    sdfg.add_node(components)
    sdfg.add_edge(init, components, dace.InterstateEdge(assignments={"n_out": "0"}))

    head = components.add_state("head", is_start_block=True)
    starts = LoopRegion("starts", "k < hi", "k", "k = lo", "k = k + 1", sdfg=sdfg)
    components.add_node(starts)
    bounds = {"base": "comp_ptr[c]", "n": "0", "lo": "start_ptr[c]", "hi": "start_ptr[c + 1]"}
    components.add_edge(head, starts, dace.InterstateEdge(assignments=bounds))
    load = starts.add_state("load", is_start_block=True)
    test = starts.add_state_after(load, "test", assignments={"s": "start_idx[k]"})
    test.add_nedge(test.add_read("enabled"), test.add_write("flag"), dace.Memlet("enabled[s] -> [0]"))
    guard = ConditionalBlock("unseen")
    starts.add_node(guard)
    starts.add_edge(test, guard, dace.InterstateEdge())
    arm = ControlFlowRegion("append", sdfg=sdfg)
    guard.add_branch(CodeBlock("flag == 0"), arm)
    mark = arm.add_state("mark", is_start_block=True)
    store_symbol(mark, "1", "enabled[s]")
    store_symbol(mark, "s", "frontier[base + n]")
    arm.add_state_after(mark, "bumped", assignments={"n": "n + 1"})

    copies = LoopRegion("copies", "j < n", "j", "j = 0", "j = j + 1", sdfg=sdfg)
    components.add_node(copies)
    components.add_edge(starts, copies, dace.InterstateEdge())
    fetch = copies.add_state("fetch", is_start_block=True)
    copy_out = copies.add_state_after(fetch, "copy_out", assignments={"y": "frontier[base + j]"})
    store_symbol(copy_out, "y", "out[base + n_out]")
    # Write every member, advance past the kept ones: the bump sits in its own region, so it stays a counter
    keep = ConditionalBlock("keep")
    copies.add_node(keep)
    copies.add_edge(copy_out, keep, dace.InterstateEdge())
    advance = ControlFlowRegion("advance", sdfg=sdfg)
    keep.add_branch(CodeBlock("y >= base"), advance)
    counted = advance.add_state("counted", is_start_block=True)
    advance.add_state_after(counted, "next", assignments={"n_out": "n_out + 1"})
    return sdfg


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_worklist_count_never_reset_per_component_keeps_the_loop():
    """Component ``c`` writes ``out[comp_ptr[c] + S_c + j]``, ``S_c`` the starts of the earlier components: a lifted
    map would restart ``n_out`` in every component and write ``out[comp_ptr[c] + j]``."""
    sdfg = worklist_count_carried_across_components()
    assert SegmentConfinedLoops().apply_pass(sdfg, {}) is None
    comp_ptr, row_ptr, col_idx, start_ptr, start_idx = graph()
    ns = row_ptr.size - 1
    out = np.zeros(2 * ns, np.int64)
    enabled = np.zeros(ns, np.int64)
    frontier = np.zeros(ns, np.int64)
    sdfg(
        comp_ptr=comp_ptr,
        start_ptr=start_ptr,
        start_idx=start_idx,
        enabled=enabled,
        frontier=frontier,
        out=out,
        C=comp_ptr.size - 1,
        NS=ns,
        NSTART=start_idx.size,
    )
    want, seen, n_out = np.zeros(2 * ns, np.int64), np.zeros(ns, np.int64), 0
    for c in range(comp_ptr.size - 1):
        for k in range(start_ptr[c], start_ptr[c + 1]):
            if seen[start_idx[k]] == 0:
                seen[start_idx[k]] = 1
                want[comp_ptr[c] + n_out] = start_idx[k]
                n_out += 1
    assert np.array_equal(out, want)


if __name__ == "__main__":
    test_a_component_loop_lifts_behind_its_entry_guard_and_computes_the_reference()
    test_a_write_one_past_a_member_keeps_the_loop()
    test_the_hint_names_the_pointer_array()
    test_a_loop_whose_interstate_value_is_a_comparison_canonicalizes()
    test_a_worklist_count_never_reset_per_component_keeps_the_loop()
