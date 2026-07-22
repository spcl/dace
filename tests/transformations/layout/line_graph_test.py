# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A2 kernel_per_state + A3 line_graph + the A6 invariant guard (GLOBAL_LAYOUT_DESIGN.md): the
fixture programs split into one kernel per state and stay bit-exact, the line graph comes back in
dependency order, and every v1 refusal (multi-kernel state, branch, LoopRegion) is loud."""
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.layout.line_graph import (check_kernel_per_state, kernel_per_state, line_graph)
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures
from tests.transformations.layout.multinest_fixtures_test import run_and_check

EXPECTED_ORDER = {"conflict2": ["B", "C"], "conflict3": ["B", "C", "D"], "agree2": ["B", "C"]}


def prepared(program_name):
    program, _, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    return sdfg


def kernel_output(kernel):
    """The single non-transient array a fixture kernel writes."""
    sdfg = kernel.state.sdfg
    written = {
        n.data
        for n in kernel.state.data_nodes() if kernel.state.in_degree(n) > 0 and not sdfg.arrays[n.data].transient
    }
    assert len(written) == 1, written
    return written.pop()


@pytest.mark.parametrize("program_name", sorted(fixtures.PROGRAMS))
def test_kernel_per_state_splits_and_stays_bitexact(program_name):
    sdfg = prepared(program_name)
    kernel_per_state(sdfg)
    check_kernel_per_state(sdfg)  # must not raise
    kernels = line_graph(sdfg)
    assert [kernel_output(k) for k in kernels] == EXPECTED_ORDER[program_name]
    assert [k.index for k in kernels] == list(range(len(kernels)))
    run_and_check(sdfg, program_name, seed=2)


def test_line_graph_refuses_multi_kernel_state():
    """conflict2 keeps both nests in one fused state until A2 runs -- the A6 guard must fire."""
    sdfg = prepared("conflict2")
    with pytest.raises(RuntimeError, match="kernel-per-state invariant"):
        line_graph(sdfg)


def test_line_graph_refuses_branch():
    sdfg = dace.SDFG("branchy")
    first = sdfg.add_state("first", is_start_block=True)
    left = sdfg.add_state("left")
    right = sdfg.add_state("right")
    join = sdfg.add_state("join")
    sdfg.add_edge(first, left, dace.InterstateEdge(condition="1 > 0"))
    sdfg.add_edge(first, right, dace.InterstateEdge(condition="0 > 1"))
    sdfg.add_edge(left, join, dace.InterstateEdge())
    sdfg.add_edge(right, join, dace.InterstateEdge())
    with pytest.raises(NotImplementedError, match="branches"):
        line_graph(sdfg)


def add_map_nest(graph, label, src, dst, add):
    """A one-nest state ``dst[i] = src[i] + add`` in ``graph`` (an SDFG or a LoopRegion)."""
    state = graph.add_state(label)
    me, mx = state.add_map(f"{label}_m", {"i": "0:N"})
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, f"b = a + {add}")
    state.add_memlet_path(state.add_read(src), me, tasklet, dst_conn="a", memlet=dace.Memlet(f"{src}[i]"))
    state.add_memlet_path(tasklet, mx, state.add_write(dst), src_conn="b", memlet=dace.Memlet(f"{dst}[i]"))
    return state


def make_loop(name):
    return LoopRegion(name, condition_expr="k < 10", loop_var="k", initialize_expr="k = 0", update_expr="k = k + 1")


def loopy_sdfg():
    """pre (top) -> loop{ b1 -> b2 } -> post (top): a flat-body single loop."""
    from dace.transformation.layout.line_graph import loop_spans  # noqa: F401 (kept local to the fixture users)
    sdfg = dace.SDFG("loopy")
    for a in ("A", "B", "C", "D"):
        sdfg.add_array(a, [dace.symbol("N")], dace.float64)
    pre = add_map_nest(sdfg, "pre", "A", "B", 1.0)
    pre.is_start_block = True
    loop = make_loop("loop")
    sdfg.add_node(loop)
    b1 = add_map_nest(loop, "b1", "B", "C", 2.0)
    b1.is_start_block = True
    b2 = add_map_nest(loop, "b2", "C", "B", 3.0)
    loop.add_edge(b1, b2, dace.InterstateEdge())
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    post = add_map_nest(sdfg, "post", "B", "D", 4.0)
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    sdfg.validate()
    return sdfg, loop


def test_line_graph_admits_flat_body_loop():
    """A LoopRegion with a flat-line body is admitted: body kernels come back in order, tagged with the loop."""
    from dace.transformation.layout.line_graph import locked_transitions, loop_spans
    sdfg, loop = loopy_sdfg()
    kernels = line_graph(sdfg)
    assert [k.state.label for k in kernels] == ["pre", "b1", "b2", "post"]
    assert [k.index for k in kernels] == [0, 1, 2, 3]
    assert [k.loop for k in kernels] == [None, loop, loop, None]
    assert loop_spans(kernels) == [(1, 3)]  # the two body kernels form one span; pre/post are outside
    # Only the internal body transition (into kernel 2) is locked; entry-into-loop (1) and exit (3) stay free.
    assert locked_transitions(kernels) == {2}


def test_line_graph_refuses_nested_loop():
    """A loop inside a loop is refused (v1 handles a single loop level)."""
    sdfg = dace.SDFG("nested_loop")
    sdfg.add_array("A", [dace.symbol("N")], dace.float64)
    sdfg.add_array("B", [dace.symbol("N")], dace.float64)
    outer = make_loop("outer")
    sdfg.add_node(outer)
    sdfg.start_block  # force resolution not needed; outer is sole node
    inner = make_loop("inner")
    outer.add_node(inner)
    body = add_map_nest(inner, "body", "A", "B", 1.0)
    body.is_start_block = True
    with pytest.raises(NotImplementedError, match="nested LoopRegion"):
        line_graph(sdfg)


def test_line_graph_refuses_branch_in_loop_body():
    """A branch inside a loop body is refused by the same DAG guard as a top-level branch."""
    sdfg = dace.SDFG("loop_branch")
    sdfg.add_array("A", [dace.symbol("N")], dace.float64)
    loop = make_loop("loop")
    sdfg.add_node(loop)
    first = add_map_nest(loop, "first", "A", "A", 1.0)
    first.is_start_block = True
    left = loop.add_state("left")
    right = loop.add_state("right")
    loop.add_edge(first, left, dace.InterstateEdge(condition="1 > 0"))
    loop.add_edge(first, right, dace.InterstateEdge(condition="0 > 1"))
    with pytest.raises(NotImplementedError, match="branches"):
        line_graph(sdfg)


def test_line_graph_refuses_nonmap_work():
    sdfg = dace.SDFG("copyonly")
    sdfg.add_array("X", [8], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    t = sdfg.add_state("t")
    sdfg.add_edge(state, t, dace.InterstateEdge())
    tasklet = t.add_tasklet("lone", {}, {"o"}, "o = 1.0")
    t.add_edge(tasklet, "o", t.add_write("X"), None, dace.Memlet("X[0]"))
    with pytest.raises(NotImplementedError, match="non-map work"):
        line_graph(sdfg)


def test_line_graph_refuses_bare_copy_state():
    """A state of two access nodes joined by a copy memlet MOVES DATA: it must be refused as
    non-map work, not slip through as structural -- apply_assignment's liveness planning never
    sees pass-through states, so a silent pass-through would corrupt its read/write analysis."""
    sdfg = dace.SDFG("an_copy")
    sdfg.add_array("B", [8, 8], dace.float64)
    sdfg.add_array("C", [8, 8], dace.float64)
    state = sdfg.add_state("copy", is_start_block=True)
    state.add_nedge(state.add_read("B"), state.add_write("C"),
                    dace.Memlet(data="B", subset="0:8, 0:8", other_subset="0:8, 0:8"))
    sdfg.validate()
    with pytest.raises(NotImplementedError, match="non-map work"):
        line_graph(sdfg)


def test_kernel_per_state_refuses_undraggable_shared_sink():
    """Two nests writing the SAME sink access node: state_fission pulls the second nest along into
    the state it creates, so no split can succeed -- kernel_per_state must raise instead of
    reporting a split that did not happen (and leaving a misleading A6 failure for later)."""
    M = dace.symbol("M")
    sdfg = dace.SDFG("shared_sink")
    sdfg.add_array("A", [M], dace.float64)
    sdfg.add_array("B", [M], dace.float64)
    sdfg.add_array("C", [M, 2], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    c_node = state.add_access("C")
    for col, src in ((0, "A"), (1, "B")):
        me, mx = state.add_map(f"m{col}", {"j": "0:M"})
        tasklet = state.add_tasklet(f"t{col}", {"a"}, {"b"}, "b = a * 2.0")
        state.add_memlet_path(state.add_access(src), me, tasklet, dst_conn="a", memlet=dace.Memlet(f"{src}[j]"))
        state.add_memlet_path(tasklet, mx, c_node, src_conn="b", memlet=dace.Memlet(f"C[j, {col}]"))
    sdfg.validate()
    with pytest.raises(RuntimeError, match="no progress"):
        kernel_per_state(sdfg)


def two_loop_sdfg():
    """pre (top) -> l1{ c1 } -> l2{ c2 -> c3 }: two spans, one of them a single kernel."""
    sdfg = dace.SDFG("two_loops")
    for a in ("A", "B", "C", "D"):
        sdfg.add_array(a, [dace.symbol("N")], dace.float64)
    pre = add_map_nest(sdfg, "pre", "A", "B", 1.0)
    pre.is_start_block = True
    l1, l2 = make_loop("l1"), make_loop("l2")
    sdfg.add_node(l1)
    sdfg.add_node(l2)
    c1 = add_map_nest(l1, "c1", "B", "C", 2.0)
    c1.is_start_block = True
    c2 = add_map_nest(l2, "c2", "C", "D", 3.0)
    c2.is_start_block = True
    c3 = add_map_nest(l2, "c3", "D", "C", 4.0)
    l2.add_edge(c2, c3, dace.InterstateEdge())
    sdfg.add_edge(pre, l1, dace.InterstateEdge())
    sdfg.add_edge(l1, l2, dace.InterstateEdge())
    sdfg.validate()
    return sdfg, l1, l2


def test_two_loops_give_two_spans():
    """Every existing span assertion covers a SINGLE loop, so nothing pins that spans stay separate:
    a one-kernel span locks nothing, and the transition BETWEEN two loops must stay free (a layout may
    change there -- it is outside both bodies)."""
    from dace.transformation.layout.line_graph import locked_transitions, loop_spans
    sdfg, l1, l2 = two_loop_sdfg()
    kernels = line_graph(sdfg)
    assert [k.state.label for k in kernels] == ["pre", "c1", "c2", "c3"]
    assert [k.loop for k in kernels] == [None, l1, l2, l2]
    assert loop_spans(kernels) == [(1, 2), (2, 4)]
    assert locked_transitions(kernels) == {3}  # only inside l2; 1 and 2 are span entries, not internal


def test_empty_loop_body_contributes_nothing():
    """A LoopRegion with no blocks is walked and silently contributes no kernel -- pin it, so the
    'no kernels' outcome is a stated result rather than an accident of the empty-region early return."""
    from dace.transformation.layout.line_graph import loop_spans
    sdfg = dace.SDFG("empty_body")
    for a in ("A", "B"):
        sdfg.add_array(a, [dace.symbol("N")], dace.float64)
    pre = add_map_nest(sdfg, "pre", "A", "B", 1.0)
    pre.is_start_block = True
    loop = make_loop("empty")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    kernels = line_graph(sdfg)
    assert [k.state.label for k in kernels] == ["pre"]
    assert loop_spans(kernels) == []


def test_inert_state_between_body_kernels_keeps_one_span():
    """A do-nothing state inside a loop body takes no kernel position, so the kernels around it stay
    consecutive and the span still covers both."""
    from dace.transformation.layout.line_graph import locked_transitions, loop_spans
    sdfg, loop = loopy_sdfg()
    b1 = next(b for b in loop.nodes() if b.label == "b1")
    b2 = next(b for b in loop.nodes() if b.label == "b2")
    gap = loop.add_state("gap")
    loop.remove_edge(loop.edges_between(b1, b2)[0])
    loop.add_edge(b1, gap, dace.InterstateEdge())
    loop.add_edge(gap, b2, dace.InterstateEdge())
    kernels = line_graph(sdfg)
    assert [k.state.label for k in kernels] == ["pre", "b1", "b2", "post"]
    assert loop_spans(kernels) == [(1, 3)] and locked_transitions(kernels) == {2}


def test_data_moving_state_in_loop_body_is_refused():
    """The sibling of the test above: an inert state is fine, but one that MOVES DATA is not -- it is
    unmodelled work between two body kernels."""
    sdfg, loop = loopy_sdfg()
    b2 = next(b for b in loop.nodes() if b.label == "b2")
    mover = loop.add_state("mover")
    loop.add_edge(b2, mover, dace.InterstateEdge())
    mover.add_nedge(mover.add_read("B"), mover.add_write("C"),
                    dace.Memlet(data="B", subset="0:N", other_subset="0:N"))
    with pytest.raises(NotImplementedError, match="non-map work"):
        line_graph(sdfg)


def test_relayout_state_in_loop_body_is_refused():
    """A LayoutChange takes no kernel position, which is right at top level and WRONG in a loop body:
    the kernels on either side would fuse into one span (pinning a single layout) while the relayout it
    sits between re-runs every iteration, unpriced. Refuse instead of silently mismodelling."""
    from dace.libraries.layout import add_layout_change
    from dace.libraries.layout.algebra import Permute
    sdfg = dace.SDFG("relayout_in_loop")
    sdfg.add_array("A", [8, 8], dace.float64)
    sdfg.add_array("B", [8, 8], dace.float64)
    loop = make_loop("loop")
    sdfg.add_node(loop)
    body = loop.add_state("body")
    body.is_start_block = True
    me, mx = body.add_map("m", {"i": "0:8", "j": "0:8"})
    tasklet = body.add_tasklet("t", {"a"}, {"b"}, "b = a + 1.0")
    body.add_memlet_path(body.add_read("A"), me, tasklet, dst_conn="a", memlet=dace.Memlet("A[i,j]"))
    body.add_memlet_path(tasklet, mx, body.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i,j]"))
    relayout = loop.add_state("relayout_B")
    loop.add_edge(body, relayout, dace.InterstateEdge())
    add_layout_change(sdfg, relayout, "B", "B_cm", [Permute((1, 0))])
    with pytest.raises(NotImplementedError, match="relayout state"):
        line_graph(sdfg)


def test_conditional_block_is_refused():
    """The deferred-conditionals refusal has its own message and its own branch; both existing branch
    tests hit the DIFFERENT successor-count guard, so this path was unexercised."""
    from dace.sdfg.state import ConditionalBlock
    sdfg = dace.SDFG("conditional")
    sdfg.add_array("A", [dace.symbol("N")], dace.float64)
    sdfg.add_array("B", [dace.symbol("N")], dace.float64)
    pre = add_map_nest(sdfg, "pre", "A", "B", 1.0)
    pre.is_start_block = True
    cond = ConditionalBlock("cond")
    sdfg.add_node(cond)
    sdfg.add_edge(pre, cond, dace.InterstateEdge())
    with pytest.raises(NotImplementedError, match="conditionals are deferred"):
        line_graph(sdfg)


def test_conditional_block_in_loop_body_is_refused():
    """Same refusal one level down: the recursion into a loop body must not lose it."""
    from dace.sdfg.state import ConditionalBlock
    sdfg, loop = loopy_sdfg()
    b2 = next(b for b in loop.nodes() if b.label == "b2")
    cond = ConditionalBlock("cond")
    loop.add_node(cond)
    loop.add_edge(b2, cond, dace.InterstateEdge())
    with pytest.raises(NotImplementedError, match="conditionals are deferred"):
        line_graph(sdfg)


if __name__ == "__main__":
    for name in sorted(fixtures.PROGRAMS):
        test_kernel_per_state_splits_and_stays_bitexact(name)
    test_line_graph_refuses_multi_kernel_state()
    test_line_graph_refuses_branch()
    test_line_graph_admits_flat_body_loop()
    test_line_graph_refuses_nested_loop()
    test_line_graph_refuses_branch_in_loop_body()
    test_line_graph_refuses_nonmap_work()
    test_line_graph_refuses_bare_copy_state()
    test_kernel_per_state_refuses_undraggable_shared_sink()
    test_two_loops_give_two_spans()
    test_empty_loop_body_contributes_nothing()
    test_inert_state_between_body_kernels_keeps_one_span()
    test_data_moving_state_in_loop_body_is_refused()
    test_relayout_state_in_loop_body_is_refused()
    test_conditional_block_is_refused()
    test_conditional_block_in_loop_body_is_refused()
    print("line_graph tests PASS")
