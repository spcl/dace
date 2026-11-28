# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.helpers import all_isedges_between, reconnect_edge_through_map


def _make_consecutive_loops_sdfg():
    sdfg = dace.SDFG("consecutive_loops")

    before_first = sdfg.add_state("before_first", is_start_block=True)
    first_body = sdfg.add_state("first_body")
    between_loops = sdfg.add_state("between_loops")
    sdfg.add_loop(before_first, first_body, between_loops, "i", "0", "i < 4", "i + 1", label="first_loop")

    second_body = sdfg.add_state("second_body")
    after_second = sdfg.add_state("after_second")
    sdfg.add_loop(between_loops, second_body, after_second, "j", "0", "j < 4", "j + 1", label="second_loop")

    return sdfg


def _make_nested_loops_sdfg():
    sdfg = dace.SDFG("nested_loops")

    before_outer = sdfg.add_state("before_outer", is_start_block=True)
    outer_before_inner = sdfg.add_state("outer_before_inner")
    inner_body = sdfg.add_state("inner_body")
    after_inner = sdfg.add_state("after_inner")
    after_outer = sdfg.add_state("after_outer")

    sdfg.add_loop(outer_before_inner, inner_body, after_inner, "j", "0", "j < 4", "j + 1", label="inner_loop")
    sdfg.add_loop(
        before_outer,
        outer_before_inner,
        after_outer,
        "i",
        "0",
        "i < 4",
        "i + 1",
        label="outer_loop",
        loop_end_block=after_inner,
    )

    return sdfg


def _edge_signature(edge):
    return (
        edge.src.label,
        type(edge.src).__name__,
        getattr(edge.src.parent_graph, "label", None),
        edge.dst.label,
        type(edge.dst).__name__,
        getattr(edge.dst.parent_graph, "label", None),
        edge.data.condition.as_string if edge.data.condition else None,
        tuple(sorted((str(k), str(v)) for k, v in edge.data.assignments.items())),
    )


def test_all_isedges_between_consecutive_loops():
    sdfg = _make_consecutive_loops_sdfg()
    blocks = {block.label: block for block in sdfg.all_control_flow_blocks()}

    edges_to_after_first = {
        _edge_signature(edge) for edge in all_isedges_between(blocks["first_body"], blocks["between_loops"])
    }
    assert edges_to_after_first == {
        ("first_loop", "LoopRegion", "consecutive_loops", "between_loops", "SDFGState", "consecutive_loops", "1", ()),
    }

    edges_to_second_body = {
        _edge_signature(edge) for edge in all_isedges_between(blocks["first_body"], blocks["second_body"])
    }
    assert edges_to_second_body == {
        ("first_loop", "LoopRegion", "consecutive_loops", "between_loops", "SDFGState", "consecutive_loops", "1", ()),
        ("between_loops", "SDFGState", "consecutive_loops", "second_loop", "LoopRegion", "consecutive_loops", "1", ()),
    }


def test_all_isedges_between_nested_loops():
    sdfg = _make_nested_loops_sdfg()
    blocks = {block.label: block for block in sdfg.all_control_flow_blocks()}

    edges_to_after_inner = {
        _edge_signature(edge) for edge in all_isedges_between(blocks["inner_body"], blocks["after_inner"])
    }
    assert edges_to_after_inner == {
        ("inner_loop", "LoopRegion", "outer_loop", "after_inner", "SDFGState", "outer_loop", "1", ()),
    }

    edges_to_after_outer = {
        _edge_signature(edge) for edge in all_isedges_between(blocks["inner_body"], blocks["after_outer"])
    }
    assert edges_to_after_outer == {
        ("inner_loop", "LoopRegion", "outer_loop", "after_inner", "SDFGState", "outer_loop", "1", ()),
        ("outer_loop", "LoopRegion", "nested_loops", "after_outer", "SDFGState", "nested_loops", "1", ()),
    }


def test_an_empty_edge_reconnects_through_a_map_without_connectors():
    """``s = 0`` inside a map is bound to the scope by an empty edge. Routed through a new inner map it must
    stay connector-free on both sides: one ``IN_``/``OUT_`` pair around a ``None`` left the memlet path
    walking back to a ``None`` connector."""
    sdfg = dace.SDFG("empty_edge_reconnect")
    sdfg.add_scalar("s", dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map("outer", dict(i="0:4"))
    seed = state.add_tasklet("seed", {}, {"__out"}, "__out = 0")
    state.add_nedge(me, seed, dace.Memlet())
    state.add_memlet_path(seed, mx, state.add_write("s"), src_conn="__out", memlet=dace.Memlet("s[0]"))
    inner_me, _ = state.add_map("inner", dict(j="0:1"))
    outer_edge, inner_edge = reconnect_edge_through_map(state, state.out_edges(me)[0], inner_me, True)
    for edge in (outer_edge, inner_edge):
        assert edge.data.is_empty() and edge.src_conn is None and edge.dst_conn is None
        assert state.memlet_path(edge) == [edge]


if __name__ == "__main__":
    test_all_isedges_between_consecutive_loops()
    test_all_isedges_between_nested_loops()
    test_an_empty_edge_reconnects_through_a_map_without_connectors()
