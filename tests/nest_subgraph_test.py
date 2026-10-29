# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

import dace
from dace.sdfg.graph import NodeNotFoundError, SubgraphView
from dace.sdfg.nodes import MapEntry, Tasklet
from dace.transformation.dataflow import tiling
from dace.transformation.helpers import nest_state_subgraph

N = dace.symbol("N")


def create_sdfg():
    sdfg = dace.SDFG("badscope_test")
    sdfg.add_array("A", [2], dace.float32)
    sdfg.add_array("B", [2], dace.float32)
    state = sdfg.add_state()
    t, me, mx = state.add_mapped_tasklet(
        "map",
        dict(i="0:2"),
        dict(a=dace.Memlet.simple("A", "i")),
        "b = a * 2",
        dict(b=dace.Memlet.simple("B", "i")),
        external_edges=True,
    )
    return sdfg, state, t, me, mx


def create_sdfg_4():
    sdfg = dace.SDFG("sdfg_4_test")
    sdfg.add_array("A", [4], dace.float32)
    sdfg.add_array("B", [4], dace.float32)
    state = sdfg.add_state()
    t, me, mx = state.add_mapped_tasklet(
        "map",
        dict(i="0:4"),
        dict(a=dace.Memlet.simple("A", "i")),
        "b = a * 2",
        dict(b=dace.Memlet.simple("B", "i")),
        external_edges=True,
    )
    return sdfg, state, t, me, mx


def create_tiled_sdfg():
    sdfg = dace.SDFG("badscope_tile_test")
    sdfg.add_array("A", [4], dace.float32)
    sdfg.add_array("B", [4], dace.float32)
    state = sdfg.add_state()
    ome, omx = state.add_map("outer_map", dict(i="0:2"))
    ime, imx = state.add_map("inner_map", dict(j="0:2"))
    t = state.add_tasklet("tasklet", {"a"}, {"b"}, "b = a * 2")
    A = state.add_read("A")
    B = state.add_write("B")
    state.add_memlet_path(A, ome, ime, t, dst_conn="a", memlet=dace.Memlet.simple("A", "i*2 + j"))
    state.add_memlet_path(t, imx, omx, B, src_conn="b", memlet=dace.Memlet.simple("B", "i*2 + j"))
    return sdfg, state


def test_simple_program():

    @dace.program
    def multiply(a: dace.float32[N]):
        a *= 2
        a *= 3

    sdfg = multiply.to_sdfg(simplify=True)
    for state in sdfg.nodes():
        if any(isinstance(node, Tasklet) for node in state.nodes()):
            break
    else:
        raise KeyError("State with tasklet not found")

    tasklet_nodes = [n for n in state.nodes() if isinstance(n, Tasklet)]
    with pytest.raises(ValueError):
        nest_state_subgraph(sdfg, state, SubgraphView(state, tasklet_nodes))

    nest_state_subgraph(sdfg, state, SubgraphView(state, [tasklet_nodes[0]]))
    sdfg.validate()
    nest_state_subgraph(sdfg, state, SubgraphView(state, [tasklet_nodes[1]]))
    sdfg.validate()


def test_simple_sdfg():
    sdfg, state, t, me, mx = create_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, [t]))
    sdfg.validate()


def test_index_propagation_in_tiled_sdfg():
    sdfg, state, t, me, mx = create_sdfg_4()
    tiling.MapTiling.apply_to(sdfg=sdfg, options={"tile_sizes": (2,)}, map_entry=me)
    nested_me = state.in_edges(t)[0].src
    nested_mx = state.out_edges(t)[0].dst
    nest_state_subgraph(sdfg, state, SubgraphView(state, [nested_me, t, nested_mx]))
    sdfg.validate()
    sdfg.compile()


def test_simple_sdfg_map():
    sdfg, state, t, me, mx = create_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, [me, t, mx]))
    sdfg.validate()


def test_simple_sdfg_program():
    sdfg, state, t, me, mx = create_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))
    sdfg.validate()


def test_badscope():
    with pytest.raises(ValueError):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t, me]))

    with pytest.raises(ValueError):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t, mx]))

    with pytest.raises(NodeNotFoundError):
        sdfg, state, t, me, mx = create_sdfg()
        b_node = state.sink_nodes()[0]
        sdfg, state, t, me, mx = create_sdfg()
        # Notice that b_node comes from another graph
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t, b_node]))


def test_tiled_program():
    # Tasklet only
    sdfg, state = create_tiled_sdfg()
    tasklet = next(n for n in state.nodes() if isinstance(n, Tasklet))
    nest_state_subgraph(sdfg, state, SubgraphView(state, [tasklet]))
    sdfg.validate()

    # Inner map scope
    sdfg, state = create_tiled_sdfg()
    tasklet = next(n for n in state.nodes() if isinstance(n, Tasklet))
    entry = state.entry_node(tasklet)
    nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))
    sdfg.validate()

    # Outer map scope
    sdfg, state = create_tiled_sdfg()
    sdc = state.scope_children()
    entry = next(n for n in sdc[None] if isinstance(n, MapEntry))
    nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))
    sdfg.validate()

    # Entire state
    sdfg, state = create_tiled_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))
    sdfg.validate()


def test_nest_scalar_with_none_subset_boundary():
    """A scalar that crosses the nested-subgraph boundary can carry a bare,
    whole-scalar memlet (``Memlet(data='s')``, ``subset is None``) on an internal
    edge while its boundary edge carries an explicit ``s[0]`` subset. Re-basing a
    ``None`` subset is a no-op, so ``nest_state_subgraph`` must skip it rather than
    dereference ``None`` -- pre-fix this raised ``AttributeError: 'NoneType' object
    has no attribute 'offset'`` (the azimint_hist histogram-accumulator shape that
    ``MapToForLoop`` hit during canonicalization)."""
    sdfg = dace.SDFG("none_subset_boundary")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    A = state.add_read("A")
    me, mx = state.add_map("m", dict(i="0:4"))
    t1 = state.add_tasklet("t1", {"a"}, {"o"}, "o = a * 2")
    state.add_memlet_path(A, me, t1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    # Internal edge into the map exit is a BARE whole-scalar memlet (subset None).
    mx.add_in_connector("IN_s")
    mx.add_out_connector("OUT_s")
    state.add_edge(t1, "o", mx, "IN_s", dace.Memlet(data="s"))
    s_out = state.add_access("s")
    # Boundary edge out of the map carries an explicit point subset.
    state.add_edge(mx, "OUT_s", s_out, None, dace.Memlet("s[0]"))
    t3 = state.add_tasklet("t3", {"x"}, {"b"}, "b = x")
    B = state.add_write("B")
    state.add_edge(s_out, None, t3, "x", dace.Memlet("s[0]"))
    state.add_edge(t3, "b", B, None, dace.Memlet("B[0]"))

    nest_state_subgraph(sdfg, state, state.scope_subgraph(me))
    sdfg.validate()


def test_nesting_a_top_level_subgraph_consolidates_nothing(monkeypatch):
    """A subgraph at the top of its state has no enclosing scope to consolidate, and the call scanned
    every state of the SDFG to find that out -- once per map ``MapToForLoop`` lowers."""
    from dace.sdfg import utils

    sdfg, state, t, me, mx = create_sdfg()
    calls = []
    monkeypatch.setattr(utils, "consolidate_edges", lambda *args, **kwargs: calls.append(args))
    nest_state_subgraph(sdfg, state, SubgraphView(state, [me, t, mx]))
    assert calls == []
    sdfg.validate()


def test_nesting_a_map_body_whose_last_write_feeds_the_exit_adds_no_self_copy():
    """A map body whose own access node of ``A`` feeds the map exit: inside the nested SDFG that node
    already writes the ``A`` connector, so no ``A -> A`` copy into a second node may appear."""
    sdfg = dace.SDFG("nest_body_last_write_feeds_exit")
    sdfg.add_array("A", [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("m", dict(i="0:N"))
    t = state.add_tasklet("t", {}, {"o"}, "o = i")
    a_in = state.add_access("A")
    a_out = state.add_write("A")
    state.add_nedge(me, t, dace.Memlet())
    state.add_edge(t, "o", a_in, None, dace.Memlet("A[i]"))
    mx.add_in_connector("IN_A")
    mx.add_out_connector("OUT_A")
    state.add_edge(a_in, None, mx, "IN_A", dace.Memlet("A[i]"))
    state.add_edge(mx, "OUT_A", a_out, None, dace.Memlet("A[0:N]"))
    sdfg.validate()

    nested = nest_state_subgraph(sdfg, state, SubgraphView(state, [t, a_in]))
    (inner,) = nested.sdfg.states()
    self_copies = [
        e
        for e in inner.edges()
        if isinstance(e.src, dace.nodes.AccessNode)
        and isinstance(e.dst, dace.nodes.AccessNode)
        and e.src.data == e.dst.data
    ]
    assert self_copies == []
    sdfg.validate()

    A = np.zeros(8)
    sdfg(A=A, N=8)
    assert np.array_equal(A, np.arange(8, dtype=np.float64))


def test_nesting_a_map_keeps_the_ordering_edge_that_orders_an_earlier_read_before_it():
    """``C = A[0]`` must run before a map overwriting ``A``, ordered only by an empty memlet into the map entry."""
    sdfg = dace.SDFG("nest_keeps_ordering_edge")
    for name, size in (("A", 4), ("B", 4), ("C", 1)):
        sdfg.add_array(name, [size], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("overwrite", dict(i="0:4"))
    t = state.add_tasklet("copy", {"b"}, {"a"}, "a = b")
    state.add_memlet_path(state.add_read("B"), me, t, dst_conn="b", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(t, mx, state.add_write("A"), src_conn="a", memlet=dace.Memlet("A[i]"))
    read = state.add_tasklet("read", {"a"}, {"c"}, "c = a")
    state.add_edge(state.add_read("A"), None, read, "a", dace.Memlet("A[0]"))
    state.add_edge(read, "c", state.add_write("C"), None, dace.Memlet("C[0]"))
    state.add_nedge(read, me, dace.Memlet())
    sdfg.validate()

    nested = nest_state_subgraph(sdfg, state, state.scope_subgraph(me))
    assert any(e.dst is nested for e in state.out_edges(read))
    sdfg.validate()

    A, B, C = np.full(4, 5.0), np.arange(4.0) + 10, np.zeros(1)
    sdfg(A=A, B=B, C=C)
    assert C[0] == 5, C


def test_a_view_on_the_nesting_boundary_is_an_array_inside_the_nested_sdfg():
    """``V = A[1, :]`` feeds a map that is nested; the parent keeps the ``views`` edge ``A -> V``."""
    sdfg = dace.SDFG("nest_boundary_view")
    sdfg.add_array("A", [4, 4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    sdfg.add_view("V", [4], dace.float64)
    state = sdfg.add_state()
    view = state.add_access("V")
    state.add_edge(state.add_read("A"), None, view, "views", dace.Memlet("A[1, 0:4]"))
    me, mx = state.add_map("m", dict(i="0:4"))
    t = state.add_tasklet("double", {"x"}, {"y"}, "y = x * 2")
    state.add_memlet_path(view, me, t, dst_conn="x", memlet=dace.Memlet("V[i]"))
    state.add_memlet_path(t, mx, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    sdfg.validate()

    nest_state_subgraph(sdfg, state, state.scope_subgraph(me))
    sdfg.validate()

    A = np.arange(16.0).reshape(4, 4).copy()
    B = np.zeros(4)
    sdfg(A=A, B=B)
    assert np.array_equal(B, A[1] * 2), B


def test_a_boundary_copy_named_by_a_moved_transient_is_restated_on_the_outer_container():
    """A map body copies ``A[i, :]`` into its local ``tmp``, the copy named by ``tmp`` with ``A``'s subset as its
    other subset; the body (and ``tmp``) is nested."""
    sdfg = dace.SDFG("nest_copy_named_by_moved_transient")
    sdfg.add_array("A", [4, 8], dace.float64)
    sdfg.add_array("B", [4, 8], dace.float64)
    sdfg.add_transient("tmp", [8], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:4"})
    tmp = state.add_access("tmp")
    entry.add_in_connector("IN_A")
    entry.add_out_connector("OUT_A")
    state.add_edge(state.add_read("A"), None, entry, "IN_A", dace.Memlet("A[0:4, 0:8]"))
    state.add_edge(entry, "OUT_A", tmp, None, dace.Memlet(data="tmp", subset="0:8", other_subset="i, 0:8"))
    inner_entry, inner_exit = state.add_map("cols", {"j": "0:8"})
    t = state.add_tasklet("double", {"t"}, {"o"}, "o = 2.0 * t")
    state.add_memlet_path(tmp, inner_entry, t, dst_conn="t", memlet=dace.Memlet("tmp[j]"))
    state.add_memlet_path(t, inner_exit, exit_node, state.add_write("B"), src_conn="o", memlet=dace.Memlet("B[i, j]"))
    sdfg.validate()

    nest_state_subgraph(sdfg, state, state.scope_subgraph(entry, include_entry=False, include_exit=False))
    sdfg.validate()

    A = np.arange(32.0).reshape(4, 8).copy()
    B = np.zeros((4, 8))
    sdfg(A=A, B=B)
    assert np.array_equal(B, 2 * A), B


if __name__ == "__main__":
    test_simple_program()
    test_simple_sdfg()
    test_index_propagation_in_tiled_sdfg()
    test_simple_sdfg_map()
    test_simple_sdfg_program()
    test_badscope()
    test_tiled_program()
    test_nest_scalar_with_none_subset_boundary()
    test_nesting_a_map_body_whose_last_write_feeds_the_exit_adds_no_self_copy()


@dace.program
def interior_double(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1 : N - 1]:
        B[i] = A[i] * 2.0


def test_a_nest_of_a_partial_write_claims_it_as_dynamic():
    """The nest's connector passes all of ``B`` but writes only ``B[1:N-1]``: claimed exact, the offloader skipped
    the copy-in and returned uninitialized border elements."""
    sdfg = interior_double.to_sdfg(simplify=True)
    state = sdfg.start_state

    nested = nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))

    (written,) = [e.data for e in state.out_edges(nested)]
    assert (written.data, written.dynamic) == ("B", True)
