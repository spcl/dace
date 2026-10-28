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


if __name__ == "__main__":
    test_simple_program()
    test_simple_sdfg()
    test_index_propagation_in_tiled_sdfg()
    test_simple_sdfg_map()
    test_simple_sdfg_program()
    test_badscope()
    test_tiled_program()
    test_nesting_a_map_body_whose_last_write_feeds_the_exit_adds_no_self_copy()
    test_nesting_a_map_keeps_the_ordering_edge_that_orders_an_earlier_read_before_it()
