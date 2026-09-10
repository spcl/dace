# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test_remove_edge_global_scope():
    sdfg = dace.SDFG("simple_edge_remover_test")
    state = sdfg.add_state()

    sdfg.add_array("a", shape=(1, ), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(1, ), dtype=dace.float64, transient=False)

    tlet = state.add_tasklet(
        "comp",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in + 1.0",
    )
    a = state.add_access("a")
    b = state.add_access("b")

    up_edge = state.add_edge(a, None, tlet, "__in", dace.Memlet("a[0]"))
    down_edge = state.add_edge(tlet, "__out", b, None, dace.Memlet("b[0]"))
    sdfg.validate()

    # Now remove the edge using the function. Note that this makes the SDFG invalid,
    #  because the tasklet wants an input. We ignore that for now.
    nb_removed_edges = dace.sdfg.utils.remove_edge_and_dangling_path(state, up_edge)
    assert nb_removed_edges == 1
    assert a not in state.nodes()
    assert tlet in state.nodes()
    assert b in state.nodes()
    assert sdfg.arrays.keys() == {"a", "b"}
    assert len(tlet.in_connectors) == 0
    assert state.in_degree(tlet) == 0
    assert tlet.out_connectors.keys() == {"__out"}
    assert state.out_degree(tlet) == 1

    # If we now also delete the `down_edge` then the state will become empty.
    nb_removed_edges = dace.sdfg.utils.remove_edge_and_dangling_path(state, down_edge)
    assert nb_removed_edges == 1
    assert state.number_of_nodes() == 0


def test_remove_edge_nested_scope():
    sdfg = dace.SDFG("nested_edge_remover_test")
    state = sdfg.add_state()

    sdfg.add_array("a", shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(10, 10, 2), dtype=dace.float64, transient=False)

    tlet = state.add_tasklet(
        "comp",
        inputs={"__in1", "__in2"},
        outputs={"__out1", "__out2"},
        code="__out1 = __in1 + 1.0\n__out2 = __in2 - 1.0",
    )
    a, b = (state.add_access(name) for name in "ab")
    me, mx = state.add_map("outer_map", ndrange={"__i": "0:10"})
    nme, nmx = state.add_map("nested_map", ndrange={"__j": "0:10"})

    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[0:10, 0:10]"))
    me.add_scope_connectors("a")
    state.add_edge(me, "OUT_a", nme, "IN_a1", dace.Memlet("a[__i, 0:10]"))
    state.add_edge(me, "OUT_a", nme, "IN_a2", dace.Memlet("a[0:10, __i]"))
    nme.add_scope_connectors("a1")
    nme.add_scope_connectors("a2")

    up_edge1 = state.add_edge(nme, "OUT_a1", tlet, "__in1", dace.Memlet("a[__i, __j]"))
    up_edge2 = state.add_edge(nme, "OUT_a2", tlet, "__in2", dace.Memlet("a[__j, __i]"))

    down_edge1 = state.add_edge(tlet, "__out1", nmx, "IN_b", dace.Memlet("b[__i, __j, 0]"))
    down_edge2 = state.add_edge(tlet, "__out2", nmx, "IN_b", dace.Memlet("b[__j, __i, 1]"))
    nmx.add_scope_connectors("b")

    state.add_edge(nmx, "OUT_b", mx, "IN_b", dace.Memlet("b[0:10, 0:10, 0:2]"))
    mx.add_scope_connectors("b")
    state.add_edge(mx, "OUT_b", b, None, dace.Memlet("b[0:10, 0:10, 0:2]"))
    sdfg.validate()

    assert state.number_of_nodes() == 7

    # Because of the fan out, the deletion will stop.
    nb_rm_up_edge1 = dace.sdfg.utils.remove_edge_and_dangling_path(state, up_edge1)
    assert nb_rm_up_edge1 == 2
    assert state.number_of_nodes() == 7
    assert set(tlet.in_connectors.keys()) == {"__in2"}
    assert state.in_degree(tlet) == 1
    assert set(tlet.out_connectors.keys()) == {"__out1", "__out2"}
    assert state.out_degree(tlet) == 2
    assert set(nme.out_connectors.keys()) == {"OUT_a2"}
    assert set(nme.in_connectors.keys()) == {"IN_a2"}
    assert state.out_degree(nme) == 1
    assert state.in_degree(nme) == 1
    assert state.out_degree(me) == 1
    assert state.in_degree(me) == 1
    assert set(me.out_connectors.keys()) == {"OUT_a"}
    assert set(me.in_connectors.keys()) == {"IN_a"}
    assert state.degree(a) == 1

    # Now the deletion will go up and remove the map entries; which leads to a
    #  technical and functional invalid SDFG.
    nb_rm_up_edge2 = dace.sdfg.utils.remove_edge_and_dangling_path(state, up_edge2)
    assert nb_rm_up_edge2 == 3
    assert state.number_of_nodes() == 4
    assert state.number_of_edges() == 4
    assert {tlet, nmx, mx, b} == set(state.nodes())
    assert len(tlet.in_connectors) == 0
    assert set(tlet.out_connectors.keys()) == {"__out1", "__out2"}

    # The first down edge, will only delete one edge, due to the location of the fan
    #  in, which is a bit different compared to the location of the upper fan out.
    nb_rm_down_edge1 = dace.sdfg.utils.remove_edge_and_dangling_path(state, down_edge1)
    assert nb_rm_down_edge1 == 1
    assert state.number_of_nodes() == 4
    assert state.number_of_edges() == 3
    assert {tlet, nmx, mx, b} == set(state.nodes())
    assert len(tlet.in_connectors) == 0
    assert set(tlet.out_connectors.keys()) == {"__out2"}
    assert len(nmx.in_connectors) == 1
    assert len(nmx.out_connectors) == 1
    assert len(mx.in_connectors) == 1
    assert len(mx.out_connectors) == 1

    # This will remove all nodes.
    nb_rm_down_edge2 = dace.sdfg.utils.remove_edge_and_dangling_path(state, down_edge2)
    assert nb_rm_up_edge2 == 3
    assert state.number_of_nodes() == 0


if __name__ == '__main__':
    test_remove_edge_global_scope()
