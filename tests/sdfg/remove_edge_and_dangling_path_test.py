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
    dace.sdfg.utils.remove_edge_and_dangling_path(state, up_edge)
    assert a not in state.nodes()
    assert tlet in state.nodes()
    assert b in state.nodes()
    assert sdfg.arrays.keys() == {"a", "b"}
    assert len(tlet.in_connectors) == 0
    assert state.in_degree(tlet) == 0
    assert tlet.out_connectors.keys() == {"__out"}
    assert state.out_degree(tlet) == 1

    # If we now also delete the `down_edge` then the state will become empty.
    dace.sdfg.utils.remove_edge_and_dangling_path(state, down_edge)
    assert state.number_of_nodes() == 0


if __name__ == '__main__':
    test_remove_edge_global_scope()
