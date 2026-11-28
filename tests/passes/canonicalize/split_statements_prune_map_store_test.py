# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pruning a clone cuts each map that stores into the suppressed output, IN_ and OUT_ connector together."""

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.split_statements import drop_dataless_access_nodes, suppress_writes


def two_maps_into_one_node(name: str, transient: bool) -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(f"prune_map_store_{name}")
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("x", [8], dace.float64, transient=transient)
    sdfg.add_array("y", [8], dace.float64)
    state = sdfg.add_state()
    x = state.add_access("x")
    for lo, hi in ((0, 3), (4, 7)):
        state.add_mapped_tasklet(
            f"half_{lo}",
            {"i": f"{lo}:{hi + 1}"},
            {"v": dace.Memlet("a[i]")},
            "o = v + 1.0",
            {"o": dace.Memlet("x[i]")},
            external_edges=True,
            output_nodes={"x": x},
        )
    state.add_mapped_tasklet(
        "keep", {"i": "0:8"}, {"v": dace.Memlet("a[i]")}, "o = v * 2.0", {"o": dace.Memlet("y[i]")}, external_edges=True
    )
    return sdfg, state


def no_dangling_exit_connector(sdfg: dace.SDFG, state: dace.SDFGState) -> None:
    for xit in [n for n in state.nodes() if isinstance(n, nodes.MapExit)]:
        out_conns = {e.src_conn for e in state.out_edges(xit)}
        assert all("OUT_" + e.dst_conn[3:] in out_conns for e in state.in_edges(xit))
    sdfg.validate()


def test_suppress_writes_cuts_every_map_store_into_a_shared_node():
    sdfg, state = two_maps_into_one_node("suppress", transient=False)
    suppress_writes(sdfg, "x")
    no_dangling_exit_connector(sdfg, state)
    assert not any(isinstance(n, nodes.AccessNode) and n.data == "x" for n in state.nodes())
    assert len([n for n in state.nodes() if isinstance(n, nodes.MapExit)]) == 1


def test_drop_dataless_access_nodes_cuts_the_maps_storing_an_unread_transient():
    sdfg, state = two_maps_into_one_node("drop", transient=True)
    drop_dataless_access_nodes(sdfg)
    no_dangling_exit_connector(sdfg, state)
    assert len([n for n in state.nodes() if isinstance(n, nodes.MapExit)]) == 1


if __name__ == "__main__":
    test_suppress_writes_cuts_every_map_store_into_a_shared_node()
    test_drop_dataless_access_nodes_cuts_the_maps_storing_an_unread_transient()
