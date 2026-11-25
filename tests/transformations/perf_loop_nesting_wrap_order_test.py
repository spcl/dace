# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrapping a tasklet in a trivial map must route its ordering (empty) edges through the scope without connectors.

The wrap gave an empty out-edge an ``IN_out_<i>`` / ``OUT_out_<i>`` pair on the map exit. InlineSDFG's order tasklets
later rerouted the outer half, leaving ``IN_out_1`` unpaired; ``TrivialMapElimination`` then followed it with
``memlet_path`` and raised ``StopIteration`` (QE ``vexx_bp_k_gpu``, ``canonicalize(target="gpu")``).
"""

import dace
import dace.transformation.dataflow.perf_loop_nesting as pln
from dace.sdfg import nodes
from dace.transformation.dataflow import TrivialMapElimination


def build_state() -> tuple[dace.SDFG, dace.SDFGState, nodes.Tasklet, nodes.AccessNode]:
    """``A[0] = 1`` in tasklet ``t``, ordered by an empty edge before the write ``B[0] = 2``."""
    sdfg = dace.SDFG("perf_loop_nesting_wrap_order")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state("main")
    t = state.add_tasklet("t", {}, {"__out"}, "__out = 1")
    state.add_edge(t, "__out", state.add_write("A"), None, dace.Memlet("A[0]"))
    u = state.add_tasklet("u", {}, {"__out"}, "__out = 2")
    b = state.add_write("B")
    state.add_edge(u, "__out", b, None, dace.Memlet("B[0]"))
    state.add_nedge(t, u, dace.Memlet())
    sdfg.validate()
    return sdfg, state, t, u


def test_an_ordering_edge_crosses_the_wrap_without_connectors():
    sdfg, state, t, u = build_state()
    pln._wrap_tasklet_in_trivial_map(state, t)
    sdfg.validate()
    for node in state.nodes():
        if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
            for e in state.all_edges(node):
                if e.data.is_empty():
                    assert e.src_conn is None and e.dst_conn is None, (e.src, e.src_conn, e.dst, e.dst_conn)
    exit_ = next(n for n in state.nodes() if isinstance(n, nodes.MapExit))
    assert state.edges_between(exit_, u), "the order t before u must survive the wrap"


def test_the_wrap_eliminates_and_keeps_the_order():
    sdfg, state, t, u = build_state()
    pln._wrap_tasklet_in_trivial_map(state, t)
    # What a later reroute does to the outer half of an ordering edge: drop it, order through another node.
    exit_ = next(n for n in state.nodes() if isinstance(n, nodes.MapExit))
    for e in list(state.out_edges(exit_)):
        if e.data.is_empty():
            state.remove_edge(e)
            state.add_nedge(exit_, u, dace.Memlet())
    assert sdfg.apply_transformations_repeated(TrivialMapElimination) == 1
    sdfg.validate()
    assert any(e.dst is u for e in state.out_edges(t)) or state.edges_between(t, u), "t must stay ordered before u"


if __name__ == "__main__":
    test_an_ordering_edge_crosses_the_wrap_without_connectors()
    test_the_wrap_eliminates_and_keeps_the_order()
