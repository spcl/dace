# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A cyclic state is left to validation by the ordering-edge passes, with a warning that names the cycle.

``nx.topological_sort`` raised ``NetworkXUnfeasible`` from inside ``guards_nothing`` when a transformation left a
cycle in a state, with no word on which state or which nodes: the whole canonicalization died on one state.
"""

import pytest

import dace
from dace import Memlet
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.redundant_ordering_edge_elimination import RedundantOrderingEdgeElimination
from dace.transformation.passes.canonicalize.route_ordering_through_data import RouteOrderingThroughData

N = dace.symbol("N")


def nest_with_an_edge_back_into_it():
    """``x -> nest -> y`` plus an ordering edge ``y -> nest``: the nest feeds ``y`` and waits on it."""
    sdfg = dace.SDFG("cyclic_state")
    for name in ("x", "y"):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    inner = dace.SDFG("inner")
    inner.add_array("a", [N], dace.float64)
    inner.add_array("b", [N], dace.float64)
    inner.add_state("body", is_start_block=True).add_mapped_tasklet(
        "copy",
        {"i": "0:N"},
        {"v": Memlet("a[i]")},
        "w = v",
        {"w": Memlet("b[i]")},
        external_edges=True,
    )
    nest = state.add_nested_sdfg(inner, {"a"}, {"b"}, symbol_mapping={"N": N})
    state.add_edge(state.add_read("x"), None, nest, "a", Memlet("x[0:N]"))
    written = state.add_write("y")
    state.add_edge(nest, "b", written, None, Memlet("y[0:N]"))
    state.add_nedge(written, nest, Memlet())
    return sdfg, state, nest


@pytest.mark.parametrize("pass_type", [RouteOrderingThroughData, RedundantOrderingEdgeElimination])
def test_a_cyclic_state_is_skipped_with_a_warning_that_names_the_cycle(pass_type):
    sdfg, state, nest = nest_with_an_edge_back_into_it()
    edges_before = state.number_of_edges()

    with pytest.warns(UserWarning, match=r"State cyclic_state/main has a cycle \(.*NestedSDFG"):
        pass_type().apply_pass(sdfg, {})

    assert state.number_of_edges() == edges_before, "the cyclic state is left untouched"
    assert any(isinstance(n, nodes.NestedSDFG) for n in state.nodes())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
