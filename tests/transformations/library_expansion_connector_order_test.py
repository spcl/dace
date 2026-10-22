# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Expanding a library node into a nested SDFG keeps the node's connectors, in the node's order."""

import warnings

import dace
from dace.libraries.blas import Dot


def make_dot_sdfg() -> tuple[dace.SDFG, Dot]:
    sdfg = dace.SDFG("expansion_connector_order")
    sdfg.add_array("x", [8], dace.float64)
    sdfg.add_array("y", [8], dace.float64)
    sdfg.add_array("result", [1], dace.float64)
    state = sdfg.add_state()
    dot = Dot("dot")
    dot.implementation = "pure"
    state.add_edge(state.add_read("x"), None, dot, "_x", dace.Memlet("x[0:8]"))
    state.add_edge(state.add_read("y"), None, dot, "_y", dace.Memlet("y[0:8]"))
    state.add_edge(dot, "_result", state.add_write("result"), None, dace.Memlet("result[0]"))
    return sdfg, dot


def test_an_expanded_library_node_keeps_its_connector_order_without_warnings():
    sdfg, dot = make_dot_sdfg()
    node_inputs, node_outputs = list(dot.in_connectors), list(dot.out_connectors)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdfg.expand_library_nodes()

    (nested,) = [node for node in sdfg.start_state.nodes() if isinstance(node, dace.nodes.NestedSDFG)]
    assert list(nested.in_connectors) == node_inputs
    assert list(nested.out_connectors) == node_outputs
    sdfg.validate()


if __name__ == "__main__":
    test_an_expanded_library_node_keeps_its_connector_order_without_warnings()
