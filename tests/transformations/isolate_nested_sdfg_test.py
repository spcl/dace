# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from typing import Tuple, Union

from dace import nodes as dace_nodes
from dace.transformation.helpers import isolate_nested_sdfg


def count_node(sdfg: Union[dace.SDFG, dace.SDFGState], node_type):
    states = [sdfg] if isinstance(sdfg, dace.SDFGState) else sdfg.states()
    nb_nodes = 0
    for state in states:
        for node in state.nodes():
            if isinstance(node, node_type):
                nb_nodes += 1
    return nb_nodes


def _make_already_isloated_nested_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    """Creates a nested SDFG that is already isolated.
    """

    def _make_nested_sdfg() -> dace.SDFG:
        sdfg = dace.SDFG("nested_sdfg")
        state = sdfg.add_state(is_start_block=True)

        for name in "AB":
            sdfg.add_array(
                name=name,
                shape=(10, ),
                dtype=dace.float64,
                transient=False,
            )
        state.add_mapped_tasklet(
            "comp",
            map_ranges={"__i": "1:9"},
            inputs={"__in": dace.Memlet("A[__i]")},
            code="__out = __in + 1.0",
            outputs={"__out": dace.Memlet("B[__i]")},
            external_edges=True,
        )
        sdfg.validate()
        return sdfg

    outer_sdfg = dace.SDFG("already_isolate_nested_sdfg")
    state = outer_sdfg.add_state(is_start_block=True)

    for name in "AB":
        outer_sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )

    inner_sdfg = _make_nested_sdfg()

    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A"},
        outputs={"B"},
        symbol_mapping={},
    )
    state.add_edge(state.add_access("A"), None, nsdfg, "A", dace.Memlet("A[0:10]"))
    state.add_edge(nsdfg, "B", state.add_access("B"), None, dace.Memlet("B[0:10]"))

    return outer_sdfg, state, nsdfg


def test_already_isolated_sdfg():
    sdfg, state, nsdfg_node = _make_already_isloated_nested_sdfg()

    assert sdfg.start_block is state
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 2
    assert count_node(nsdfg_node.sdfg, dace_nodes.AccessNode) == 2
    assert count_node(nsdfg_node.sdfg, dace_nodes.MapEntry) == 1

    # Now perform the split, because the nested SDFG is already isolated, the pre
    #  and post states should be empty.
    pre_state, middle_state, post_state = isolate_nested_sdfg(state=state, nsdfg_node=nsdfg_node)

    sdfg.validate()
    assert sdfg.number_of_nodes() == 3
    assert sdfg.start_block is pre_state
    assert pre_state.number_of_nodes() == 0
    assert post_state.number_of_nodes() == 0
    assert count_node(middle_state, dace_nodes.NestedSDFG) == 1
    assert count_node(middle_state, dace_nodes.AccessNode) == 2
