# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from typing import Tuple, Union

from dace import nodes as dace_nodes
from dace.transformation.helpers import isolate_nested_sdfg


def count_node(sdfg: Union[dace.SDFG, dace.SDFGState], node_type, return_nodes: bool = False):
    states = [sdfg] if isinstance(sdfg, dace.SDFGState) else sdfg.states()
    found_nodes = []
    for state in states:
        for node in state.nodes():
            if isinstance(node, node_type):
                found_nodes.append(node)

    return found_nodes if return_nodes else len(found_nodes)


def count_writes(sdfg: dace.SDFG):
    """Count the number of write nodes.

    A write is defined as an incoming edge on an AccessNode. Thus this function
    essentially sums up the input degree of all AccessNodes.
    """
    nb_writes = 0
    for state in sdfg.states():
        for dnode in state.data_nodes():
            nb_writes += state.in_degree(dnode)
    return nb_writes


def _make_nested_sdfg_simple() -> dace.SDFG:
    """Make a simple nested SDFG.
    """
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


def _make_already_isloated_nested_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    """Creates a nested SDFG that is already isolated.
    """
    outer_sdfg = dace.SDFG("already_isolate_nested_sdfg")
    state = outer_sdfg.add_state(is_start_block=True)

    for name in "AB":
        outer_sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )

    inner_sdfg = _make_nested_sdfg_simple()
    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A"},
        outputs={"B"},
        symbol_mapping={},
    )
    state.add_edge(state.add_access("A"), None, nsdfg, "A", dace.Memlet("A[0:10]"))
    state.add_edge(nsdfg, "B", state.add_access("B"), None, dace.Memlet("B[0:10]"))
    outer_sdfg.validate()

    return outer_sdfg, state, nsdfg


def _make_non_empty_pre_set_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    """Generates an SDFG that has a non empty pre set.
    """
    outer_sdfg = dace.SDFG("non_empty_pre_set_nested_sdfg")
    state = outer_sdfg.add_state(is_start_block=True)

    for name in "ABT":
        outer_sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    outer_sdfg.arrays["T"].transient = True

    A, B, T = (state.add_access(name) for name in "ABT")

    state.add_mapped_tasklet(
        "outer_comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("T[__i]")},
        external_edges=True,
        input_nodes={A},
        output_nodes={T},
    )

    inner_sdfg = _make_nested_sdfg_simple()
    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A"},
        outputs={"B"},
        symbol_mapping={},
    )
    state.add_edge(T, None, nsdfg, "A", dace.Memlet("T[0:10]"))
    state.add_edge(nsdfg, "B", B, None, dace.Memlet("B[0:10]"))
    outer_sdfg.validate()

    return outer_sdfg, state, nsdfg


def _make_non_empty_post_state_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    """Generates an SDFG that will have an non empty post state and an empty pre state.
    """
    outer_sdfg = dace.SDFG("non_empty_post_set_nested_sdfg")
    state = outer_sdfg.add_state(is_start_block=True)

    for name in "ABT":
        outer_sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    outer_sdfg.arrays["T"].transient = True
    A, B, T = (state.add_access(name) for name in "ABT")

    inner_sdfg = _make_nested_sdfg_simple()
    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A"},
        outputs={"B"},
        symbol_mapping={},
    )
    state.add_edge(A, None, nsdfg, "A", dace.Memlet("A[0:10]"))
    state.add_edge(nsdfg, "B", T, None, dace.Memlet("T[0:10]"))

    state.add_mapped_tasklet(
        "outer_comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("T[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
        input_nodes={T},
        output_nodes={B},
    )
    outer_sdfg.validate()

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


def test_non_empty_pre_set():
    sdfg, state, nsdfg_node = _make_non_empty_pre_set_sdfg()

    assert sdfg.start_block is state
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 3
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert count_node(nsdfg_node.sdfg, dace_nodes.AccessNode) == 2
    assert count_node(nsdfg_node.sdfg, dace_nodes.MapEntry) == 1
    initial_writes = count_writes(sdfg)

    # Now perform the split. The post state should be empty, and the pre state should
    #  contain the map.
    pre_state, middle_state, post_state = isolate_nested_sdfg(state=state, nsdfg_node=nsdfg_node)

    sdfg.validate()
    assert sdfg.number_of_nodes() == 3
    assert sdfg.start_block is pre_state
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 4
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert initial_writes == count_writes(sdfg)

    pre_ac_nodes = count_node(pre_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(pre_ac_nodes) == 2
    assert {"A", "T"} == {n.data for n in pre_ac_nodes}
    assert count_node(pre_state, dace_nodes.MapEntry) == 1
    assert pre_state.number_of_nodes() == 5

    middle_ac_nodes = count_node(middle_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(middle_ac_nodes) == 2
    assert {"T", "B"} == {n.data for n in middle_ac_nodes}
    assert count_node(middle_state, dace_nodes.NestedSDFG) == 1
    assert middle_state.number_of_nodes() == 3

    assert post_state.number_of_nodes() == 0


def test_non_empty_post_set():
    sdfg, state, nsdfg_node = _make_non_empty_post_state_sdfg()

    assert sdfg.start_block is state
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 3
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert count_node(nsdfg_node.sdfg, dace_nodes.AccessNode) == 2
    assert count_node(nsdfg_node.sdfg, dace_nodes.MapEntry) == 1
    initial_writes = count_writes(sdfg)

    # Now perform the split. The pre state will be empty, but the map will be relocated to the post state.
    pre_state, middle_state, post_state = isolate_nested_sdfg(state=state, nsdfg_node=nsdfg_node)

    assert sdfg.number_of_nodes() == 3
    assert sdfg.start_block is pre_state
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 4
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert initial_writes == count_writes(sdfg)

    assert pre_state.number_of_nodes() == 0

    middle_ac_nodes = count_node(middle_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(middle_ac_nodes) == 2
    assert {"T", "A"} == {n.data for n in middle_ac_nodes}
    assert count_node(middle_state, dace_nodes.NestedSDFG) == 1
    assert middle_state.number_of_nodes() == 3

    post_ac_nodes = count_node(post_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(post_ac_nodes) == 2
    assert {"T", "B"} == {n.data for n in post_ac_nodes}
    assert count_node(post_state, dace_nodes.MapEntry) == 1
    assert post_state.number_of_nodes() == 5
