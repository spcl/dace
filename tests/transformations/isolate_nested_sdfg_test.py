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


def _make_nested_sdfg_adding() -> dace.SDFG:
    """Make an SDFG that adds the inputs together.
    """
    sdfg = dace.SDFG("adding_nested_sdfg")
    state = sdfg.add_state(is_start_block=True)

    for name in "ABC":
        sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    state.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("A[__i]"),
            "__in2": dace.Memlet("B[__i]"),
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet("C[__i]")},
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


def _make_non_empty_pre_set_sdfg_2() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    outer_sdfg = dace.SDFG("non_empty_pre_set_nested_sdfg_2")
    state = outer_sdfg.add_state(is_start_block=True)

    for name in "ABCT":
        outer_sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    outer_sdfg.arrays["T"].transient = True

    A, B, C, T = (state.add_access(name) for name in "ABCT")

    state.add_mapped_tasklet(
        "outer_comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("B[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("T[__i]")},
        external_edges=True,
        input_nodes={B},
        output_nodes={T},
    )

    inner_sdfg = _make_nested_sdfg_adding()
    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A", "B"},
        outputs={"C"},
        symbol_mapping={},
    )

    state.add_edge(A, None, nsdfg, "A", dace.Memlet("A[0:10]"))
    state.add_edge(T, None, nsdfg, "B", dace.Memlet("T[0:10]"))
    state.add_edge(nsdfg, "C", C, None, dace.Memlet("C[0:10]"))
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


def _make_non_empty_post_state_sdfg_2() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    outer_sdfg = dace.SDFG("non_empty_post_set_nested_sdfg_2")
    state = outer_sdfg.add_state(is_start_block=True)

    for name in "ABC":
        outer_sdfg.add_array(
            name=name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    A, B, C = (state.add_access(name) for name in "ABC")

    inner_sdfg = _make_nested_sdfg_simple()
    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A"},
        outputs={"B"},
        symbol_mapping={},
    )
    state.add_edge(A, None, nsdfg, "A", dace.Memlet("A[0:10]"))
    state.add_edge(nsdfg, "B", B, None, dace.Memlet("B[0:10]"))

    state.add_mapped_tasklet(
        "outer_comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("C[__i]")},
        external_edges=True,
        input_nodes={A},
        output_nodes={C},
    )
    outer_sdfg.validate()

    return outer_sdfg, state, nsdfg


def _make_multi_path_nested_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG]:
    """Creates an SDFG that has a path around the nested SDFG.
    """
    outer_sdfg = dace.SDFG("multi_path_nested_sdfg")
    state = outer_sdfg.add_state(is_start_block=True)

    aname_small = ["A", "T1", "T2", "T3"]
    aname_big = ["T4", "B"]

    for name in aname_small + aname_big:
        outer_sdfg.add_array(
            name=name,
            shape=((10, ) if name in aname_small else (20, )),
            dtype=dace.float64,
            transient=(len(name) != 1),
        )
    A, T1, T2, T3, T4, B = (state.add_access(name) for name in aname_small + aname_big)

    state.add_mapped_tasklet(
        "comp_pre1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("T1[__i]")},
        external_edges=True,
        input_nodes={A},
        output_nodes={T1},
    )

    inner_sdfg = _make_nested_sdfg_adding()
    nsdfg_node = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"A", "B"},
        outputs={"C"},
        symbol_mapping={},
    )
    state.add_edge(A, None, nsdfg_node, "A", dace.Memlet("A[0:10]"))
    state.add_edge(T1, None, nsdfg_node, "B", dace.Memlet("T1[0:10]"))

    state.add_edge(nsdfg_node, "C", T2, None, dace.Memlet("T2[0:10]"))
    state.add_mapped_tasklet(
        "comp_post1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("T3[__i]")},
        external_edges=True,
        input_nodes={A},
        output_nodes={T3},
    )

    state.add_nedge(T3, T4, dace.Memlet("T3[0:10] -> [0:10]"))
    state.add_nedge(T2, T4, dace.Memlet("T2[0:10] -> [10:20]"))
    state.add_mapped_tasklet(
        "comp_post2",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("T4[__i]")},
        code="__out = __in + 3.0",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
        input_nodes={T4},
        output_nodes={B},
    )
    outer_sdfg.validate()

    return outer_sdfg, state, nsdfg_node


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


def test_non_empty_pre_set_2():
    sdfg, state, nsdfg_node = _make_non_empty_pre_set_sdfg_2()

    assert sdfg.start_block is state
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 4
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert count_node(nsdfg_node.sdfg, dace_nodes.AccessNode) == 3
    assert count_node(nsdfg_node.sdfg, dace_nodes.MapEntry) == 1
    initial_writes = count_writes(sdfg)

    # Now we apply the isolation. The pre set is not empty, but only one AccessNode,
    #  that is used as input of the nested SDFG, is involved. So the other must be
    #  ignored.
    pre_state, middle_state, post_state = isolate_nested_sdfg(state=state, nsdfg_node=nsdfg_node)

    sdfg.validate()
    assert sdfg.number_of_nodes() == 3
    assert sdfg.start_block is pre_state
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 5
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert initial_writes == count_writes(sdfg)

    pre_ac_nodes = count_node(pre_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(pre_ac_nodes) == 2
    assert {"B", "T"} == {n.data for n in pre_ac_nodes}
    assert count_node(pre_state, dace_nodes.MapEntry) == 1

    middle_ac_nodes = count_node(middle_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(middle_ac_nodes) == 3
    assert {"A", "C", "T"} == {n.data for n in middle_ac_nodes}
    assert count_node(middle_state, dace_nodes.NestedSDFG) == 1

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


def test_non_empty_post_set_2():
    sdfg, state, nsdfg_node = _make_non_empty_post_state_sdfg_2()

    assert sdfg.start_block is state
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 3
    assert count_node(sdfg, dace_nodes.MapEntry) == 1
    assert count_node(nsdfg_node.sdfg, dace_nodes.AccessNode) == 2
    assert count_node(nsdfg_node.sdfg, dace_nodes.MapEntry) == 1
    initial_writes = count_writes(sdfg)

    # Now apply the isolation. The post set here uses an input of the nested SDFG.
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
    assert {"A", "B"} == {n.data for n in middle_ac_nodes}
    assert count_node(middle_state, dace_nodes.NestedSDFG) == 1

    post_ac_nodes = count_node(post_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(post_ac_nodes) == 2
    assert {"A", "C"} == {n.data for n in post_ac_nodes}
    assert count_node(post_state, dace_nodes.MapEntry) == 1


def test_multi_path_islolation():
    sdfg, state, nsdfg_node = _make_multi_path_nested_sdfg()

    assert sdfg.start_block is state
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, dace_nodes.NestedSDFG) == 1
    assert count_node(sdfg, dace_nodes.AccessNode) == 6
    assert count_node(sdfg, dace_nodes.MapEntry) == 3
    assert count_node(nsdfg_node.sdfg, dace_nodes.AccessNode) == 3
    assert count_node(nsdfg_node.sdfg, dace_nodes.MapEntry) == 1
    initial_writes = count_writes(sdfg)

    # Now perform the split
    pre_state, middle_state, post_state = isolate_nested_sdfg(state=state, nsdfg_node=nsdfg_node)

    assert sdfg.number_of_nodes() == 3
    assert sdfg.start_block is pre_state
    assert initial_writes == count_writes(sdfg)

    pre_ac_nodes = count_node(pre_state, dace_nodes.AccessNode, return_nodes=True)
    pre_me_nodes = count_node(pre_state, dace_nodes.MapEntry, return_nodes=True)
    assert len(pre_ac_nodes) == 2
    assert {"A", "T1"} == {n.data for n in pre_ac_nodes}
    assert len(pre_me_nodes) == 1
    assert all(me.map.label.startswith("comp_pre") for me in pre_me_nodes)

    middle_ac_nodes = count_node(middle_state, dace_nodes.AccessNode, return_nodes=True)
    assert len(middle_ac_nodes) == 3
    assert {"A", "T1", "T2"} == {n.data for n in middle_ac_nodes}
    assert count_node(middle_state, dace_nodes.NestedSDFG) == 1

    post_ac_nodes = count_node(post_state, dace_nodes.AccessNode, return_nodes=True)
    post_me_nodes = count_node(post_state, dace_nodes.MapEntry, return_nodes=True)
    assert len(post_ac_nodes) == 5
    assert {"A", "T3", "T2", "T4", "B"} == {n.data for n in post_ac_nodes}
    assert len(post_me_nodes) == 2
    assert all(me.map.label.startswith("comp_post") for me in post_me_nodes)
