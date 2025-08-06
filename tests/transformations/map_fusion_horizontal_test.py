# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Union, Tuple, Type, Optional, List

import numpy as np

import dace
from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.transformation import dataflow as dftrans

from .map_fusion_vertical_test import count_nodes, unique_name

# NOTE: MapFusionHorizontal is essentially implemented in terms of `relocate_node()` which is
#   also used by `MapFusionVertical` thus the majority of tests if included there and not here.


def _make_horizontal_map_sdfg(common_ancestor: bool):
    sdfg = dace.SDFG(unique_name("parallel_maps_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    names = ["A", "B", "C", "D", "out"]
    for name in names:
        sdfg.add_array(
            name,
            shape=((10, 4) if name == "out" else (10, )),
            dtype=dace.float64,
            transient=False,
        )

    out = state.add_access("out")

    if common_ancestor:
        input_nodes = {state.add_access("A")}
    else:
        input_nodes = set()

    for i, name in enumerate(["A", "B", "C"]):
        it = f"__{i}"
        state.add_mapped_tasklet(
            f"comp_{i}",
            map_ranges={it: "0:10"},
            inputs={"__in": dace.Memlet(f"{name}[{it}]")},
            code=f"__out = __in + {i}.0",
            outputs={"__out": dace.Memlet(f"out[{it}, {i}]")},
            input_nodes=input_nodes,
            output_nodes={out},
            external_edges=True,
        )

    state.add_mapped_tasklet(
        "comp_4",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("A[__i]"),
            "__in2": dace.Memlet("D[__i]")
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet(f"out[__i, 3]")},
        input_nodes=input_nodes,
        output_nodes={out},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg, state


def _make_vertical_map_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(unique_name("serial_maps_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    names = ["a", "t", "b"]
    for name in names:
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    t = state.add_access("t")
    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.",
        outputs={"__out": dace.Memlet("t[__i]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("t[__i]")},
        code="__out = __in + 44.",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={t},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_vertical_map_fusion_common_ancestor_is_required():
    sdfg, _ = _make_horizontal_map_sdfg(common_ancestor=False)
    assert count_nodes(sdfg, nodes.AccessNode) == 6
    assert count_nodes(sdfg, nodes.MapExit) == 4

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal(only_if_common_ancestor=True)],
        validate=True,
        validate_all=True,
    )
    assert count == 0


def test_vertical_map_fusion_no_common_ancestor_not_required():
    sdfg, _ = _make_horizontal_map_sdfg(common_ancestor=False)
    assert count_nodes(sdfg, nodes.MapExit) == 4

    ac_nodes_before = count_nodes(sdfg, nodes.AccessNode, True)
    assert len(ac_nodes_before) == 6
    assert {ac.data for ac in ac_nodes_before} == {"A", "B", "C", "D", "out"}
    assert len([ac for ac in ac_nodes_before if ac.data == "A"]) == 2

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal(only_if_common_ancestor=False)],
        validate=True,
        validate_all=True,
    )
    assert count == 3
    assert count_nodes(sdfg, nodes.AccessNode) == 5
    assert count_nodes(sdfg, nodes.MapExit) == 1


def test_vertical_map_fusion_with_common_ancestor_is_required():
    sdfg, state = _make_horizontal_map_sdfg(common_ancestor=True)
    assert count_nodes(sdfg, nodes.MapExit) == 4

    ac_nodes_before = count_nodes(sdfg, nodes.AccessNode, True)
    assert len(ac_nodes_before) == 5
    assert {ac.data for ac in ac_nodes_before} == {"A", "B", "C", "D", "out"}
    ac_A_node = next(iter(ac for ac in ac_nodes_before if ac.data == "A"))
    assert state.out_degree(ac_A_node) == 2

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal(only_if_common_ancestor=True)],
        validate=True,
        validate_all=True,
    )
    assert count == 1
    assert count_nodes(sdfg, nodes.AccessNode) == 5
    assert count_nodes(sdfg, nodes.MapExit) == 3

    # Because of the consolidation it was reduced to 1 node.
    assert state.out_degree(ac_A_node) == 1


def test_vertical_map_fusion_with_common_ancestor_is_required_no_consolidation():
    sdfg, state = _make_horizontal_map_sdfg(common_ancestor=True)
    assert count_nodes(sdfg, nodes.MapExit) == 4

    ac_nodes_before = count_nodes(sdfg, nodes.AccessNode, True)
    assert len(ac_nodes_before) == 5
    assert {ac.data for ac in ac_nodes_before} == {"A", "B", "C", "D", "out"}
    ac_A_node = next(iter(ac for ac in ac_nodes_before if ac.data == "A"))
    assert state.out_degree(ac_A_node) == 2

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal(only_if_common_ancestor=True, never_consolidate_edges=True)],
        validate=True,
        validate_all=True,
    )
    assert count == 1
    assert count_nodes(sdfg, nodes.AccessNode) == 5
    assert count_nodes(sdfg, nodes.MapExit) == 3

    # Because consolidation is disabled, there are two edges to the same Map.
    ac_A_node_oedges = list(state.out_edges(ac_A_node))
    assert len(ac_A_node_oedges) == 2
    assert all(isinstance(e.dst, nodes.MapEntry) for e in ac_A_node_oedges)
    assert len({e.dst for e in ac_A_node_oedges}) == 1

    # Now look at the Map that was fused. It has 3 inputs, two from `A` and one from `D`.
    fused_map = ac_A_node_oedges[0].dst
    fused_map_iedges = list(state.in_edges(fused_map))
    assert len(fused_map_iedges) == 3
    assert all(isinstance(e.src, nodes.AccessNode) for e in fused_map_iedges)
    assert {e.src.data for e in fused_map_iedges} == {"A", "D"}


def test_vertical_maps_are_not_fused_horizontally():
    sdfg = _make_vertical_map_sdfg()
    assert count_nodes(sdfg, nodes.MapExit) == 2
    assert count_nodes(sdfg, nodes.AccessNode) == 3

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal(only_if_common_ancestor=False, never_consolidate_edges=True)],
        validate=True,
        validate_all=True,
    )
    assert count == 0


if __name__ == '__main__':
    test_vertical_map_fusion_common_ancestor_is_required()
    test_vertical_map_fusion_no_common_ancestor_not_required()
    test_vertical_map_fusion_with_common_ancestor_is_required()
    test_vertical_map_fusion_with_common_ancestor_is_required_no_consolidation()
    test_vertical_maps_are_not_fused_horizontally()
