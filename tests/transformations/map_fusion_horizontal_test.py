# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
from typing import Tuple

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation import dataflow as dftrans

from .map_fusion_vertical_test import count_nodes, unique_name

# NOTE: MapFusionHorizontal is essentially implemented in terms of `relocate_node()` which is
#   also used by `MapFusionVertical` thus the majority of tests is performed there and not here.


def _make_horizontal_map_sdfg(common_ancestor: bool):
    sdfg = dace.SDFG(unique_name("horizontal_maps_sdfg"))
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
    sdfg = dace.SDFG(unique_name("vertical_maps_sdfg"))
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


def _make_simple_horizontal_map_sdfg() -> Tuple[dace.SDFG, dace.nodes.MapEntry, dace.nodes.MapEntry]:
    sdfg = dace.SDFG(unique_name("horizontal_simple"))
    state = sdfg.add_state(is_start_block=True)

    for aname in "abc":
        sdfg.add_array(
            aname,
            shape=(10, 20),
            dtype=dace.float64,
            transient=False,
        )
    a = state.add_access("a")

    _, me_a, _ = state.add_mapped_tasklet(
        "comp_a",
        map_ranges={
            "__i": "0:10",
            "__j": "0:20",
        },
        inputs={"__in": dace.Memlet("a[__i, __j]")},
        code="__out = __in + 1.2",
        outputs={"__out": dace.Memlet("b[__i, __j]")},
        input_nodes={a},
        external_edges=True,
    )
    _, me_b, _ = state.add_mapped_tasklet(
        "comp_b",
        map_ranges={
            "__i": "0:10",
            "__j": "0:20",
        },
        inputs={"__in": dace.Memlet("a[__i, __j]")},
        code="__out = __in + 1.3",
        outputs={"__out": dace.Memlet("c[__i, __j]")},
        input_nodes={a},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, me_a, me_b


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


@pytest.mark.parametrize("first_order", [True, False])
def test_deterministic_label_in_horizontal_map_fusion(first_order: bool):
    sdfg, me_a, me_b = _make_simple_horizontal_map_sdfg()
    assert {me_a, me_b} == set(count_nodes(sdfg, dace.nodes.MapEntry, True))

    expected_final_label = me_a.map.label
    assert expected_final_label < me_b.map.label

    if first_order:
        dftrans.MapFusionHorizontal.apply_to(
            sdfg=sdfg,
            first_parallel_map_entry=me_a,
            second_parallel_map_entry=me_b,
        )
        # Always preserve the scope nodes of the first Map,
        assert {me_a} == set(count_nodes(sdfg, dace.nodes.MapEntry, True))
        final_me = me_a

    else:
        dftrans.MapFusionHorizontal.apply_to(
            sdfg=sdfg,
            first_parallel_map_entry=me_b,
            second_parallel_map_entry=me_a,
        )
        # Always preserve the scope nodes of the first Map,
        assert {me_b} == set(count_nodes(sdfg, dace.nodes.MapEntry, True))
        final_me = me_b

    # Regardless in which order they are fused, the label is deterministic.
    assert expected_final_label == final_me.map.label


def _make_shared_dynamic_map_range_sdfg(binding: str) -> dace.SDFG:
    """Two parallel Maps whose ranges are both bounded by a dynamic map range named ``lim``.

    ``binding`` selects how the two bounds are read: ``same_node`` (one AccessNode feeding both),
    ``same_data`` (two AccessNodes of the same never-written array), ``other_data`` (two different
    arrays) or ``written_data`` (same array, but a tasklet writes it in this very state).
    """
    sdfg = dace.SDFG(unique_name(f"shared_dmr_{binding}"))
    for name in ("bound", "other_bound"):
        sdfg.add_array(name, shape=(1, ), dtype=dace.int64, transient=False)
    for name in ("A", "B"):
        sdfg.add_array(name, shape=(10, ), dtype=dace.float64, transient=False)
    state = sdfg.add_state(is_start_block=True)

    first_source = state.add_access("bound")
    if binding == "same_node":
        second_source = first_source
    elif binding == "other_data":
        second_source = state.add_access("other_bound")
    else:
        second_source = state.add_access("bound")
    if binding == "written_data":
        writer = state.add_tasklet("set_bound", {}, {"__out"}, "__out = 4")
        state.add_edge(writer, "__out", state.add_access("bound"), None, dace.Memlet("bound[0]"))

    for name, source, value in (("A", first_source, 1.0), ("B", second_source, 2.0)):
        _, map_entry, _ = state.add_mapped_tasklet(
            f"comp_{name}",
            map_ranges={"__i": "0:lim"},
            inputs={},
            code=f"__out = {value}",
            outputs={"__out": dace.Memlet(f"{name}[__i]")},
            output_nodes={name: state.add_access(name)},
            external_edges=True,
        )
        map_entry.add_in_connector("lim")
        state.add_edge(source, None, map_entry, "lim", dace.Memlet(f"{source.data}[0]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("binding", ["same_node", "same_data"])
def test_horizontal_fusion_drops_an_equal_dynamic_map_range(binding: str):
    """A dynamic map range both Maps bind to the same value is redundant, so it is dropped.

    Relocation cannot rename a colliding symbol; before the fix any collision raised
    ``NotImplementedError`` from the middle of ``apply()``, leaving the state half-rewired.
    """
    sdfg = _make_shared_dynamic_map_range_sdfg(binding)
    state = sdfg.start_block

    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 1

    fused_entry = count_nodes(state, nodes.MapEntry, return_nodes=True)
    assert len(fused_entry) == 1
    # The surviving Map still binds `lim` exactly once, through exactly one edge.
    assert {c for c in fused_entry[0].in_connectors if not c.startswith("IN_")} == {"lim"}
    assert len(list(state.in_edges_by_connector(fused_entry[0], "lim"))) == 1
    assert count_nodes(state, nodes.AccessNode) == 3  # bound, A, B -- the duplicate source is gone
    sdfg.validate()


def test_horizontal_fusion_keeps_the_dynamic_map_range_bound():
    """Dropping the redundant binding must not change the iteration space of the fused Map."""
    sdfg = _make_shared_dynamic_map_range_sdfg("same_data")
    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 1
    csdfg = sdfg.compile()

    bound = np.full((1, ), 4, dtype=np.int64)
    args = {name: np.full((10, ), -1.0, dtype=np.float64) for name in ("A", "B")}
    expected = {name: np.where(np.arange(10) < 4, value, -1.0) for name, value in (("A", 1.0), ("B", 2.0))}

    csdfg(bound=bound, other_bound=bound.copy(), **args)
    for name in args:
        assert np.array_equal(args[name], expected[name]), f"fused Map did not honor the dynamic bound for {name}"


@pytest.mark.parametrize("binding", ["other_data", "written_data"])
def test_horizontal_fusion_rejects_a_disagreeing_dynamic_map_range(binding: str):
    """Two Maps binding the same symbol to different values cannot fuse -- and must not mutate."""
    sdfg = _make_shared_dynamic_map_range_sdfg(binding)
    state = sdfg.start_block
    before = (state.number_of_nodes(), state.number_of_edges())

    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 0

    assert (state.number_of_nodes(), state.number_of_edges()) == before
    assert count_nodes(state, nodes.MapEntry) == 2
    sdfg.validate()


if __name__ == '__main__':
    test_vertical_map_fusion_common_ancestor_is_required()
    test_vertical_map_fusion_no_common_ancestor_not_required()
    test_vertical_map_fusion_with_common_ancestor_is_required()
    test_vertical_map_fusion_with_common_ancestor_is_required_no_consolidation()
    test_vertical_maps_are_not_fused_horizontally()
    test_deterministic_label_in_horizontal_map_fusion(first_order=True)
    test_deterministic_label_in_horizontal_map_fusion(first_order=False)
    test_horizontal_fusion_drops_an_equal_dynamic_map_range("same_node")
    test_horizontal_fusion_drops_an_equal_dynamic_map_range("same_data")
    test_horizontal_fusion_keeps_the_dynamic_map_range_bound()
    test_horizontal_fusion_rejects_a_disagreeing_dynamic_map_range("other_data")
    test_horizontal_fusion_rejects_a_disagreeing_dynamic_map_range("written_data")
