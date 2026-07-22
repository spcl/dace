# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Tuple

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


def test_horizontal_fusion_preserves_distinct_ordering_in_edges():
    """Regression: fusing two parallel Maps that each carry an empty-Memlet
    ordering (WAW) in-edge from a *different* source AccessNode must preserve
    BOTH ordering edges on the fused Map.

    The empty-Memlet dedup in ``relocate_nodes`` used to key on the edge
    destination alone while iterating ``all_edges(to_node)``. Every empty
    in-edge shares ``dst == to_node``, so all but one were removed -- silently
    dropping a real ordering dependency, and (because the survivor depended on
    edge iteration order) producing an order-dependent miscompile. This showed
    up as npbench ``cavity_flow`` at the larger dataset: the boundary writes to
    ``u`` lost their sequencing when it got rerouted through ``v``. The dedup
    now keys on the ``(src, dst)`` pair, so distinct-source ordering edges
    survive while genuine duplicates still collapse.
    """
    sdfg = dace.SDFG(unique_name("horizontal_ordering_edges"))
    state = sdfg.add_state(is_start_block=True)
    for name in ("A", "u", "v", "su", "sv"):
        sdfg.add_array(name, shape=(10, ), dtype=dace.float64, transient=False)

    a = state.add_access("A")
    su = state.add_access("su")  # stand-in for a prior writer of u (ordering source)
    sv = state.add_access("sv")  # stand-in for a prior writer of v (ordering source)

    _, me_u, _ = state.add_mapped_tasklet(
        "comp_u",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("u[__i]")},
        input_nodes={a},
        external_edges=True,
    )
    _, me_v, _ = state.add_mapped_tasklet(
        "comp_v",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("v[__i]")},
        input_nodes={a},
        external_edges=True,
    )
    # Distinct-source empty-Memlet ordering in-edges, one into each Map entry.
    state.add_nedge(su, me_u, dace.Memlet())
    state.add_nedge(sv, me_v, dace.Memlet())
    sdfg.validate()

    def empty_in_sources(me):
        return sorted(e.src.data for e in state.in_edges(me)
                      if e.data.is_empty() and isinstance(e.src, nodes.AccessNode))

    assert empty_in_sources(me_u) == ["su"]
    assert empty_in_sources(me_v) == ["sv"]

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal(only_if_common_ancestor=True)],
        validate=True,
        validate_all=True,
    )
    assert count == 1, "the two same-range parallel maps must fuse"

    fused = [me for me in state.nodes() if isinstance(me, nodes.MapEntry)]
    assert len(fused) == 1
    # BOTH ordering dependencies must survive on the fused Map -- neither array's
    # write-ordering chain may be dropped by the empty-Memlet dedup.
    assert empty_in_sources(fused[0]) == ["su", "sv"], \
        "distinct-source empty ordering in-edges must not be deduped away"


def _make_slicing_map(state: dace.SDFGState, name: str, source: nodes.AccessNode, target: str,
                      slices: int) -> Tuple[nodes.MapEntry, nodes.MapExit]:
    """A Map whose body writes ``slices`` disjoint columns of ``target`` from ``slices`` tasklets.

    All of those writes land on the SAME ``IN_1`` of the MapExit -- one connector carrying several
    edges, which is legal (the one-edge-per-connector rule binds a scope exit's *out*-connectors,
    not its in-connectors). That is the shape CLOUDSC arrives in: a Fortran body assigning
    ``zqxn2d(jl, jk, 1:5)`` becomes five nested SDFGs that ``InlineSDFGs`` flattens into five
    tasklets all writing through one MapExit connector. ``add_mapped_tasklet`` gives each tasklet
    its own connector, so the wiring is explicit here.
    """
    entry, exit_node = state.add_map(name, {"__i": "0:10"})

    entry.add_in_connector("IN_1")
    entry.add_out_connector("OUT_1")
    state.add_edge(source, None, entry, "IN_1", dace.Memlet("A[0:10]"))

    exit_node.add_in_connector("IN_1")
    exit_node.add_out_connector("OUT_1")
    for column in range(slices):
        tasklet = state.add_tasklet(f"{name}_t{column}", {"__in"}, {"__out"}, f"__out = __in + {column}.0")
        state.add_edge(entry, "OUT_1", tasklet, "__in", dace.Memlet("A[__i]"))
        state.add_edge(tasklet, "__out", exit_node, "IN_1", dace.Memlet(f"{target}[__i, {column}]"))
    state.add_edge(exit_node, "OUT_1", state.add_access(target), None, dace.Memlet(f"{target}[0:10, 0:{slices}]"))
    return entry, exit_node


def test_horizontal_fusion_relocates_a_multi_edge_connector_once():
    """Regression: a scope connector carrying SEVERAL edges must be relocated exactly once.

    ``relocate_nodes`` moves a whole ``IN_x`` / ``OUT_x`` group in one step but iterated over a
    snapshot of the source node's *edges*. A connector holding n edges was therefore visited n
    times: the first visit moved the group, and each of the n-1 later visits found the group
    already empty, minted a fresh connector pair on the surviving node and attached nothing to it
    -- ``InvalidSDFGNodeError: Dangling in-connector IN_4``. Surfaced by CLOUDSC ``full_cpu``,
    whose MapExits carry five ``zqxn2d[i, j, 0..4]`` edges on one connector.
    """
    sdfg = dace.SDFG(unique_name("horizontal_multi_edge_connector"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_array("A", shape=(10, ), dtype=dace.float64, transient=False)
    for name in ("out1", "out2"):
        sdfg.add_array(name, shape=(10, 3), dtype=dace.float64, transient=False)

    a = state.add_access("A")
    _, exit1 = _make_slicing_map(state, "comp_1", a, "out1", slices=3)
    _, exit2 = _make_slicing_map(state, "comp_2", a, "out2", slices=3)
    sdfg.validate()
    edges_before = len(state.in_edges(exit1)) + len(state.in_edges(exit2))
    assert edges_before == 6

    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionHorizontal()],
        validate=False,
        validate_all=False,
    )
    assert count == 1, "the two same-range parallel maps must fuse"

    fused_exits = count_nodes(state, nodes.MapExit, return_nodes=True)
    assert len(fused_exits) == 1
    fused_exit = fused_exits[0]

    dangling_in = sorted(c for c in fused_exit.in_connectors
                         if not any(e.dst_conn == c for e in state.in_edges(fused_exit)))
    dangling_out = sorted(c for c in fused_exit.out_connectors
                          if not any(e.src_conn == c for e in state.out_edges(fused_exit)))
    assert not dangling_in, f"MapExit kept in-connectors with no edge: {dangling_in}"
    assert not dangling_out, f"MapExit kept out-connectors with no edge: {dangling_out}"
    # Nothing may be lost either: every slice write of both Maps still reaches the fused exit.
    assert len(state.in_edges(fused_exit)) == edges_before
    sdfg.validate()


if __name__ == '__main__':
    test_horizontal_fusion_relocates_a_multi_edge_connector_once()
    test_horizontal_fusion_preserves_distinct_ordering_in_edges()
    test_vertical_map_fusion_common_ancestor_is_required()
    test_vertical_map_fusion_no_common_ancestor_not_required()
    test_vertical_map_fusion_with_common_ancestor_is_required()
    test_vertical_map_fusion_with_common_ancestor_is_required_no_consolidation()
    test_vertical_maps_are_not_fused_horizontally()
    test_deterministic_label_in_horizontal_map_fusion(first_order=True)
    test_deterministic_label_in_horizontal_map_fusion(first_order=False)
