# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import os
import sys
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation import dataflow as dftrans
from dace.transformation.dataflow import map_fusion_helper as mfhelper

from tests.helpers.isolation import call_in_child, exit_code

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
    assert fused_entry[0].dynamic_input_connectors == {"lim"}
    assert len(list(state.in_edges_by_connector(fused_entry[0], "lim"))) == 1
    assert count_nodes(state, nodes.AccessNode) == 3  # bound, A, B -- the duplicate source is gone
    sdfg.validate()


def test_horizontal_fusion_keeps_the_dynamic_map_range_bound():
    """Dropping the redundant binding must not change the iteration space of the fused Map.

    Run in a spawned child so a miscompiled kernel segfaulting cannot take the session down. Spawn,
    not fork: this process has already run compiled kernels, so a fork child would inherit an
    OpenMP team it does not own and deadlock instead of reporting anything.
    """
    sdfg = _make_shared_dynamic_map_range_sdfg("same_data")
    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 1

    bound = np.full((1, ), 4, dtype=np.int64)
    args = {name: np.full((10, ), -1.0, dtype=np.float64) for name in ("A", "B")}
    expected = {name: np.where(np.arange(10) < 4, value, -1.0) for name, value in (("A", 1.0), ("B", 2.0))}

    kwargs = dict(args, bound=bound, other_bound=bound.copy())
    code = exit_code(sdfg, kwargs, [(args[name], expected[name]) for name in args], exact=True)
    assert code >= 0, f"fused kernel killed by signal {-code}"
    assert code == 0, f"fused Map did not honor the dynamic bound (code {code})"


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


def test_horizontal_fusion_rejects_a_dynamic_map_range_bound_by_tasklets():
    """Two Maps whose ranges are bound by DIFFERENT tasklets must not be treated as agreeing.

    The agreement proof for distinct sources is "nothing in this state writes that data", which is
    vacuously true when the binding does not come from an AccessNode at all -- no AccessNode of that
    data exists, so the scan finds no writer and declares the two bounds equal. Fusing then keeps one
    binding and silently gives the other Map the wrong trip count.
    """
    sdfg = dace.SDFG(unique_name("dmr_bound_by_tasklets"))
    sdfg.add_scalar("bound", dace.int64, transient=True)
    for name in ("A", "B"):
        sdfg.add_array(name, shape=(10, ), dtype=dace.float64, transient=False)
    state = sdfg.add_state(is_start_block=True)

    for name, value in (("A", 3), ("B", 5)):
        _, map_entry, _ = state.add_mapped_tasklet(
            f"comp_{name}",
            map_ranges={"__i": "0:lim"},
            inputs={},
            code=f"__out = {float(value)}",
            outputs={"__out": dace.Memlet(f"{name}[__i]")},
            output_nodes={name: state.add_access(name)},
            external_edges=True,
        )
        bound = state.add_tasklet(f"bound_{name}", set(), {"__out"}, f"__out = {value}")
        map_entry.add_in_connector("lim")
        state.add_edge(bound, "__out", map_entry, "lim", dace.Memlet("bound[0]"))
    sdfg.validate()

    entries = count_nodes(state, nodes.MapEntry, return_nodes=True)
    assert not mfhelper.dynamic_map_ranges_agree(entries[0], entries[1], state), \
        "two different tasklet-produced bounds were declared equal"
    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 0
    assert count_nodes(state, nodes.MapEntry) == 2


def _make_inner_transient_hazard_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, nodes.MapEntry, nodes.MapEntry]:
    """Two Maps whose only link is an ordering edge, over a transient neither one exports.

    ``tmp`` is written by an AccessNode inside the first body and read by an AccessNode inside the
    second, so it never crosses a MapEntry/MapExit connector and the boundary scan behind
    ``analyze_happens_before_fusion()`` books nothing for it. The RAW between the two Maps rests
    entirely on the empty-Memlet ordering edge ``exit(first) -> entry(second)``, and the second Map
    reads ``tmp`` REVERSED, so no per-iteration ordering could stand in for it.

    ``tmp`` needs a State lifetime for this to be one buffer before fusion too: a ``Scope``-lifetime
    transient is allocated per scope, which would make the unfused SDFG read its own private copy.
    """
    sdfg = dace.SDFG(unique_name("horizontal_inner_transient"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_array("a", shape=(8, ), dtype=dace.float64)
    sdfg.add_array("o", shape=(8, ), dtype=dace.float64)
    sdfg.add_array("tmp",
                   shape=(8, ),
                   dtype=dace.float64,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.State)

    first_entry, first_exit = state.add_map("first_map", {"__i": "0:8"})
    write = state.add_tasklet("write", {"__in"}, {"__out"}, "__out = __in + 1.0")
    tmp_write = state.add_access("tmp")
    state.add_memlet_path(state.add_read("a"), first_entry, write, dst_conn="__in", memlet=dace.Memlet("a[__i]"))
    state.add_edge(write, "__out", tmp_write, None, dace.Memlet("tmp[__i]"))
    state.add_nedge(tmp_write, first_exit, dace.Memlet())  # ordering only, keeps `tmp_write` in scope

    second_entry, second_exit = state.add_map("second_map", {"__j": "0:8"})
    tmp_read = state.add_access("tmp")
    read = state.add_tasklet("read", {"__in"}, {"__out"}, "__out = __in")
    state.add_nedge(second_entry, tmp_read, dace.Memlet())  # ordering only, keeps `tmp_read` in scope
    state.add_edge(tmp_read, None, read, "__in", dace.Memlet("tmp[7 - __j]"))
    state.add_memlet_path(read, second_exit, state.add_write("o"), src_conn="__out", memlet=dace.Memlet("o[__j]"))

    state.add_nedge(first_exit, second_entry, dace.Memlet())
    sdfg.validate()
    return sdfg, state, first_entry, second_entry


def test_horizontal_fusion_refuses_an_inner_shared_transient():
    """Regression: the happens-before plan must not drop an ordering edge it cannot replace.

    ``analyze_happens_before_fusion()`` used to reject an inner AccessNode only when its data was
    NON-transient. A transient that both bodies touch internally is just as invisible to the
    boundary scan, so the analysis returned "remove this ordering edge, add nothing back" -- a
    silent race, and here a deterministic wrong answer: every element the fused Map reads before
    it has been written comes back stale. Transience was never the discriminator, since two sibling
    Map scopes sharing even a ``Scope``-lifetime transient get one allocation folded up to the
    state; naming the same data is.
    """
    sdfg, state, first_entry, second_entry = _make_inner_transient_hazard_sdfg()
    param_repl = mfhelper.find_parameter_remapping(first_entry.map, second_entry.map)
    assert param_repl is not None

    assert mfhelper.analyze_happens_before_fusion(
        state=state,
        sdfg=sdfg,
        first_map_entry=first_entry,
        second_map_entry=second_entry,
        param_repl=param_repl,
    ) is None, "a hazard the boundary scan cannot see must not produce a fusion plan"

    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 0
    assert count_nodes(state, nodes.MapEntry) == 2
    # The ordering edge is the whole dependency; losing it is the miscompile.
    assert any(edge.data.is_empty() and edge.dst is second_entry for edge in state.in_edges(second_entry))

    a = np.arange(1.0, 9.0, dtype=np.float64)
    o = np.zeros(8, dtype=np.float64)
    kwargs = {"a": a.copy(), "o": o, "tmp": np.zeros(8, dtype=np.float64)}
    code = exit_code(sdfg, kwargs, [(o, a[::-1] + 1.0)], exact=True)
    assert code >= 0, f"kernel killed by signal {-code}"
    assert code == 0, f"the second Map did not read what the first one wrote (code {code})"


def _make_side_effect_ordering_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, nodes.MapEntry, nodes.MapEntry]:
    """Two Sequential Maps of side-effecting tasklets, ordered only by an empty Memlet.

    Each body prints; nothing else observes the ordering. Sequential is load-bearing -- a parallel
    schedule would make the print order nondeterministic and the oracle below meaningless.
    """
    sdfg = dace.SDFG(unique_name("horizontal_side_effects"))
    state = sdfg.add_state(is_start_block=True)
    for name in ("A", "B"):
        sdfg.add_array(name, shape=(4, ), dtype=dace.float64)
    sdfg.add_array("t", shape=(4, ), dtype=dace.float64, transient=True)

    first_entry, first_exit = state.add_map("first_map", {"__i": "0:4"}, schedule=dace.ScheduleType.Sequential)
    first = state.add_tasklet("io_a", {"__in"}, {"__out"},
                              'printf("A%d ", (int)__in); __out = __in;',
                              language=dace.Language.CPP)
    first.side_effects = True
    state.add_memlet_path(state.add_read("A"), first_entry, first, dst_conn="__in", memlet=dace.Memlet("A[__i]"))
    state.add_memlet_path(first, first_exit, state.add_access("t"), src_conn="__out", memlet=dace.Memlet("t[__i]"))

    second_entry, second_exit = state.add_map("second_map", {"__j": "0:4"}, schedule=dace.ScheduleType.Sequential)
    second = state.add_tasklet("io_b", {}, {"__out"},
                               'printf("B%d ", (int)__j); __out = 1.0;',
                               language=dace.Language.CPP)
    second.side_effects = True
    state.add_nedge(second_entry, second, dace.Memlet())
    state.add_memlet_path(second, second_exit, state.add_write("B"), src_conn="__out", memlet=dace.Memlet("B[__j]"))

    state.add_nedge(first_exit, second_entry, dace.Memlet())
    sdfg.validate()
    return sdfg, state, first_entry, second_entry


def stdout_of_compiled_sdfg(sdfg: dace.SDFG, kwargs: Dict[str, Any]) -> str:
    """Child body: compile and call ``sdfg``, handing back what its tasklets printed.

    A side effect is by definition in no Memlet, so the only oracle for its ordering is the effect
    itself. ``printf`` writes to fd 1 from inside the compiled library, past anything Python could
    intercept, so fd 1 is redirected around the call and the C streams flushed before restoring it.
    """
    csdfg = sdfg.compile()
    with tempfile.TemporaryFile(mode="w+") as sink:
        sys.stdout.flush()
        saved = os.dup(1)
        os.dup2(sink.fileno(), 1)
        try:
            csdfg(**kwargs)
            ctypes.CDLL(None).fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        sink.seek(0)
        return sink.read().strip()


def test_horizontal_fusion_refuses_to_reorder_side_effects():
    """Regression: a side effect is a hazard the Memlet scan cannot see, so it must block the plan.

    ``_scope_boundary_accesses()`` derives hazards from data Memlets alone, so a
    ``side_effects=True`` tasklet contributed nothing and the ordering edge was removed with no
    replacement. That does not merely leave the effects unordered against a write: the whole-Map
    ordering of the two print streams degenerates into a per-iteration interleaving of them.
    """
    sdfg, state, first_entry, second_entry = _make_side_effect_ordering_sdfg()
    param_repl = mfhelper.find_parameter_remapping(first_entry.map, second_entry.map)
    assert param_repl is not None

    assert mfhelper.analyze_happens_before_fusion(
        state=state,
        sdfg=sdfg,
        first_map_entry=first_entry,
        second_map_entry=second_entry,
        param_repl=param_repl,
    ) is None, "a Map holding a side-effecting node must not get a fusion plan"

    assert sdfg.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True) == 0
    assert count_nodes(state, nodes.MapEntry) == 2
    assert any(edge.data.is_empty() and edge.dst is second_entry for edge in state.in_edges(second_entry))

    kwargs = {"A": np.arange(4.0, dtype=np.float64), "B": np.zeros(4, dtype=np.float64)}
    printed = call_in_child(stdout_of_compiled_sdfg, sdfg, kwargs)
    assert printed == "A0 A1 A2 A3 B0 B1 B2 B3", f"the two effect streams were interleaved: {printed!r}"


if __name__ == '__main__':
    test_horizontal_fusion_refuses_an_inner_shared_transient()
    test_horizontal_fusion_refuses_to_reorder_side_effects()
    test_horizontal_fusion_relocates_a_multi_edge_connector_once()
    test_horizontal_fusion_preserves_distinct_ordering_in_edges()
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
    test_horizontal_fusion_rejects_a_dynamic_map_range_bound_by_tasklets()
