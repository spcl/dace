# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A parallel CPU map never shares out a SMALL axis.

``nblks`` is declared ``SMALL``: a handful of blocks, much smaller than ``nproma``. The CPU code generator shares
out the first parameter of a map, so a map over ``(b, i)`` gave each of ``nblks`` threads a whole block and left the
rest idle. The SMALL axes now run in an inner sequential map under a parallel map over the others; a map over SMALL
axes only, around a map over large ones, is collapsed with it first, so the large axes are the ones shared out.
"""

import numpy as np

import dace
from dace import dtypes
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg import infer_types, nodes
from dace.symbolic import Scale
from dace.transformation.passes.cpu_specialization.pipeline import cpu_specialize
from dace.transformation.passes.cpu_specialization.sequentialize_small_axes import SequentializeSmallAxes

PAR = dtypes.ScheduleType.CPU_Multicore
SEQ = dtypes.ScheduleType.Sequential
NBLKS, NPROMA = 3, 50


def blocked_sdfg(name: str, small: bool = True) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("nblks", dace.int64, scale=Scale.SMALL if small else Scale.DEFAULT)
    sdfg.add_symbol("nproma", dace.int64)
    sdfg.add_array("A", ["nblks", "nproma"], dace.float64)
    sdfg.add_array("B", ["nblks", "nproma"], dace.float64)
    return sdfg


def one_map(name: str, small: bool = True) -> dace.SDFG:
    """``B = 2 * A + 1`` in one parallel map over ``(b, i)``."""
    sdfg = blocked_sdfg(name, small)
    state = sdfg.add_state("compute")
    state.add_mapped_tasklet(
        "scale",
        {"b": "0:nblks", "i": "0:nproma"},
        {"x": dace.Memlet("A[b, i]")},
        "y = 2.0 * x + 1.0",
        {"y": dace.Memlet("B[b, i]")},
        schedule=PAR,
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def nested_maps(name: str, inner_range: str = "0:nproma") -> dace.SDFG:
    """``B = 2 * A + 1`` in a parallel map over ``b`` around a sequential map over ``i``."""
    sdfg = blocked_sdfg(name)
    state = sdfg.add_state("compute")
    outer_entry, outer_exit = state.add_map("blocks", {"b": "0:nblks"}, schedule=PAR)
    inner_entry, inner_exit = state.add_map("rows", {"i": inner_range}, schedule=SEQ)
    tasklet = state.add_tasklet("scale", {"x"}, {"y"}, "y = 2.0 * x + 1.0")
    state.add_memlet_path(
        state.add_read("A"), outer_entry, inner_entry, tasklet, dst_conn="x", memlet=dace.Memlet("A[b, i]")
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[b, i]")
    )
    sdfg.validate()
    return sdfg


def map_nest(sdfg: dace.SDFG) -> list[tuple[list[str], dtypes.ScheduleType]]:
    """Every map's parameters and schedule, outermost first."""
    state = next(s for s in sdfg.states() if s.label == "compute")
    scopes = state.scope_dict()

    def depth(node: nodes.Node) -> int:
        level = 0
        while scopes[node] is not None:
            node, level = scopes[node], level + 1
        return level

    entries = sorted((n for n in state.nodes() if isinstance(n, nodes.MapEntry)), key=depth)
    return [(e.map.params, e.map.schedule) for e in entries]


def check_values(sdfg: dace.SDFG) -> None:
    A = np.random.default_rng(11).random((NBLKS, NPROMA))
    B = np.zeros_like(A)
    sdfg(A=A, B=B, nblks=NBLKS, nproma=NPROMA)
    np.testing.assert_allclose(B, 2.0 * A + 1.0)


def test_a_block_axis_moves_into_an_inner_sequential_map():
    sdfg = one_map("cpu_block_first")
    assert SequentializeSmallAxes().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert map_nest(sdfg) == [(["i"], PAR), (["b"], SEQ)]
    check_values(sdfg)


def test_a_block_map_around_a_row_map_shares_out_the_rows():
    sdfg = nested_maps("cpu_block_around_rows")
    assert SequentializeSmallAxes().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert map_nest(sdfg) == [(["i"], PAR), (["b"], SEQ)]
    check_values(sdfg)


def test_a_row_map_whose_range_reads_the_block_is_left_alone():
    """The rows depend on the block, so the two maps do not collapse and the block map stays the parallel one."""
    sdfg = nested_maps("cpu_block_dependent_rows", inner_range="0:nproma - b")
    assert SequentializeSmallAxes().apply_pass(sdfg, {}) is None
    assert map_nest(sdfg) == [(["b"], PAR), (["i"], SEQ)]


def test_a_block_map_without_a_large_axis_is_left_alone():
    sdfg = blocked_sdfg("cpu_blocks_only")
    state = sdfg.add_state("compute")
    state.add_mapped_tasklet(
        "first",
        {"b": "0:nblks"},
        {"x": dace.Memlet("A[b, 0]")},
        "y = x",
        {"y": dace.Memlet("B[b, 0]")},
        schedule=PAR,
        external_edges=True,
    )
    assert SequentializeSmallAxes().apply_pass(sdfg, {}) is None
    assert map_nest(sdfg) == [(["b"], PAR)]


def test_without_small_symbols_nothing_changes():
    sdfg = one_map("cpu_no_small", small=False)
    assert SequentializeSmallAxes().apply_pass(sdfg, {}) is None
    assert map_nest(sdfg) == [(["b", "i"], PAR)]


def test_cpu_specialize_never_shares_out_a_block_axis():
    for sdfg in (one_map("cpu_spec_one_map"), nested_maps("cpu_spec_nested")):
        for entry in (e for e, _ in sdfg.all_nodes_recursive() if isinstance(e, nodes.MapEntry)):
            entry.map.schedule = dtypes.ScheduleType.Default
        cpu_specialize(sdfg)
        for params, schedule in map_nest(sdfg):
            assert schedule != PAR or "b" not in params, map_nest(sdfg)
        check_values(sdfg)


def expanded_fill_maps(small: bool) -> list[tuple[list[str], dtypes.ScheduleType]]:
    """Expand a whole-array fill of ``B`` into its element map and return that map nest, outermost first."""
    sdfg = blocked_sdfg("cpu_expanded_fill_small" if small else "cpu_expanded_fill", small)
    state = sdfg.add_state("compute")
    fill = FillLibraryNode("fill", value=0.0)
    fill.implementation = "pure"
    state.add_node(fill)
    state.add_edge(
        fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write("B"), None, dace.Memlet("B[0:nblks, 0:nproma]")
    )
    sdfg.validate()
    # As finalization leaves it for code generation: the top-level node scheduled on the device.
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    fill.expand(state)
    nested = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    inner = next(iter(nested.sdfg.states()))
    scopes = inner.scope_dict()
    entries = [n for n in inner.nodes() if isinstance(n, nodes.MapEntry)]
    entries.sort(key=lambda e: scopes[e] is not None)
    B = np.ones((NBLKS, NPROMA))
    sdfg(A=np.zeros((NBLKS, NPROMA)), B=B, nblks=NBLKS, nproma=NPROMA)
    np.testing.assert_allclose(B, 0.0)
    return [(e.map.params, e.map.schedule) for e in entries]


def test_a_library_expansion_never_shares_out_a_block_axis():
    """A library node expands at code generation, after ``SequentializeSmallAxes``: a whole-array fill of a
    ``(nblks, nproma)`` temporary was one map shared out over ``nblks`` -- velocity_tendencies at one block
    filled five ``(nblks, nlev, nproma)`` temporaries per call on one thread each."""
    nest = expanded_fill_maps(small=True)
    assert nest[0][1] == PAR and nest[0][0] == ["__i1"], nest
    assert nest[1] == (["__i0"], SEQ), nest
    assert expanded_fill_maps(small=False) == [(["__i0", "__i1"], PAR)]


if __name__ == "__main__":
    test_a_block_axis_moves_into_an_inner_sequential_map()
    test_a_block_map_around_a_row_map_shares_out_the_rows()
    test_a_row_map_whose_range_reads_the_block_is_left_alone()
    test_a_block_map_without_a_large_axis_is_left_alone()
    test_without_small_symbols_nothing_changes()
    test_cpu_specialize_never_shares_out_a_block_axis()
    test_a_library_expansion_never_shares_out_a_block_axis()
