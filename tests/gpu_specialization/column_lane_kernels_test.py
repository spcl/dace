# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A column kernel (a level loop around maps over the columns) runs one block per kernel iteration, a lane per column,
block-stride over columns in chunks of the lane count, with no barrier between its column maps. The structural checks
need no GPU; the numeric ones run on the device, at column extents below, at, and above a block's lanes (1100 is above
the 1024 threads a block may hold, 1500 no multiple of the 128 lanes)."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.gpu_specialization.column_lane_kernels import (
    BLOCK_LANES,
    COLUMN_LANES_HINT,
    ColumnLaneKernels,
)

KLON, KLEV, NB = (dace.symbol(s) for s in ("KLON", "KLEV", "NB"))


@dace.program
def column_physics(t: dace.float64[KLON, KLEV, NB], q: dace.float64[KLON, KLEV, NB], out: dace.float64[KLON, KLEV, NB]):
    for jb in dace.map[0:NB]:
        acc = np.zeros((KLON,), dtype=np.float64)
        for jk in range(1, KLEV):
            for jl in dace.map[0:KLON]:
                acc[jl] = acc[jl] + t[jl, jk, jb] * q[jl, jk - 1, jb]
            for jl in dace.map[0:KLON]:
                out[jl, jk, jb] = acc[jl] * 2.0


@dace.program
def sixty_four_columns(t: dace.float64[64, KLEV, NB], q: dace.float64[64, KLEV, NB], out: dace.float64[64, KLEV, NB]):
    for jb in dace.map[0:NB]:
        acc = np.zeros((64,), dtype=np.float64)
        for jk in range(1, KLEV):
            for jl in dace.map[0:64]:
                acc[jl] = acc[jl] + t[jl, jk, jb] * q[jl, jk - 1, jb]
            for jl in dace.map[0:64]:
                out[jl, jk, jb] = acc[jl] * 2.0


@dace.program
def last_column_feeds_all(t: dace.float64[KLON, KLEV, NB], out: dace.float64[KLON, KLEV, NB]):
    for jb in dace.map[0:NB]:
        acc = np.zeros((KLON,), dtype=np.float64)
        for jk in range(1, KLEV):
            for jl in dace.map[0:KLON]:
                acc[jl] = acc[jl] + t[jl, jk, jb]
            last = 0.0
            if t[0, jk, jb] >= 0.0:  # a state of its own, between the two column maps
                last = acc[KLON - 1]
            for jl in dace.map[0:KLON]:
                out[jl, jk, jb] = acc[jl] - last


@dace.program
def species_reset_per_level(t: dace.float64[KLON, KLEV, NB], out: dace.float64[KLON, KLEV, NB]):
    for jb in dace.map[0:NB]:
        acc = np.zeros((KLON, 3), dtype=np.float64)
        for jk in range(1, KLEV):
            for jl in dace.map[0:KLON]:
                for s in range(3):
                    acc[jl, s] = acc[jl, s] + t[jl, jk, jb] * s
            last = 0.0
            if t[0, jk, jb] >= 0.0:
                last = acc[KLON - 1, 2]
            for jl in dace.map[0:KLON]:
                out[jl, jk, jb] = acc[jl, 1] - last


def zero_species_per_level(sdfg: dace.SDFG) -> None:
    """Zero every column of ``acc`` at the start of each level: a map over the species around a fill of one species'
    columns, the form canonicalization gives CloudSC's ``zpfplsx``. The fill's memlet ends at the map exit, not at an
    access node."""
    (loop,) = [
        region
        for region in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(region, LoopRegion) and region.loop_variable == "jk"
    ]
    state = loop.add_state_before(loop.start_block, "zero_species")
    entry, exit_node = state.add_map("species", {"s": "0:3"})
    zero = FillLibraryNode("zero_acc", value=0)
    state.add_nedge(entry, zero, Memlet())
    state.add_memlet_path(
        zero,
        exit_node,
        state.add_write("acc"),
        src_conn=FillLibraryNode.OUTPUT_CONNECTOR_NAME,
        memlet=Memlet("acc[0:KLON, s]"),
    )


@dace.program
def column_map_leaks_a_scalar(t: dace.float64[KLON, KLEV, NB], out: dace.float64[KLEV, NB]):
    for jb in dace.map[0:NB]:
        for jk in range(KLEV):
            s = 0.0
            for jl in dace.map[0:KLON]:
                s = t[jl, jk, jb]
            out[jk, jb] = s


@dace.program
def species_columns(t: dace.float64[KLON, KLEV, NB], out: dace.float64[KLON, KLEV, NB]):
    for jb in dace.map[0:NB]:
        acc = np.zeros((KLON, 5), dtype=np.float64)
        for jk in range(KLEV):
            for jl in dace.map[0:KLON]:
                for s in range(5):
                    if t[jl, jk, jb] > 0.2 * s:
                        acc[jl, s] = acc[jl, s] + t[jl, jk, jb]
            for jl in dace.map[0:KLON]:
                out[jl, jk, jb] = acc[jl, 0] - acc[jl, 4]


def offloaded(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    if program is species_reset_per_level:
        zero_species_per_level(sdfg)
    sdfg.apply_gpu_transformations()
    return sdfg


def column_maps(sdfg: dace.SDFG, column: str) -> list[nodes.MapEntry]:
    """The maps over the columns, which start at ``column``."""
    return [
        node
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and str(node.map.range[-1][0]).startswith(column)
    ]


def chunk_loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [
        region
        for region in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(region, LoopRegion) and region.loop_variable == "__col"
    ]


def lane_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [
        node
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]


def barriers(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and "__syncthreads" in node.code.as_string
    )


def test_a_level_loop_over_column_maps_runs_its_body_per_chunk_of_columns_without_barriers():
    sdfg = offloaded(column_physics)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (lanes,) = lane_maps(sdfg)
    assert lanes.map.range.size() == [BLOCK_LANES]
    assert lanes.specialization_hint == COLUMN_LANES_HINT.format(lanes=BLOCK_LANES)
    (chunks,) = chunk_loops(sdfg)
    assert chunks.update_statement.as_string == f"__col = (__col + {BLOCK_LANES})"
    assert len(column_maps(sdfg, "__col")) == 3  # the zero fill and the two column maps
    assert barriers(sdfg) == 0
    # Each lane's copy of the column buffer is its current column alone
    acc = [nested.arrays["acc"] for nested in sdfg.all_sdfgs_recursive() if "acc" in nested.arrays]
    assert acc and all(tuple(desc.shape) == (1,) and desc.storage != dtypes.StorageType.GPU_Global for desc in acc)


def test_a_constant_extent_up_to_the_lane_limit_is_one_chunk_of_a_lane_per_column():
    sdfg = offloaded(sixty_four_columns)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (lanes,) = lane_maps(sdfg)
    assert lanes.map.range.size() == [64]
    assert not chunk_loops(sdfg)
    assert len(column_maps(sdfg, "__tid")) == 3


def test_a_read_of_another_lanes_column_is_fenced_by_barriers_every_lane_reaches():
    sdfg = offloaded(last_column_feeds_all)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert barriers(sdfg) == 2
    # No chunk loop around the barriers: each column map strides over the lane's columns itself
    assert not chunk_loops(sdfg)
    strided = column_maps(sdfg, "__tid")
    assert len(strided) == 3 and all(entry.map.range[-1][2] == BLOCK_LANES for entry in strided)
    # The lanes share the buffer whose columns they read across: one per block, outside the kernel
    acc = [nested.arrays["acc"] for nested in sdfg.all_sdfgs_recursive() if "acc" in nested.arrays]
    assert all(desc.storage == dtypes.StorageType.GPU_Global for desc in acc)


def test_a_fill_inside_a_map_writes_only_its_lanes_columns():
    sdfg = offloaded(species_reset_per_level)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    fills = [
        edge.data.subset
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, FillLibraryNode)
        for edge in state.out_edges(node)
    ]
    # It used to fill every column on every lane, racing the lanes still reading the last level's columns
    (fill,) = fills
    assert any(str(begin) == "__tid" and step == BLOCK_LANES for begin, end, step in fill)


def test_a_scalar_a_column_map_leaves_for_later_code_keeps_the_kernel_untiled():
    sdfg = offloaded(column_map_leaks_a_scalar)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) is None
    assert not lane_maps(sdfg)


def test_a_column_buffer_with_a_species_axis_keeps_only_the_species_per_lane():
    sdfg = offloaded(species_columns)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert barriers(sdfg) == 0
    acc = [nested.arrays["acc"] for nested in sdfg.all_sdfgs_recursive() if "acc" in nested.arrays]
    assert acc and all(tuple(desc.shape) == (5,) for desc in acc)


def reference(program, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    if program is species_columns:
        acc = np.zeros((t.shape[0], 5, t.shape[2]))
        for jk in range(t.shape[1]):
            for s in range(5):
                acc[:, s, :] += np.where(t[:, jk, :] > 0.2 * s, t[:, jk, :], 0.0)
            out[:, jk, :] = acc[:, 0, :] - acc[:, 4, :]
        return out
    if program is species_reset_per_level:
        out[:, 1:, :] = t[:, 1:, :] - 2.0 * t[-1:, 1:, :]
        return out
    acc = np.zeros((t.shape[0], t.shape[2]))
    for jk in range(1, t.shape[1]):
        if program is column_physics:
            acc += t[:, jk, :] * q[:, jk - 1, :]
            out[:, jk, :] = acc * 2.0
        else:
            acc += t[:, jk, :]
            out[:, jk, :] = acc - acc[-1]
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("program", [column_physics, last_column_feeds_all, species_columns, species_reset_per_level])
@pytest.mark.parametrize("klon", [37, 128, 1100, 1500])
def test_column_lanes_match_numpy_on_the_gpu(program, klon: int):
    sdfg = offloaded(program)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    rng = np.random.default_rng(0)
    t, q = rng.random((klon, 7, 5)), rng.random((klon, 7, 5))
    out = np.zeros_like(t)
    args = {"t": t, "out": out, "KLON": klon, "KLEV": 7, "NB": 5}
    if program is column_physics:
        args["q"] = q
    sdfg(**args)
    assert np.allclose(out, reference(program, t, q), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_a_level_loop_over_column_maps_runs_its_body_per_chunk_of_columns_without_barriers()
    test_a_constant_extent_up_to_the_lane_limit_is_one_chunk_of_a_lane_per_column()
    test_a_read_of_another_lanes_column_is_fenced_by_barriers_every_lane_reaches()
    test_a_fill_inside_a_map_writes_only_its_lanes_columns()
    test_a_scalar_a_column_map_leaves_for_later_code_keeps_the_kernel_untiled()
    test_a_column_buffer_with_a_species_axis_keeps_only_the_species_per_lane()
    for program in (column_physics, last_column_feeds_all, species_columns, species_reset_per_level):
        for klon in (37, 128, 1100, 1500):
            test_column_lanes_match_numpy_on_the_gpu(program, klon)
