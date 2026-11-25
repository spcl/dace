# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A column kernel (a level loop around maps over the columns) runs one block per kernel iteration and one lane per
column, with no barrier between its column maps. The structural checks need no GPU; the numeric ones run on the
device, at a column count above the lane cap so that lanes take more than one column."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.gpu_specialization.column_lane_kernels import (
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
def column_map_leaks_a_scalar(t: dace.float64[KLON, KLEV, NB], out: dace.float64[KLEV, NB]):
    for jb in dace.map[0:NB]:
        for jk in range(KLEV):
            s = 0.0
            for jl in dace.map[0:KLON]:
                s = t[jl, jk, jb]
            out[jk, jb] = s


def offloaded(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    return sdfg


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


def test_a_level_loop_over_column_maps_runs_one_lane_per_column_without_barriers():
    sdfg = offloaded(column_physics)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (lanes,) = lane_maps(sdfg)
    assert lanes.specialization_hint == COLUMN_LANES_HINT
    column_maps = [
        node
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and str(node.map.range[-1][0]).startswith("__tid")
    ]
    assert len(column_maps) == 3  # the zero fill and the two column maps
    assert barriers(sdfg) == 0
    # The column buffer is one per block, outside the kernel
    acc = [nested.arrays["acc"] for nested in sdfg.all_sdfgs_recursive() if "acc" in nested.arrays]
    assert all(desc.storage == dtypes.StorageType.GPU_Global for desc in acc)
    assert any(desc.transient and str(desc.shape[0]) == "NB" for desc in acc)


def test_a_read_of_another_lanes_column_is_fenced_by_barriers():
    sdfg = offloaded(last_column_feeds_all)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert barriers(sdfg) == 2


def test_a_scalar_a_column_map_leaves_for_later_code_keeps_the_kernel_untiled():
    sdfg = offloaded(column_map_leaks_a_scalar)
    assert ColumnLaneKernels().apply_pass(sdfg, {}) is None
    assert not lane_maps(sdfg)


def reference(program, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    acc = np.zeros((t.shape[0], t.shape[2]))
    out = np.zeros_like(t)
    for jk in range(1, t.shape[1]):
        if program is column_physics:
            acc += t[:, jk, :] * q[:, jk - 1, :]
            out[:, jk, :] = acc * 2.0
        else:
            acc += t[:, jk, :]
            out[:, jk, :] = acc - acc[-1]
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("program", [column_physics, last_column_feeds_all])
@pytest.mark.parametrize("klon", [37, 300])
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
    test_a_level_loop_over_column_maps_runs_one_lane_per_column_without_barriers()
    test_a_read_of_another_lanes_column_is_fenced_by_barriers()
    test_a_scalar_a_column_map_leaves_for_later_code_keeps_the_kernel_untiled()
    for program in (column_physics, last_column_feeds_all):
        for klon in (37, 300):
            test_column_lanes_match_numpy_on_the_gpu(program, klon)
