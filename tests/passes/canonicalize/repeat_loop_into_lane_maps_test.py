# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop moves into the maps of its body when a copy between containers outside the maps moves one lane's slice per
lane, as a map over the lanes would. On CPU a loop that only repeats its body (time steps) moves into the maps its body
shares an axis with: the column map runs outermost with the step and level loops inside. On GPU a reseed of several
levels per column moves too: one kernel runs the step and level loops in each column's thread."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.passes.canonicalize import canonicalize, finalize

NL = dace.symbol("NL")
NP = dace.symbol("NP")
nst = dace.symbol("nst")
NC = dace.symbol("NC")
NLEV = dace.symbol("NLEV")


@dace.program
def scan_levels_k1(a: dace.float64[NL, NP], c: dace.float64[NL, NP], x: dace.float64[NL, NP]):
    # The time steps, each a sweep over the levels and a reseed
    for step in range(nst):
        # The levels above the reseeded ones, each from one a fixed distance below
        for k in range(1, NL):
            a[k, :] = c[k, :] * a[k - 1, :] + x[k, :]
        if NL >= 2:
            a[0:1, :] = a[NL - 1 : NL, :]


@dace.program
def scan_levels_k4(a: dace.float64[NL, NP], c: dace.float64[NL, NP], x: dace.float64[NL, NP]):
    # The time steps, each a sweep over the levels and a reseed
    for step in range(nst):
        # The levels above the reseeded ones, each from one a fixed distance below
        for k in range(4, NL):
            a[k, :] = c[k, :] * a[k - 4, :] + x[k, :]
        if NL >= 8:
            a[0:4, :] = a[NL - 4 : NL, :]


@dace.program
def scan_levels_k32(a: dace.float64[NL, NP], c: dace.float64[NL, NP], x: dace.float64[NL, NP]):
    # The time steps, each a sweep over the levels and a reseed
    for step in range(nst):
        # The levels above the reseeded ones, each from one a fixed distance below
        for k in range(32, NL):
            a[k, :] = c[k, :] * a[k - 32, :] + x[k, :]
        if NL >= 64:
            a[0:32, :] = a[NL - 32 : NL, :]


@dace.program
def shifted_reseed_k4(a: dace.float64[NL, NP], c: dace.float64[NL, NP], x: dace.float64[NL, NP]):
    # The time steps, each a sweep over the levels and a reseed from the neighbouring column
    for step in range(nst):
        # The levels above the reseeded ones, each from one a fixed distance below
        for k in range(4, NL):
            a[k, :] = c[k, :] * a[k - 4, :] + x[k, :]
        a[0:4, 1:NP] = a[NL - 4 : NL, 0 : NP - 1]


@dace.program
def shifted_reseed(a: dace.float64[NL, NP], c: dace.float64[NL, NP], x: dace.float64[NL, NP]):
    # The time steps, each a sweep over the levels and a reseed
    for step in range(nst):
        # The levels above the reseeded ones, each from one a fixed distance below
        for k in range(1, NL):
            a[k, :] = c[k, :] * a[k - 1, :] + x[k, :]
        a[0, 1:NP] = a[NL - 1, 0 : NP - 1]


@dace.program
def level_scratch(w: dace.float64[NC, NLEV], coef: dace.float64[NC], z: dace.float64[NC, NLEV]):
    acc = np.ndarray((NC,), np.float64)
    # The levels, each accumulated in the scratch and copied out
    for k in range(NLEV):
        acc[:] = 0.0
        # The columns, independent of each other
        for c in dace.map[0:NC]:
            acc[c] = acc[c] + coef[c] * w[c, k]
        z[:, k] = acc[:]


@dace.program
def level_scratch_shifted(w: dace.float64[NC, NLEV], coef: dace.float64[NC], z: dace.float64[NC, NLEV]):
    acc = np.ndarray((NC,), np.float64)
    # The levels, each accumulated in the scratch and copied out
    for k in range(NLEV):
        acc[:] = 0.0
        # The columns, independent of each other
        for c in dace.map[0:NC]:
            acc[c] = acc[c] + coef[c] * w[c, k]
        z[1:NC, k] = acc[0 : NC - 1]


def level_loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(block for block in sdfg.nodes() if isinstance(block, LoopRegion))


def scan_levels_reference(a: np.ndarray, c: np.ndarray, x: np.ndarray, steps: int, depth: int) -> None:
    # The time steps, each a sweep over the levels and a reseed of the first ``depth`` levels
    for step in range(steps):
        # The levels above the reseeded ones, each from the one ``depth`` below
        for k in range(depth, a.shape[0]):
            a[k, :] = c[k, :] * a[k - depth, :] + x[k, :]
        if a.shape[0] >= 2 * depth:
            a[0:depth, :] = a[a.shape[0] - depth :, :]


def shifted_reseed_reference(a: np.ndarray, c: np.ndarray, x: np.ndarray, steps: int) -> None:
    # The time steps, each a sweep over the levels and a reseed shifted by one column
    for step in range(steps):
        # The levels above the first, each from the one below
        for k in range(1, a.shape[0]):
            a[k, :] = c[k, :] * a[k - 1, :] + x[k, :]
        a[0, 1:] = a[-1, :-1].copy()


def canonical(program, target: str = "cpu") -> dace.SDFG:
    sdfg = program.to_sdfg()
    canonicalize(sdfg, validate=True, target=target)
    return sdfg


def inputs(seed: int, shape: tuple[int, int] = (40, 16)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random(shape), 0.2 + 0.7 * rng.random(shape), rng.random(shape)


def top_level_loops(sdfg: dace.SDFG) -> int:
    return sum(isinstance(block, LoopRegion) for block in sdfg.nodes())


def check_scan_levels(program, depth: int) -> None:
    sdfg = canonical(program)
    # The step loop runs inside the column map: no loop is left around it
    assert top_level_loops(sdfg) == 0
    a, c, x = inputs(depth)
    ref = a.copy()
    scan_levels_reference(ref, c, x, 3, depth)
    sdfg(a=a, c=c, x=x, NL=40, NP=16, nst=3)
    assert np.allclose(a, ref)


# Reproducer: canon left `for step > for k > map columns` (a fork/join per level) for LLR scan_levels_k1
def test_step_loop_with_one_level_reseed_moves_into_column_map():
    check_scan_levels(scan_levels_k1, 1)


# Reproducer: the lane copy a[0:4, lane] = a[NL-4:NL, lane] ran backwards (LLR scan_levels_k4)
def test_step_loop_with_four_level_reseed_moves_into_column_map():
    check_scan_levels(scan_levels_k4, 4)


# Reproducer: a reseed that moves values across columns keeps the step loop outside the column map
def test_reseed_across_columns_keeps_step_loop_outside():
    sdfg = canonical(shifted_reseed)
    assert top_level_loops(sdfg) == 1
    a, c, x = inputs(7)
    ref = a.copy()
    shifted_reseed_reference(ref, c, x, 3)
    sdfg(a=a, c=c, x=x, NL=40, NP=16, nst=3)
    assert np.allclose(a, ref)


# Reproducer: the per-level copy z[:, k] = scratch beside the level's maps counted as lane-independent code (ICON
# velocity_tendencies kept its level loops on the host, a memset, kernel and copy launch per level)
def test_level_loop_with_lane_copy_moves_into_column_map():
    sdfg = level_scratch.to_sdfg(simplify=True)
    loop = level_loop(sdfg)
    assert MoveLoopIntoMap.can_be_applied_to(sdfg, options={"cfg_body": True}, loop=loop)
    MoveLoopIntoMap.apply_to(sdfg, options={"cfg_body": True}, loop=loop)
    assert top_level_loops(sdfg) == 0
    rng = np.random.default_rng(3)
    w, coef, z = rng.random((24, 5)), rng.random(24), np.zeros((24, 5))
    sdfg(w=w, coef=coef, z=z, NC=24, NLEV=5)
    assert np.allclose(z, coef[:, None] * w)


# Reproducer: a per-level copy shifted by one column moves values across lanes, so the level loop stays
def test_level_loop_with_shifted_copy_stays():
    sdfg = level_scratch_shifted.to_sdfg(simplify=True)
    assert not MoveLoopIntoMap.can_be_applied_to(sdfg, options={"cfg_body": True}, loop=level_loop(sdfg))


GPU_SCANS = {1: scan_levels_k1, 4: scan_levels_k4, 32: scan_levels_k32}


def device_kernels(sdfg: dace.SDFG) -> list[str]:
    # Over every node of every nested SDFG: the device maps are the kernels
    return [
        node.map.label
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device
    ]


def run_on_gpu(sdfg: dace.SDFG, a: np.ndarray, c: np.ndarray, x: np.ndarray, steps: int) -> np.ndarray:
    import cupy  # Only present on GPU runners.

    # Over the three arrays: each goes to the device
    args = {name: cupy.asarray(value) for name, value in dict(a=a, c=c, x=x).items()}
    sdfg(**args, NL=a.shape[0], NP=a.shape[1], nst=steps)
    return cupy.asnumpy(args["a"])


# Reproducer: on GPU a reseed of several levels per column (LLR scan_levels_k4, k32) left the step loop on the host, a
# launch and a copy per step; the plain lane analysis took only one element per lane
@pytest.mark.parametrize("depth", [1, 4, 32])
def test_gpu_step_loop_with_level_reseed_moves_into_column_map(depth: int):
    sdfg = canonical(GPU_SCANS[depth], "gpu")
    assert top_level_loops(sdfg) == 0


# Reproducer: the GPU form of LLR scan_levels: one kernel with the step and level loops inside each column's thread;
# 300 columns span more than one thread block
@pytest.mark.gpu
@pytest.mark.parametrize("depth", [1, 4, 32])
def test_gpu_scan_levels_runs_as_one_kernel_and_matches_numpy(depth: int):
    sdfg = canonical(GPU_SCANS[depth], "gpu")
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, "gpu")
    # One kernel, launched once: no host loop is left around it
    assert len(device_kernels(sdfg)) == 1 and top_level_loops(sdfg) == 0, device_kernels(sdfg)
    a, c, x = inputs(depth, (2 * depth + 16, 300))
    ref = a.copy()
    scan_levels_reference(ref, c, x, 3, depth)
    assert np.allclose(run_on_gpu(sdfg, a, c, x, 3), ref)


# Reproducer: a reseed of several levels that reads the neighbouring column moves values across lanes, so the GPU step
# loop stays on the host
@pytest.mark.gpu
def test_gpu_reseed_from_neighbouring_column_keeps_step_loop_on_host():
    sdfg = canonical(shifted_reseed_k4, "gpu")
    assert top_level_loops(sdfg) == 1
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, "gpu")
    a, c, x = inputs(11, (40, 300))
    ref = a.copy()
    # The time steps, each a sweep over the levels and a reseed shifted by one column
    for step in range(3):
        # The levels above the reseeded ones, each from the one four below
        for k in range(4, ref.shape[0]):
            ref[k, :] = c[k, :] * ref[k - 4, :] + x[k, :]
        ref[0:4, 1:] = ref[-4:, :-1].copy()
    assert np.allclose(run_on_gpu(sdfg, a, c, x, 3), ref)


if __name__ == "__main__":
    test_step_loop_with_one_level_reseed_moves_into_column_map()
    test_step_loop_with_four_level_reseed_moves_into_column_map()
    test_reseed_across_columns_keeps_step_loop_outside()
    test_level_loop_with_lane_copy_moves_into_column_map()
    test_level_loop_with_shifted_copy_stays()
    test_gpu_step_loop_with_level_reseed_moves_into_column_map(1)
    test_gpu_step_loop_with_level_reseed_moves_into_column_map(4)
    test_gpu_step_loop_with_level_reseed_moves_into_column_map(32)
    test_gpu_scan_levels_runs_as_one_kernel_and_matches_numpy(1)
    test_gpu_scan_levels_runs_as_one_kernel_and_matches_numpy(4)
    test_gpu_scan_levels_runs_as_one_kernel_and_matches_numpy(32)
    test_gpu_reseed_from_neighbouring_column_keeps_step_loop_on_host()
