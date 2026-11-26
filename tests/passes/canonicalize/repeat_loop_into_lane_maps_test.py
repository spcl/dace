# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop moves into the maps of its body when a copy between containers outside the maps moves one lane's slice per
lane, as a map over the lanes would. On CPU a loop that only repeats its body (time steps) moves into the maps its body
shares an axis with: the column map runs outermost with the step and level loops inside."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.passes.canonicalize import canonicalize

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


def canonical(program) -> dace.SDFG:
    sdfg = program.to_sdfg()
    canonicalize(sdfg, validate=True, target="cpu")
    return sdfg


def inputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((40, 16)), 0.2 + 0.7 * rng.random((40, 16)), rng.random((40, 16))


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


if __name__ == "__main__":
    test_step_loop_with_one_level_reseed_moves_into_column_map()
    test_step_loop_with_four_level_reseed_moves_into_column_map()
    test_reseed_across_columns_keeps_step_loop_outside()
    test_level_loop_with_lane_copy_moves_into_column_map()
    test_level_loop_with_shifted_copy_stays()
