# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Comprehensive numerical BLOCK-layout sweep over 3D and 4D elementwise kernels.

The BLOCK family (``SplitDimensions``) splits one or more dimensions of an array ``A`` into an outer
tile-index and an inner tile: an access ``A[.., i, ..]`` where ``i`` is blocked by ``b`` becomes
``A[.., int_floor(i, b), .., Mod(i, b)]``, with the inner tile axes APPENDED LAST in mask (left to
right) order. Because the descriptor shape CHANGES, the caller must physically PACK the logical input
into that ``[.. positions .., .. tiles ..]`` layout (a plain C-reshape only works when the blocked
dim is already final; otherwise the tile axis is moved to the end).

This sweep drives that end to end -- for every case it applies ``SplitDimensions`` then
``normalize_schedule_for_layout`` (re-tiling the map to the block width), compiles, runs, and asserts
the result is bit-exact (``numpy.allclose``) against a flat numpy oracle. Coverage: 3D and 4D
elementwise ``C = A*2 + 1``; each single dimension blocked by {2, 4}; and every pair of dimensions
blocked at once (multi-mask) with mixed factors.
"""
import itertools

import numpy
import pytest

import dace

from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout
from dace.transformation.layout.split_dimensions import SplitDimensions

N = dace.symbol("N")


@dace.program
def elt3d(A: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j, k] = A[i, j, k] * 2.0 + 1.0


@dace.program
def elt4d(A: dace.float64[N, N, N, N], C: dace.float64[N, N, N, N]):
    for i, j, k, m in dace.map[0:N, 0:N, 0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j, k, m] = A[i, j, k, m] * 2.0 + 1.0


PROGRAMS = {3: elt3d, 4: elt4d}


def masks_and_factors(ndim, dims, factors):
    """Build the length-``ndim`` (mask, factor) arrays ``SplitDimensions`` expects from the blocked
    dimension indices ``dims`` and their per-dim ``factors`` (unblocked dims carry factor 1)."""
    masks = [d in dims for d in range(ndim)]
    facs = [1] * ndim
    for d, b in zip(dims, factors):
        facs[d] = b
    return masks, facs


def block_pack(A, masks, factors):
    """Lay logical ``A`` into the physical descriptor ``SplitDimensions`` produces.

    Split each blocked dim ``d`` into ``(outer = extent/b, tile = b)`` in place, then move EVERY tile
    axis to the end in mask (left to right) order, leaving the position axes (outer for a blocked dim,
    the single axis for an unblocked dim) in original dimension order up front. A fresh contiguous
    copy is returned -- DaCe rejects numpy views as arguments.
    """
    new_shape = []
    pos_axes = []
    tile_axes = []
    axis = 0
    for d, extent in enumerate(A.shape):
        if masks[d]:
            b = factors[d]
            new_shape.append(extent // b)
            new_shape.append(b)
            pos_axes.append(axis)
            tile_axes.append(axis + 1)
            axis += 2
        else:
            new_shape.append(extent)
            pos_axes.append(axis)
            axis += 1
    perm = pos_axes + tile_axes
    return A.reshape(new_shape).transpose(perm).copy()


def build_cases():
    """Enumerate ``(ndim, dims, factors)`` cases: every single dimension blocked by {2, 4}, then
    every pair of dimensions blocked at once with mixed factors, over 3D and 4D."""
    single_factors = (2, 4)
    pair_factors_3d = ((2, 2), (2, 4), (4, 2), (4, 4))
    pair_factors_4d = ((2, 2), (4, 4), (2, 4))
    cases = []
    for ndim in (3, 4):
        for d in range(ndim):
            for b in single_factors:
                cases.append((ndim, (d, ), (b, )))
        pair_factors = pair_factors_3d if ndim == 3 else pair_factors_4d
        for d0, d1 in itertools.combinations(range(ndim), 2):
            for b0, b1 in pair_factors:
                cases.append((ndim, (d0, d1), (b0, b1)))
    return cases


CASES = build_cases()


def case_id(case):
    ndim, dims, factors = case
    dimspec = "_".join("d%dx%d" % (d, b) for d, b in zip(dims, factors))
    return "%dd_%s" % (ndim, dimspec)


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_block_elementwise(case):
    ndim, dims, factors = case
    n = 8  # divisible by every block factor (2, 4)
    masks, facs = masks_and_factors(ndim, dims, factors)

    sdfg = PROGRAMS[ndim].to_sdfg(simplify=True)
    SplitDimensions(split_map={"A": (masks, facs)}).apply_pass(sdfg, {})
    normalize_schedule_for_layout(sdfg)
    sdfg.validate()

    A_logical = numpy.random.default_rng(0).random((n, ) * ndim)
    oracle = A_logical * 2.0 + 1.0

    phys_shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["A"].shape)
    A_in = block_pack(A_logical, masks, facs)
    assert A_in.shape == phys_shape, (A_in.shape, phys_shape)

    C = numpy.zeros((n, ) * ndim)
    sdfg(A=A_in, C=C, N=n)

    assert numpy.allclose(C, oracle), "mismatch for %s: max abs err %g" % (case_id(case), numpy.abs(C - oracle).max())


if __name__ == "__main__":
    for c in CASES:
        test_block_elementwise(c)
    print("block 3d/4d sweep PASS (%d cases)" % len(CASES))