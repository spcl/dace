# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Comprehensive numerical BLOCK-layout sweep over 1D and 2D kernels.

The BLOCK family (``SplitDimensions``) splits one dimension of an array ``A`` into an outer
tile-index and an inner tile: an access ``A[.., i, ..]`` where ``i`` is blocked by ``b`` becomes
``A[.., int_floor(i, b), .., Mod(i, b)]``, with the inner tile axis APPENDED LAST. Because the
descriptor shape CHANGES, the run closure must physically PACK the logical operand into that
``[.. positions .., tile]`` layout:

  * blocking the FINAL dimension leaves the tile axis already last -> a plain C-reshape suffices
    (1D ``x`` -> ``[N/b, b]``; 2D dim-1 ``A`` -> ``[M, N/b, b]``).
  * blocking a NON-final dimension moves the tile axis to the end -> reshape THEN transpose
    (2D dim-0 ``A`` -> ``[M/b, N, b] = A.reshape(M/b, b, N).transpose(0, 2, 1)``).

When the blocked array is the OUTPUT, the physical buffer is recovered back to logical shape (the
inverse pack) before comparing.

This sweep drives that end to end. For every case it applies ``SplitDimensions`` then
``normalize_schedule_for_layout`` (re-tiling the map to the block width), compiles, runs, and asserts
the result matches a numpy oracle (``numpy.allclose``). Coverage: four kernels (1D scale, 1D 3-point
stencil, 2D elementwise, 2D out-of-place transpose), dtypes float64 and float32, blocking the input
OR the output, blocking the final dim (plain reshape) OR a non-final dim (reshape+transpose), by
factors {2, 4, 8} over divisible extents.
"""
import numpy
import pytest

import dace

from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout
from dace.transformation.layout.split_dimensions import SplitDimensions

ALPHA = 2.0
BETA = 1.0
W0, W1, W2 = 0.25, 0.5, 0.25


# --------------------------------------------------------------------------- #
#  Kernel builders (fresh dtype-parametrized SDFGs)
# --------------------------------------------------------------------------- #
def build_scale1d(dt):
    N = dace.symbol("N")

    @dace.program
    def scale1d(x: dt[N], y: dt[N]):
        for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
            y[i] = ALPHA * x[i]

    return scale1d.to_sdfg(simplify=True)


def build_stencil1d(dt):
    N = dace.symbol("N")

    @dace.program
    def stencil1d(x: dt[N], y: dt[N]):
        for i in dace.map[1:N - 1] @ dace.ScheduleType.Sequential:
            y[i] = W0 * x[i - 1] + W1 * x[i] + W2 * x[i + 1]

    return stencil1d.to_sdfg(simplify=True)


def build_elem2d(dt):
    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def elem2d(A: dt[M, N], B: dt[M, N]):
        for i, j in dace.map[0:M, 0:N] @ dace.ScheduleType.Sequential:
            B[i, j] = ALPHA * A[i, j] + BETA

    return elem2d.to_sdfg(simplify=True)


def build_transpose2d(dt):
    N = dace.symbol("N")

    @dace.program
    def transpose2d(A: dt[N, N], B: dt[N, N]):
        for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
            B[j, i] = A[i, j]

    return transpose2d.to_sdfg(simplify=True)


# --------------------------------------------------------------------------- #
#  Inputs and numpy oracles (computed in the operand dtype)
# --------------------------------------------------------------------------- #
def scale1d_inputs(np_dt):
    return {"x": numpy.random.default_rng(0).random(32).astype(np_dt)}


def scale1d_oracle(inputs):
    return {"y": (ALPHA * inputs["x"]).astype(inputs["x"].dtype)}


def stencil1d_inputs(np_dt):
    return {"x": numpy.random.default_rng(1).random(32).astype(np_dt)}


def stencil1d_oracle(inputs):
    x = inputs["x"]
    y = numpy.zeros_like(x)  # boundaries y[0], y[-1] are never written -> 0
    y[1:-1] = W0 * x[:-2] + W1 * x[1:-1] + W2 * x[2:]
    return {"y": y}


def elem2d_inputs(np_dt):
    return {"A": numpy.random.default_rng(2).random((16, 24)).astype(np_dt)}


def elem2d_oracle(inputs):
    return {"B": (ALPHA * inputs["A"] + BETA).astype(inputs["A"].dtype)}


def transpose2d_inputs(np_dt):
    return {"A": numpy.random.default_rng(3).random((32, 32)).astype(np_dt)}


def transpose2d_oracle(inputs):
    return {"B": inputs["A"].T.copy()}


# ``arrays`` maps NAME -> (role, logical_shape); ``role`` is "in" or "out".
KERNELS = {
    "scale1d": {
        "build": build_scale1d,
        "syms": {
            "N": 32
        },
        "arrays": {
            "x": ("in", (32, )),
            "y": ("out", (32, ))
        },
        "inputs": scale1d_inputs,
        "oracle": scale1d_oracle,
    },
    "stencil1d": {
        "build": build_stencil1d,
        "syms": {
            "N": 32
        },
        "arrays": {
            "x": ("in", (32, )),
            "y": ("out", (32, ))
        },
        "inputs": stencil1d_inputs,
        "oracle": stencil1d_oracle,
    },
    "elem2d": {
        "build": build_elem2d,
        "syms": {
            "M": 16,
            "N": 24
        },
        "arrays": {
            "A": ("in", (16, 24)),
            "B": ("out", (16, 24))
        },
        "inputs": elem2d_inputs,
        "oracle": elem2d_oracle,
    },
    "transpose2d": {
        "build": build_transpose2d,
        "syms": {
            "N": 32
        },
        "arrays": {
            "A": ("in", (32, 32)),
            "B": ("out", (32, 32))
        },
        "inputs": transpose2d_inputs,
        "oracle": transpose2d_oracle,
    },
}

DACE_DTYPE = {"float64": dace.float64, "float32": dace.float32}
NP_DTYPE = {"float64": numpy.float64, "float32": numpy.float32}
FACTORS = (2, 4, 8)


# --------------------------------------------------------------------------- #
#  Pack (logical -> physical blocked descriptor) and its inverse
# --------------------------------------------------------------------------- #
def split_layout(logical_shape, masks, factors):
    """Split order axes for a blocked descriptor: the intermediate ``reshape`` shape plus the
    position axes (kept up front, in dim order) and tile axes (moved to the end)."""
    split_shape = []
    pos_axes = []
    tile_axes = []
    axis = 0
    for d, extent in enumerate(logical_shape):
        if masks[d]:
            split_shape.append(extent // factors[d])
            split_shape.append(factors[d])
            pos_axes.append(axis)
            tile_axes.append(axis + 1)
            axis += 2
        else:
            split_shape.append(extent)
            pos_axes.append(axis)
            axis += 1
    return split_shape, pos_axes + tile_axes


def block_pack(arr, masks, factors):
    """Lay logical ``arr`` into the physical descriptor ``SplitDimensions`` produces: split each
    blocked dim into ``(outer, tile)`` then move every tile axis to the end. Returns a fresh
    contiguous copy (DaCe rejects numpy views as arguments)."""
    split_shape, perm = split_layout(arr.shape, masks, factors)
    return arr.reshape(split_shape).transpose(perm).copy()


def block_unpack(phys, masks, factors, logical_shape):
    """Inverse of :func:`block_pack`: recover a physical blocked buffer back to ``logical_shape``."""
    _, perm = split_layout(logical_shape, masks, factors)
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return phys.transpose(inv).reshape(logical_shape).copy()


# --------------------------------------------------------------------------- #
#  Case enumeration
# --------------------------------------------------------------------------- #
def build_cases():
    """Enumerate ``(kernel, dtype, array, dim, factor)`` -- each blockable dimension of every array
    (input and output) blocked by every divisible factor, over both dtypes."""
    cases = []
    for kname, meta in KERNELS.items():
        for dtn in ("float64", "float32"):
            for aname, (role, shape) in meta["arrays"].items():
                for dim in range(len(shape)):
                    for factor in FACTORS:
                        if shape[dim] % factor == 0:
                            cases.append((kname, dtn, aname, dim, factor))
    return cases


CASES = build_cases()


def case_id(case):
    kname, dtn, aname, dim, factor = case
    return "%s-%s-%s-d%d-b%d" % (kname, dtn, aname, dim, factor)


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_block_sweep(case):
    kname, dtn, aname, dim, factor = case
    meta = KERNELS[kname]
    dt = DACE_DTYPE[dtn]
    np_dt = NP_DTYPE[dtn]
    syms = dict(meta["syms"])

    rank = len(meta["arrays"][aname][1])
    masks = [i == dim for i in range(rank)]
    facs = [factor if i == dim else 1 for i in range(rank)]

    sdfg = meta["build"](dt)
    SplitDimensions(split_map={aname: (masks, facs)}).apply_pass(sdfg, {})
    normalize_schedule_for_layout(sdfg)
    sdfg.validate()

    inputs = meta["inputs"](np_dt)
    reference = meta["oracle"](inputs)

    args = dict(syms)

    # Inputs: pack the blocked one into its physical descriptor, pass the rest logical.
    for name, (role, shape) in meta["arrays"].items():
        if role != "in":
            continue
        arr = inputs[name]
        if name == aname:
            phys_shape = tuple(int(dace.symbolic.evaluate(s, syms)) for s in sdfg.arrays[name].shape)
            arr = block_pack(arr, masks, facs)
            assert arr.shape == phys_shape, (arr.shape, phys_shape)
        args[name] = arr.copy()

    # Outputs: allocate the physical (possibly blocked) buffer, recover it after the run.
    out_buffers = {}
    for name, (role, shape) in meta["arrays"].items():
        if role != "out":
            continue
        phys_shape = tuple(int(dace.symbolic.evaluate(s, syms)) for s in sdfg.arrays[name].shape)
        buf = numpy.zeros(phys_shape, dtype=np_dt)
        args[name] = buf
        out_buffers[name] = buf

    sdfg(**args)

    tol = dict(rtol=1e-5, atol=1e-6) if dtn == "float32" else dict(rtol=1e-10, atol=1e-12)
    for name, buf in out_buffers.items():
        logical_shape = meta["arrays"][name][1]
        got = block_unpack(buf, masks, facs, logical_shape) if name == aname else buf
        err = numpy.abs(got.astype(numpy.float64) - reference[name].astype(numpy.float64)).max()
        assert numpy.allclose(got, reference[name], **tol), "mismatch for %s: max abs err %g" % (case_id(case), err)


if __name__ == "__main__":
    for c in CASES:
        test_block_sweep(c)
    print("block 1d/2d sweep PASS (%d cases)" % len(CASES))
