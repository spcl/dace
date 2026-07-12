# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Modular / non-affine index READS vectorize as per-lane GATHERS (1-D and multi-dim).

A tile iter-var nested inside a function -- ``a[i % S]`` (a cyclic/residue index), ``a[i*i]`` --
is single-element but NOT contiguous across lanes, so it MUST NOT be contiguous-widened. The
tile path classifies it GATHER and builds a per-lane index tile ``[f(i+0), .., f(i+W-1)]`` (the
function EXPANDED inside per lane), then gathers. A gather READ is always sound (reads never
collide), so it stays bit-exact with the un-vectorized reference regardless of whether the period
``S`` divides the tile width.

This is the READ counterpart of the residue-scan seed ``a[i mod K]`` canon leaves beside its Scan
libnode (tsvc_2_5 scan_strided_sym / scan_strided_2), isolated from any recurrence.
"""
import numpy
import pytest

import dace
from tests.passes.vectorization.helpers.harness import S, X, Y, run_vectorization_test

pytestmark = pytest.mark.tile_nodes


@dace.program
def modular_gather_read_1d(out: dace.float64[X], a: dace.float64[S]):
    for i, in dace.map[0:X:1]:
        out[i] = a[i % S]


@dace.program
def modular_gather_read_2d(out: dace.float64[Y, X], a: dace.float64[Y, S]):
    # ``i`` is a plain (linear) tile dim; the inner index ``j % S`` wraps -> mixed linear + gather.
    for i, j in dace.map[0:Y:1, 0:X:1]:
        out[i, j] = a[i, j % S]


@dace.program
def square_gather_read_1d(out: dace.float64[X], a: dace.float64[S]):
    # ``(i * i) % S`` is non-affine (AFFINE-with-no-int-stride, wrapped) -> per-lane index
    # ``((i+l)**2) mod S``; kept in [0, S) so it stays in bounds for every i.
    for i, in dace.map[0:X:1]:
        out[i] = a[(i * i) % S]


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_modular_gather_read_1d(branch_mode, remainder_strategy):
    """``out[i] = a[i % S]`` -> per-lane gather ``a[(i+l) mod S]``; S=3 < 8 exercises the wrap."""
    xv, sv = 60, 3  # xv not a multiple of 8 -> remainder tile; sv < W -> cyclic wrap within a tile
    run_vectorization_test(
        dace_func=modular_gather_read_1d,
        arrays={"out": numpy.zeros(xv), "a": numpy.random.random(sv)},
        params={"X": xv, "S": sv},
        vector_width=8,
        sdfg_name="modular_gather_read_1d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_modular_gather_read_2d(branch_mode, remainder_strategy):
    """2-D ``out[i, j] = a[i, j % S]`` -- linear ``i`` tile dim + modular ``j`` gather dim."""
    yv, xv, sv = 8, 60, 3
    run_vectorization_test(
        dace_func=modular_gather_read_2d,
        arrays={"out": numpy.zeros((yv, xv)), "a": numpy.random.random((yv, sv))},
        params={"Y": yv, "X": xv, "S": sv},
        vector_width=8,
        sdfg_name="modular_gather_read_2d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_square_gather_read_1d(branch_mode, remainder_strategy):
    """``out[i] = a[(i * i) % S]`` -- a non-affine (no integer stride) index -> per-lane
    ``((i+l)**2) mod S``."""
    xv, sv = 60, 5  # xv not a multiple of 8 -> remainder tile
    run_vectorization_test(
        dace_func=square_gather_read_1d,
        arrays={"out": numpy.zeros(xv), "a": numpy.random.random(sv)},
        params={"X": xv, "S": sv},
        vector_width=8,
        sdfg_name="square_gather_read_1d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )
