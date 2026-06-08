# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Equivalence tests for :func:`classify_tile_access_compat`.

The compat shim drives the legacy ``TileAccessClassification`` API from
the per-dim classifier in :mod:`tile_access`. Each test feeds the same
input to BOTH the legacy classifier and the compat shim, then asserts
they agree on ``kind`` and the load-bearing fields. Once the shim is
wired into the descent (in place of the legacy classifier), this test
prevents silent drift between the two implementations.
"""
import pytest

import dace
from dace import symbolic
from dace.subsets import Range
from dace.transformation.passes.vectorization.utils.tile_access_compat import (
    classify_tile_access_compat, )
from dace.transformation.passes.vectorization.utils.tile_dims import (
    classify_tile_access as _classify_legacy,
    TileAccessKind,
)


def _R(*ranges):
    out = []
    for r in ranges:
        if len(r) == 2:
            lo, hi = r
            step = 1
        else:
            lo, hi, step = r
        out.append((symbolic.pystr_to_symbolic(str(lo)) if isinstance(lo, str) else lo,
                    symbolic.pystr_to_symbolic(str(hi)) if isinstance(hi, str) else hi,
                    symbolic.pystr_to_symbolic(str(step)) if isinstance(step, str) else step))
    return Range(out)


def _check_agree(subset, array_strides, iter_vars, *, expect_kind=None):
    legacy = _classify_legacy(subset, array_strides, iter_vars)
    shim = classify_tile_access_compat(subset, array_strides, iter_vars)
    if expect_kind is not None:
        assert legacy.kind == expect_kind, f"legacy classifier disagrees: {legacy.kind} != {expect_kind}"
        assert shim.kind == expect_kind, f"compat shim disagrees: {shim.kind} != {expect_kind}"
    assert legacy.kind == shim.kind, f"legacy={legacy.kind} shim={shim.kind}"
    if shim.kind in (TileAccessKind.CONTIGUOUS, TileAccessKind.STRIDED):
        assert tuple(shim.dim_strides) == tuple(legacy.dim_strides), \
            f"dim_strides: legacy={legacy.dim_strides} shim={shim.dim_strides}"
        # match_dims may differ on edge cases the legacy used to derive
        # from array_strides; the shim defers to per-dim iter-var order.
        # Don't strictly compare match_dims here.


# ---- whole-subset kind equivalence ----------------------------------


def test_compat_contiguous():
    """``arr[i, j]`` (identity stride on each dim) -> CONTIGUOUS."""
    r = _R(("i", "i"), ("j", "j"))
    _check_agree(r, (8, 1), iter_vars=("i", "j"), expect_kind=TileAccessKind.CONTIGUOUS)


def test_compat_contiguous_offset():
    """``arr[i + 1, j]`` (identity coeff with constant offset) -> CONTIGUOUS."""
    r = _R(("i + 1", "i + 1"), ("j", "j"))
    _check_agree(r, (8, 1), iter_vars=("i", "j"), expect_kind=TileAccessKind.CONTIGUOUS)


def test_compat_strided():
    """``arr[2*i]`` (non-unit coefficient) -> STRIDED."""
    r = _R(("2*i", "2*i"))
    _check_agree(r, (1, ), iter_vars=("i", ), expect_kind=TileAccessKind.STRIDED)


def test_compat_broadcast_constant():
    """``arr[0, 0]`` (no iter-var) -> BROADCAST_SYMBOL."""
    r = _R((0, 0), (0, 0))
    _check_agree(r, (8, 1), iter_vars=("i", "j"), expect_kind=TileAccessKind.BROADCAST_SYMBOL)


def test_compat_broadcast_outer_symbol():
    """``arr[M, N]`` (outer-scope symbols) -> BROADCAST_SYMBOL."""
    r = _R(("M", "M"), ("N", "N"))
    _check_agree(r, (8, 1), iter_vars=("i", "j"), expect_kind=TileAccessKind.BROADCAST_SYMBOL)


def test_compat_gather_via_subscript():
    """``arr[idx[i]]`` -> GATHER."""
    r = _R(("idx[i]", "idx[i]"))
    _check_agree(r, (1, ), iter_vars=("i", ), expect_kind=TileAccessKind.GATHER)


# ---- K-rank-lift transitions ----------------------------------------


def test_compat_k0_to_k2_full_splat():
    """Both dims invariant -> BROADCAST_SYMBOL on the K=2 tile."""
    r = _R(("M", "M"), ("N + 1", "N + 1"))
    _check_agree(r, (1, 1), iter_vars=("i", "j"), expect_kind=TileAccessKind.BROADCAST_SYMBOL)


# ---- AFFINE-with-non-isolable-coefficient -> legacy STRUCTURED ------


def test_compat_int_floor_falls_to_gather():
    """``arr[i // 2]`` -> legacy GATHER (lowered as a TileLoad (gather) over a
    computed index map at emit time). The new classifier tags this as
    AFFINE with no integer stride; the compat shim routes it to legacy
    GATHER to match."""
    r = _R(("i // 2", "i // 2"))
    legacy = _classify_legacy(r, (1, ), ("i", ))
    shim = classify_tile_access_compat(r, (1, ), ("i", ))
    assert legacy.kind == TileAccessKind.GATHER
    assert shim.kind == TileAccessKind.GATHER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
