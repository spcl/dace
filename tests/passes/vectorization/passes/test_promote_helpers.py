# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`classify_box_for_widths` -- FROZEN.

The :mod:`utils.promote_helpers` module was deleted in the walker-primary
migration (commit 1e55ea4ce). This test file is preserved for history
but skipped at module level until walker-primary equivalents land.
"""
import pytest

pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent-helper test -- promote_helpers module was deleted in the "
                              "walker-primary migration (commit 1e55ea4ce); will be replaced once the walker's lattice "
                              "classification has its own equivalent test slice.")


def _arr(shape, strides=None):
    """Build a throwaway 1D Array descriptor for classifier testing."""
    sdfg = dace.SDFG("probe")
    name = sdfg.add_array("X", shape, dace.float64, transient=False, strides=strides)[0]
    return sdfg.arrays[name]


def test_already_tile_shape_is_contiguous_short_circuit():
    """``arr.shape == widths`` short-circuits to CONTIGUOUS regardless of subset."""
    widths = (4, 8)
    arr = _arr(widths)
    # Use an iter-var-free subset; the short-circuit fires before the classifier runs.
    subset = subsets.Range([(0, w - 1, 1) for w in widths])
    cls = classify_box_for_widths(subset, arr, iter_vars=("i", "j"), widths=widths)
    assert cls.kind == TileAccessKind.CONTIGUOUS
    assert tuple(cls.dim_strides) == (1, 1)
    assert tuple(cls.match_dims) == (0, 1)


def test_contiguous_box_classified_via_classifier():
    """A genuine iter-var-based contiguous access is classified by the full classifier
    (no short-circuit fires because the array shape is larger than the tile shape)."""
    widths = (4, 8)
    arr = _arr((16, 32))
    subset = subsets.Range([
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i + 3"), 1),
        (dace.symbolic.pystr_to_symbolic("j"), dace.symbolic.pystr_to_symbolic("j + 7"), 1),
    ])
    cls = classify_box_for_widths(subset, arr, iter_vars=("i", "j"), widths=widths)
    assert cls.kind == TileAccessKind.CONTIGUOUS


def test_k1_broadcast_symbol_treated_as_contiguous_one_lane():
    """At K=1 the degenerate ``BROADCAST_SYMBOL`` case (iter-var-free single-element
    access) falls through to CONTIGUOUS with widths=(1,)."""
    widths = (1, )
    arr = _arr((16, ))
    # iter-var-free single-element subset
    subset = subsets.Range([(0, 0, 1)])
    cls = classify_box_for_widths(subset, arr, iter_vars=("i", ), widths=widths)
    assert cls.kind == TileAccessKind.CONTIGUOUS
    assert tuple(cls.dim_strides) == (1, )
    assert tuple(cls.match_dims) == (0, )


def test_non_box_access_raises():
    """A K>=2 broadcast-symbol access (iter-var-free subset, but K is not the
    K=1 single-lane endpoint) raises ``NotImplementedError`` so callers can
    fall back to gather / structured promotion instead of silently
    accepting a degenerate classification."""
    widths = (4, 8)
    arr = _arr((16, 32))
    # iter-var-free literal subset -- classifier sees BROADCAST_SYMBOL,
    # which is not in BOX_KINDS and not the K=1 single-lane endpoint, so
    # the helper raises.
    subset = subsets.Range([(0, 3, 1), (0, 7, 1)])
    with pytest.raises(NotImplementedError, match="classify_box_for_widths"):
        classify_box_for_widths(subset, arr, iter_vars=("i", "j"), widths=widths)


def test_box_kinds_constant_matches_classifier():
    """``BOX_KINDS`` covers exactly the kinds the classifier accepts as a perfect box."""
    # Sanity-check the constant rather than its internals -- the helper's
    # contract is "anything in BOX_KINDS is accepted; anything else raises".
    assert TileAccessKind.CONTIGUOUS in BOX_KINDS
    assert TileAccessKind.STRIDED in BOX_KINDS
    assert TileAccessKind.GATHER not in BOX_KINDS
    assert TileAccessKind.BROADCAST_SYMBOL not in BOX_KINDS
