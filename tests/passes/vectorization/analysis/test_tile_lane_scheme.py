# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``TileLaneScheme``, ``TileNameScheme`` and the
``assert_no_laneid_in_tile_path`` audit helper.

The v2 K-dim tile-op track never mints per-lane scalars; this audit is
the load-bearing invariant that catches accidental leaks from a buggy
prep pass.
"""
import pytest

import dace
from dace.transformation.passes.vectorization.utils.name_schemes import (
    LaneIdScheme,
    TileLaneScheme,
    TileNameScheme,
    assert_no_laneid_in_tile_path,
)


@pytest.mark.parametrize("indices", [(0,), (3,), (1, 2), (0, 1, 7), (4, 0, 2)])
def test_tile_lane_scheme_roundtrip(indices):
    """``make`` ∘ ``parse`` is the identity for K ∈ {1, 2, 3}."""
    encoded = TileLaneScheme.make("foo", indices)
    parsed = TileLaneScheme.parse(encoded)
    assert parsed is not None
    base, idx = parsed
    assert base == "foo"
    assert idx == indices


def test_tile_lane_scheme_rejects_empty_indices():
    """``make`` refuses an empty index tuple (K must be > 0)."""
    with pytest.raises(ValueError, match="non-empty"):
        TileLaneScheme.make("foo", ())


def test_tile_lane_scheme_parse_returns_none_for_non_lane_name():
    """Names that are not lane-encoded parse as ``None``."""
    assert TileLaneScheme.parse("just_a_name") is None
    assert TileLaneScheme.parse("foo_tilelane_") is None
    assert TileLaneScheme.parse("foo_tilelane_x1") is None


def test_tile_lane_scheme_is_tilelane_accepts_legacy_1d_laneid():
    """The K=1 legacy ``LaneIdScheme`` form is also recognized."""
    legacy = LaneIdScheme.make("bar", 3)
    assert TileLaneScheme.is_tilelane(legacy)
    assert TileLaneScheme.is_tilelane(TileLaneScheme.make("bar", (3,)))
    assert not TileLaneScheme.is_tilelane("plain_name")


def test_tile_name_scheme_make_roundtrip():
    """``TileNameScheme.make_*`` produce the canonical suffixes."""
    assert TileNameScheme.make_tile("a") == "a_tile"
    assert TileNameScheme.make_idx("a") == "a_tile_idx"
    assert TileNameScheme.make_cond_mask("br") == "br_tile_cond_mask"
    assert TileNameScheme.ITER_MASK == "_tile_iter_mask"


def test_tile_name_scheme_is_tile_transient():
    """The classifier recognizes the four reserved suffixes."""
    assert TileNameScheme.is_tile_transient("a_tile")
    assert TileNameScheme.is_tile_transient("a_tile_idx")
    assert TileNameScheme.is_tile_transient("br_tile_cond_mask")
    assert TileNameScheme.is_tile_transient("_tile_iter_mask")
    assert not TileNameScheme.is_tile_transient("a")
    assert not TileNameScheme.is_tile_transient("a_vec")
    assert not TileNameScheme.is_tile_transient("a_packed")


def test_assert_no_laneid_passes_on_clean_sdfg():
    """A trivial SDFG with no per-lane names triggers no failure."""
    sdfg = dace.SDFG("clean")
    sdfg.add_array("A", (8,), dace.float64)
    state = sdfg.add_state("main")
    state.add_access("A")
    assert_no_laneid_in_tile_path(sdfg)


def test_assert_no_laneid_flags_legacy_1d_lane_in_arrays():
    """The audit catches a leaked 1D ``_laneid_<i>`` array name."""
    sdfg = dace.SDFG("leaked_1d")
    sdfg.add_array("a_laneid_3", (1,), dace.float64)
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_laneid_in_tile_path(sdfg)


def test_assert_no_laneid_flags_multi_dim_tilelane_in_symbols():
    """The audit catches a leaked multi-dim ``_tilelane_<...>`` symbol."""
    sdfg = dace.SDFG("leaked_kdim")
    sdfg.add_symbol(TileLaneScheme.make("idx", (2, 3)), dace.int64)
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_laneid_in_tile_path(sdfg)


def test_assert_no_laneid_walks_nested_sdfgs():
    """A leak inside a nested SDFG is also caught."""
    outer = dace.SDFG("outer_clean")
    outer.add_array("X", (4,), dace.float64)
    inner = dace.SDFG("inner_leaked")
    inner.add_array("y_laneid_0", (1,), dace.float64)
    state = outer.add_state("main")
    state.add_nested_sdfg(inner, inputs=set(), outputs=set())
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_laneid_in_tile_path(outer)
