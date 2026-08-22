# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``LaneIdScheme`` Option B chunked form, ``TileNameScheme``
and the ``assert_no_laneid_in_tile_path`` audit helper.

After Step A.1, the legacy ``TileLaneScheme`` is replaced by the unified
``LaneIdScheme`` whose chunked ``_lane<d>id_<n>`` form covers K=1 and
K>=2 under one scheme. The v2 K-dim tile-op track never mints per-lane
scalars; this audit is the load-bearing invariant that catches accidental
leaks from a buggy prep pass.
"""
import pytest

import dace
from dace.transformation.passes.vectorization.utils.name_schemes import (
    LaneIdScheme,
    TileNameScheme,
    assert_no_laneid_in_tile_path,
)


@pytest.mark.parametrize("chunks", [((0, 0), ), ((0, 3), ), ((0, 1), (1, 2)), ((0, 0), (1, 1), (2, 7)),
                                    ((0, 4), (1, 0), (2, 2))])
def test_make_multi_parse_chunks_roundtrip(chunks):
    """``make_multi`` ∘ ``parse_chunks`` is the identity for K ∈ {1, 2, 3}."""
    encoded = LaneIdScheme.make_multi("foo", chunks)
    parsed = LaneIdScheme.parse_chunks(encoded)
    assert parsed is not None
    base, parsed_chunks = parsed
    assert base == "foo"
    assert parsed_chunks == chunks


def test_make_multi_empty_yields_base():
    """``make_multi(base, [])`` is the bare base; no chunk to round-trip."""
    assert LaneIdScheme.make_multi("foo", []) == "foo"
    assert LaneIdScheme.parse_chunks("foo") is None


def test_parse_chunks_returns_none_for_non_lane_name():
    """Names that are not lane-encoded parse as ``None``."""
    assert LaneIdScheme.parse_chunks("just_a_name") is None
    assert LaneIdScheme.parse_chunks("foo_lane0id_") is None
    assert LaneIdScheme.parse_chunks("foo_lane0id_x") is None


def test_is_lane_fanned_accepts_legacy_1d_laneid():
    """The legacy 1D ``LaneIdScheme.make`` form is also recognised."""
    legacy = LaneIdScheme.make("bar", 3)
    assert LaneIdScheme.is_lane_fanned(legacy)
    assert LaneIdScheme.is_lane_fanned(LaneIdScheme.make_dim("bar", 0, 3))
    assert LaneIdScheme.is_lane_fanned(LaneIdScheme.make_multi("bar", [(0, 3)]))
    assert not LaneIdScheme.is_lane_fanned("plain_name")


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
    sdfg.add_array("A", (8, ), dace.float64)
    state = sdfg.add_state("main")
    state.add_access("A")
    assert_no_laneid_in_tile_path(sdfg)


def test_assert_no_laneid_flags_legacy_1d_lane_in_arrays():
    """The audit catches a leaked 1D ``_laneid_<i>`` array name."""
    sdfg = dace.SDFG("leaked_1d")
    sdfg.add_array("a_laneid_3", (1, ), dace.float64)
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_laneid_in_tile_path(sdfg)


def test_assert_no_laneid_flags_chunked_form_in_symbols():
    """The audit catches a leaked Option B chunked-form symbol."""
    sdfg = dace.SDFG("leaked_kdim")
    sdfg.add_symbol(LaneIdScheme.make_multi("idx", [(0, 2), (1, 3)]), dace.int64)
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_laneid_in_tile_path(sdfg)


def test_assert_no_laneid_walks_nested_sdfgs():
    """A leak inside a nested SDFG is also caught."""
    outer = dace.SDFG("outer_clean")
    outer.add_array("X", (4, ), dace.float64)
    inner = dace.SDFG("inner_leaked")
    inner.add_array("y_laneid_0", (1, ), dace.float64)
    state = outer.add_state("main")
    state.add_nested_sdfg(inner, inputs=set(), outputs=set())
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_laneid_in_tile_path(outer)
