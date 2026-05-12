# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``LaneIdScheme`` — the central lane-id naming helper used by the
vectorization pipeline. Covers the cases that previously bit
``expand_interstate_assignments_to_lanes``:

- Round-trip of ``make`` / ``parse``.
- Detection of doubly-laneid-encoded names like ``foo_laneid_3_laneid_0`` —
  the regex hack the pass used to carry could not classify these reliably.
- Boundary cases (empty base, non-digit suffix, no suffix at all).
"""
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def test_make_roundtrip():
    assert LaneIdScheme.make("idx", 0) == "idx_laneid_0"
    assert LaneIdScheme.make("idx", 7) == "idx_laneid_7"
    base, lane = LaneIdScheme.parse(LaneIdScheme.make("idx", 5))
    assert (base, lane) == ("idx", 5)


def test_parse_simple():
    assert LaneIdScheme.parse("idx_laneid_3") == ("idx", 3)
    assert LaneIdScheme.parse("a_laneid_0") == ("a", 0)
    assert LaneIdScheme.parse("__tmp_42_19_r_laneid_2") == ("__tmp_42_19_r", 2)


def test_parse_double_suffix_peels_only_trailing():
    """`foo_laneid_3_laneid_0` is the bug shape that broke the old regex hack.

    Strict parse peels only the *trailing* lane suffix; the base is left
    encoding the prior lane. Repeated parsing converges to the original base.
    """
    parsed = LaneIdScheme.parse("foo_laneid_3_laneid_0")
    assert parsed == ("foo_laneid_3", 0)
    parsed2 = LaneIdScheme.parse(parsed[0])
    assert parsed2 == ("foo", 3)
    assert LaneIdScheme.parse(parsed2[0]) is None


def test_parse_rejects_non_laneid():
    assert LaneIdScheme.parse("idx") is None
    assert LaneIdScheme.parse("idx_3") is None  # missing 'laneid' marker
    assert LaneIdScheme.parse("idx_laneid_") is None  # no digits
    assert LaneIdScheme.parse("idx_laneid_x") is None  # non-digit suffix
    assert LaneIdScheme.parse("idx_laneid_3a") is None  # non-digit tail


def test_parse_empty_base_is_allowed():
    """An empty base (`_laneid_5`) parses to ``("", 5)``. Not used in practice but
    documented behavior so callers don't have to special-case it."""
    assert LaneIdScheme.parse("_laneid_5") == ("", 5)


def test_is_laneid_matches_parse():
    cases = [
        ("idx", False),
        ("idx_laneid_0", True),
        ("idx_laneid_3_laneid_0", True),
        ("foo_laneid_x", False),
        ("foo_laneid_", False),
        ("_laneid_5", True),
    ]
    for name, expected in cases:
        assert LaneIdScheme.is_laneid(name) is expected, name


def test_make_with_str_base_and_int_lane_only():
    """``make`` accepts ``str`` base and ``int`` lane. No coercion shortcuts."""
    assert LaneIdScheme.make("a", 1) == "a_laneid_1"
    # Round-trip even when the base contains underscores or starts with `_laneid_`.
    assert LaneIdScheme.parse(LaneIdScheme.make("foo_bar", 4)) == ("foo_bar", 4)
    assert LaneIdScheme.parse(LaneIdScheme.make("_laneid_5", 2)) == ("_laneid_5", 2)
