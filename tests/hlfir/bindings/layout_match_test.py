"""``layout_match.decide_strategy`` — per-arg strategy picker.

Rule: only deep-copy when dimensionality / element type actually
differs.  If caller and callee agree on shape + dtype, alias.
"""
from __future__ import annotations

from dace.frontend.hlfir.bindings import (
    AliasStrategy,
    ComplexSplitStrategy,
    ExplicitCopyStrategy,
    FrozenArg,
    decide_strategy,
)
from dace.frontend.hlfir.bindings.fortran_interface import Member, OriginalArg


def test_alias_when_layouts_match():
    """Plain real array passed through unchanged → alias."""
    fa = FrozenArg(fortran_name="a",
                   sdfg_name="a",
                   kind="array",
                   dtype="float64",
                   rank=2,
                   shape=("n", "m"),
                   intent="inout")
    outer = OriginalArg(name="a", fortran_type="real(c_double)", rank=2, shape=("n", "m"), intent="inout")
    s = decide_strategy(fa, outer)
    assert isinstance(s, AliasStrategy)
    assert s.outer_expr == "a"
    assert s.inner_name == "a"
    assert s.shape_exprs == ("size(a, dim=1)", "size(a, dim=2)")


def test_alias_for_struct_member_when_layouts_match():
    """Struct member with compatible layout → alias via c_loc(st%u)."""
    fa = FrozenArg(fortran_name="u",
                   sdfg_name="st_u",
                   kind="array",
                   dtype="float64",
                   rank=2,
                   shape=("nproma", "nlev"),
                   intent="inout",
                   from_struct_member="st%u")
    member = Member(name="u", fortran_type="real(c_double)", rank=2, shape=("nproma", "nlev"))
    s = decide_strategy(fa, member)
    assert isinstance(s, AliasStrategy)
    assert s.outer_expr == "st%u"
    assert s.inner_name == "st_u"


def test_complex_split_when_layout_tag_set():
    """complex(c_double) outer + layout='complex_split' → two real arrays."""
    fa = FrozenArg(fortran_name="u",
                   sdfg_name="st_u_re",
                   kind="array",
                   dtype="float64",
                   rank=2,
                   shape=("nproma", "nlev"),
                   intent="inout",
                   from_struct_member="st%u",
                   layout="complex_split")
    member = Member(name="u", fortran_type="complex(c_double)", rank=2, shape=("nproma", "nlev"))
    s = decide_strategy(fa, member)
    assert isinstance(s, ComplexSplitStrategy)
    assert s.re_name == "st_u_re"
    assert s.im_name == "st_u_im"
    assert s.writeback is True  # intent inout


def test_explicit_copy_on_rank_mismatch():
    """Rank changes → ExplicitCopyStrategy fallback."""
    fa = FrozenArg(fortran_name="a",
                   sdfg_name="a_flat",
                   kind="array",
                   dtype="float64",
                   rank=1,
                   shape=("n_total", ),
                   intent="in")
    outer = OriginalArg(name="a", fortran_type="real(c_double)", rank=2, shape=("n", "m"), intent="in")
    s = decide_strategy(fa, outer)
    assert isinstance(s, ExplicitCopyStrategy)
    assert s.writeback is False  # intent in


def test_no_deep_copy_when_same_layout_and_dtype():
    """User's invariant: if ndim + dtype match, we must alias (zero
    copy), never emit a deep-copy loop.  Covered by the first two
    tests; this one guards against a regression that would prefer the
    ``ExplicitCopyStrategy`` fallback for the common case."""
    fa = FrozenArg(fortran_name="v", sdfg_name="v", kind="array", dtype="float64", rank=1, shape=("n", ), intent="in")
    outer = OriginalArg(name="v", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="in")
    s = decide_strategy(fa, outer)
    assert isinstance(s, AliasStrategy), f"expected alias, got {type(s).__name__}"
