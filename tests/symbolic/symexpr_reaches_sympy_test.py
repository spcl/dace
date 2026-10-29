# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``SymExpr`` (exact form plus over-approximation) must not be handed to sympy."""

from dace import symbolic
from dace.symbolic import SymExpr
from dace.transformation.passes.move_array_out_of_kernel import slot


def _facts() -> symbolic.Facts:
    return symbolic.Facts((), frozenset({"N", "M"}), frozenset({"N"}))


def test_is_small_reads_a_symexpr_by_its_exact_form():
    N = symbolic.symbol("N")
    M = symbolic.symbol("M")
    facts = _facts()
    assert symbolic.is_small(SymExpr(N, N + 1), facts)
    assert symbolic.is_small(N, facts)
    assert not symbolic.is_small(SymExpr(M, N), facts)


def test_is_small_extent_reads_a_symexpr_by_its_exact_form():
    assert symbolic.is_small_extent(SymExpr(4, 8), _facts())
    assert not symbolic.is_small_extent(SymExpr(128, 256), _facts())


def test_slot_divides_each_form_of_a_symexpr_bound():
    N = symbolic.symbol("N")
    got = slot(SymExpr(N, 2 * N), SymExpr(2, 3))
    assert isinstance(got, SymExpr)
    assert got.expr == symbolic.int_floor(N, 2)
    assert got.approx == symbolic.int_floor(2 * N, 3)


def test_slot_of_a_unit_step_keeps_the_symexpr_offset():
    N = symbolic.symbol("N")
    offset = SymExpr(N, N + 1)
    assert slot(offset, 1) == offset
