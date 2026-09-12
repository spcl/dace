# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from sympy import Min, Max

import dace
from dace.symbolic import pystr_to_symbolic, shapes_equal, simplify_ext, symbol


def test_simplify_ext_min() -> None:
    N = symbol("N")

    assert simplify_ext(Min(N, 4) + 1) == Min(N + 1, 5)
    assert simplify_ext(Max(N, 4) + 1) == Max(N + 1, 5)

    untouched = Min(N, 4)
    assert simplify_ext(untouched) == untouched


def test_shapes_equal_compares_by_name() -> None:
    """Two instances of one name -- what a rebuilt descriptor and a reparsed bound produce -- are
    the same dimension. Raw '==' calls them different, which is the bug this exists to stop."""
    wide = symbol("SEQ", dace.int32)
    narrow = symbol("SEQ", dace.int64)
    parsed = pystr_to_symbolic("SEQ")
    # the premise: identity disagrees with the name
    assert wide is not narrow and wide != narrow

    assert shapes_equal([4, wide], [4, narrow])
    assert shapes_equal([wide, narrow], [parsed, parsed])
    assert shapes_equal([wide * 2], [narrow * 2])
    assert shapes_equal([], [])

    # a genuine mismatch must still be reported, in rank and in extent
    assert not shapes_equal([4, wide], [4, wide, 1])
    assert not shapes_equal([4, wide], [5, wide])
    assert not shapes_equal([wide], [symbol("SOTHER")])
    assert not shapes_equal([wide + 1], [narrow])


def test_refold_booleans_folds_literal_arms():
    """The parser builds non-evaluating AND/OR nodes; refold_booleans re-constructs them so a
    literal arm short-circuits even when the other arm is an unresolved relational (cloudsc:
    ``AND(False, k >= m)`` guarded a dead branch that DeadStateElimination could not prune)."""
    import sympy

    from dace.symbolic import pystr_to_symbolic, refold_booleans

    assert refold_booleans(pystr_to_symbolic('(k < k) and (k >= m)')) is sympy.false
    assert refold_booleans(pystr_to_symbolic('(k <= k) or (k >= m)')) is sympy.true
    # A nested literal folds bottom-up; an undecidable pair stays symbolic.
    assert refold_booleans(pystr_to_symbolic('((k < k) or (k > m)) and (k < n)')) == \
        refold_booleans(pystr_to_symbolic('(k > m) and (k < n)'))
    folded = refold_booleans(pystr_to_symbolic('(k < m) and (k < n)'))
    assert str(folded.func) == 'AND'


def test_floordiv_on_a_symbol_is_int_floor():
    """One extent must have ONE spelling. ``pystr_to_symbolic`` maps ``//`` to ``int_floor`` and
    ``SymExpr`` routes it there too; a symbol inheriting sympy's ``floor(x/y)`` made the Python
    spelling of the same extent compare unequal to the parsed one."""
    from dace.symbolic import equal, pystr_to_symbolic, symbol

    n = symbol("n", dtype=dace.int64, positive=True)
    assert equal(n // 2, pystr_to_symbolic('n // 2')) is True
    assert equal(64 // n, pystr_to_symbolic('64 // n')) is True


def test_relax_int_floor_hands_the_solver_a_head_it_can_invert():
    """``int_floor`` is a bare Function to sympy, which raises rather than declining on one."""
    from dace.symbolic import pystr_to_symbolic, relax_int_floor

    assert str(relax_int_floor(pystr_to_symbolic('(h - 7) // 2 + 1'))) == 'floor(h/2 - 7/2) + 1'
    assert str(relax_int_floor(pystr_to_symbolic('int_ceil(h, 2)'))) == 'ceiling(h/2)'


if __name__ == "__main__":
    test_simplify_ext_min()
    test_shapes_equal_compares_by_name()
    test_refold_booleans_folds_literal_arms()
    test_floordiv_on_a_symbol_is_int_floor()
    test_relax_int_floor_hands_the_solver_a_head_it_can_invert()
