# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import sympy
from sympy import Min, Max

import dace
from dace import symbolic
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


def test_and_or_of_two_symbolic_booleans_keep_both_operands():
    """``AND``/``OR`` evaluated two symbolic booleans with Python's ``and``/``or``, which returns one
    operand: ``(not p) and (not q)`` became ``not q``. ConditionFusion writes simplified guards back
    from these trees, so the dropped conjunct reached the generated code."""
    import sympy
    from dace.symbolic import pystr_to_symbolic
    p, q = sympy.symbols('p q')
    for text, want in (('(not p) and (not q)', sympy.And(~p, ~q)), ('(not p) or (not q)', sympy.Or(~p, ~q)),
                       ('p and ((not p) and (not q))', sympy.false)):
        parsed = pystr_to_symbolic(text)
        for pv in (False, True):
            for qv in (False, True):
                assert parsed.subs({p: pv, q: qv}) == want.subs({p: pv, q: qv}), (text, parsed, pv, qv)


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


@pytest.mark.parametrize('text', ['N - 1 < M', 'i + 2*j', 'x ^ 2', 'min(a, b)', 'a\n+ b', 'True', '3.5e-3'])
@pytest.mark.parametrize('evaluate', [None, True, False])
def test_string_parse_on_the_shared_namespace_matches_sympify(text: str, evaluate: bool | None) -> None:
    expected = sympy.sympify(text, symbolic._PYSTR2SYM_locals, evaluate=evaluate)
    assert sympy.srepr(symbolic.sympify_text(text, evaluate)) == sympy.srepr(expected)


def test_string_parse_error_is_a_sympify_error() -> None:
    with pytest.raises(sympy.SympifyError):
        symbolic.sympify_text('1 +', None)


def test_parse_leaves_the_shared_namespace_unchanged() -> None:
    before = dict(symbolic.SYMPY_PARSER_GLOBALS)
    symbolic.sympify_text('sqrt(x) + pi + x ^ y', None)
    assert symbolic.SYMPY_PARSER_GLOBALS == before


def dace_renaming(expr: sympy.Basic) -> dict:
    return {atom: symbol(atom.name, dace.int64) for atom in expr.atoms(sympy.Symbol)}


@pytest.mark.parametrize('text', ['i + 1 < N', 'Min(i, N - 1) + 2*j**2', 'int_floor(N, 2) >= j'])
def test_renaming_plain_symbols_matches_subs(text: str) -> None:
    raw = symbolic.sympify_text(text, None)
    repl = dace_renaming(raw)
    renamed = symbolic.rename_symbols(raw, repl)
    assert sympy.srepr(renamed) == sympy.srepr(raw.subs(repl))
    assert {s.dtype for s in renamed.free_symbols} == {dace.int64}


def test_renaming_under_a_piecewise_matches_subs() -> None:
    x = sympy.Symbol('x')
    raw = sympy.Piecewise((x, x < 0), (x + 1, True))
    repl = dace_renaming(raw)
    assert sympy.srepr(symbolic.rename_symbols(raw, repl)) == sympy.srepr(raw.subs(repl))


def test_renaming_two_symbols_of_one_name_matches_subs() -> None:
    plain, typed = sympy.Symbol('N'), symbol('N', dace.int32)
    raw = plain + 2 * typed
    repl = {plain: symbol('N', dace.int64), typed: symbol('N', dace.uint8)}
    assert sympy.srepr(symbolic.rename_symbols(raw, repl)) == sympy.srepr(raw.subs(repl))


def test_a_container_named_like_a_sympy_function_parses_as_a_subscript() -> None:
    # ``rf`` is sympy's RisingFactorial; a subscript on a container of that name must stay a subscript.
    parsed = symbolic.pystr_to_symbolic('rf[0, jl]')
    assert isinstance(parsed, symbolic.Subscript)
    assert isinstance(parsed.args[0], sympy.Symbol) and parsed.args[0].name == 'rf'


if __name__ == "__main__":
    test_simplify_ext_min()
    test_shapes_equal_compares_by_name()
    test_refold_booleans_folds_literal_arms()
    test_and_or_of_two_symbolic_booleans_keep_both_operands()
    test_floordiv_on_a_symbol_is_int_floor()
    test_relax_int_floor_hands_the_solver_a_head_it_can_invert()
