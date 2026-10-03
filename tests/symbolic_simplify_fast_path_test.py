# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.symbolic.simplify`` answers numbers, plain symbols, integer linear forms and integer monomials
without running ``sympy.simplify``; the answer must be exactly what ``sympy.simplify`` returns."""
import random
from collections.abc import Callable

import pytest
import sympy

import dace

SEED = 20260914

#: The real ``sympy.simplify``, captured before any test replaces it with a spy.
REFERENCE_SIMPLIFY = sympy.simplify

#: Assumption sets ``sympy.simplify`` may exploit; none makes a symbol zero or infinite.
FLAVORS = ({}, {
    "integer": True
}, {
    "positive": True
}, {
    "nonnegative": True
}, {
    "negative": True
}, {
    "nonzero": True
}, {
    "positive": True,
    "integer": True
}, {
    "real": True
}, {
    "extended_positive": True
}, {
    "odd": True
}, {
    "imaginary": True
})


def spy_on_sympy_simplify(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every expression ``dace.symbolic.simplify`` hands to ``sympy.simplify``, starting from a cold cache."""
    calls: list[object] = []

    def recording_simplify(expr: object, **kwargs: object) -> object:
        calls.append(expr)
        return REFERENCE_SIMPLIFY(expr, **kwargs)

    monkeypatch.setattr(sympy, "simplify", recording_simplify)
    dace.symbolic.simplify.cache_clear()
    return calls


def plain_symbols() -> list[sympy.Symbol]:
    symbols: list[sympy.Symbol] = []
    for index, flavor in enumerate(FLAVORS):
        symbols.extend(sympy.Symbol(f"{name}{index}", **flavor) for name in ("N", "M", "i"))
    symbols.extend([dace.symbol("N"), dace.symbol("M", dace.int64), dace.symbol("K", positive=True)])
    return symbols


def structurally_equal(first: object, second: object) -> bool:
    return type(first) is type(second) and first == second and sympy.srepr(first) == sympy.srepr(second)


def random_linear(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    expr = sympy.Integer(rng.randint(-6, 6))
    for sym in rng.sample(pool, rng.randint(1, 4)):
        expr = expr + rng.choice((-7, -3, -2, -1, 1, 2, 5)) * sym
    return expr


def random_monomial(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    expr = sympy.Integer(rng.choice((1, -1, 2, -3, 12)))
    for sym in rng.sample(pool, rng.randint(1, 3)):
        expr = expr * sym**rng.choice((1, 2, 3, -1, -2))
    return expr


def random_affine_product(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    return rng.choice(pool) * random_linear(rng, pool) * random_linear(rng, pool)


def random_affine_power(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    return rng.choice((1, -1, 2)) * random_linear(rng, pool)**rng.choice((2, 3, -1))


def random_rational_form(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    return random_linear(rng, pool) / rng.choice((2, 3, 4)) + rng.choice(pool) / 2


def random_unevaluated_add(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    sym = rng.choice(pool)
    return sympy.Add(sym, rng.choice((sym, sympy.Integer(1), 2 * sym, rng.choice(pool))), evaluate=False)


def random_degenerate_symbol_form(rng: random.Random, pool: list[sympy.Symbol]) -> sympy.Expr:
    degenerate = rng.choice((sympy.Symbol("z", zero=True), sympy.Symbol("w", infinite=True)))
    return rng.choice((degenerate + rng.choice(pool), 2 * degenerate * rng.choice(pool), 1 / degenerate))


def generate(makers: tuple[Callable[[random.Random, list[sympy.Symbol]], sympy.Expr], ...],
             count: int) -> list[sympy.Expr]:
    rng = random.Random(SEED)
    pool = plain_symbols()
    return [rng.choice(makers)(rng, pool) for index in range(count)]


@pytest.mark.parametrize("expr", [
    sympy.Integer(0),
    sympy.Integer(-4),
    sympy.Rational(3, 7),
    sympy.Float(2.5),
    sympy.oo,
    sympy.Symbol("N"),
    sympy.Symbol("N", positive=True, integer=True),
    sympy.Symbol("N") - 1,
    1 - sympy.Symbol("N"),
    -sympy.Symbol("N") - sympy.Symbol("M"),
    2 * sympy.Symbol("N") - 3 * sympy.Symbol("M", nonnegative=True) + 7,
    sympy.Symbol("N") * sympy.Symbol("M"),
    -2 * sympy.Symbol("N")**2 * sympy.Symbol("M"),
    sympy.Symbol("N", positive=True)**3,
    -3 / (sympy.Symbol("N") * sympy.Symbol("M")**2),
],
                         ids=str)
def test_numbers_symbols_linear_forms_and_monomials_skip_sympy_and_come_back_unchanged(
        expr: sympy.Basic, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_on_sympy_simplify(monkeypatch)

    result = dace.symbolic.simplify(expr)

    assert calls == []
    assert result is expr
    assert structurally_equal(result, REFERENCE_SIMPLIFY(expr))


def test_dace_symbol_keeps_its_dtype_through_a_linear_form(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_on_sympy_simplify(monkeypatch)
    expr = 4 * dace.symbol("N", dace.int64) - 1

    result = dace.symbolic.simplify(expr)

    assert calls == []
    assert structurally_equal(result, REFERENCE_SIMPLIFY(expr))
    assert [s.dtype for s in result.free_symbols] == [dace.int64]


def test_python_int_becomes_the_sympy_integer_sympy_would_return(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_on_sympy_simplify(monkeypatch)

    result = dace.symbolic.simplify(7)

    assert calls == []
    assert type(result) is sympy.Integer
    assert structurally_equal(result, REFERENCE_SIMPLIFY(7))


@pytest.mark.parametrize(
    "expr",
    [
        (sympy.Symbol("N") - 1) * (sympy.Symbol("N") + 1),  # expands to N**2 - 1
        (2 - sympy.Symbol("N"))**2,  # base sign is normalized
        sympy.Symbol("N") * (2 * sympy.Symbol("M") + 2),  # content 2 is pulled out
        sympy.Symbol("z", zero=True),  # folds to 0
        1 / sympy.Symbol("w", infinite=True),  # folds to 0
        sympy.Add(sympy.Symbol("N"), sympy.Symbol("N"), evaluate=False),  # folds to 2*N
        sympy.Symbol("N") / 2,
        sympy.Mod(sympy.Symbol("N"), 2),
        sympy.Min(sympy.Symbol("N"), sympy.Symbol("M")),
        dace.symbolic.int_floor(sympy.Symbol("N"), 2),
    ],
    ids=str)
def test_shapes_outside_the_proven_rules_are_handed_to_sympy(expr: sympy.Basic,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_on_sympy_simplify(monkeypatch)

    result = dace.symbolic.simplify(expr)

    assert calls == [expr]
    assert structurally_equal(result, REFERENCE_SIMPLIFY(expr))


def test_generated_linear_forms_and_monomials_skip_sympy_and_match_it(monkeypatch: pytest.MonkeyPatch) -> None:
    exprs = generate((random_linear, random_monomial), 400)
    calls = spy_on_sympy_simplify(monkeypatch)

    results = [dace.symbolic.simplify(e) for e in exprs]

    assert calls == []
    assert [(e, r) for e, r in zip(exprs, results) if not structurally_equal(r, REFERENCE_SIMPLIFY(e))] == []


def test_generated_near_misses_match_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    exprs = generate((random_affine_product, random_affine_power, random_rational_form, random_unevaluated_add,
                      random_degenerate_symbol_form, random_linear, random_monomial), 300)
    calls = spy_on_sympy_simplify(monkeypatch)

    results = [dace.symbolic.simplify(e) for e in exprs]

    assert [(e, r) for e, r in zip(exprs, results) if not structurally_equal(r, REFERENCE_SIMPLIFY(e))] == []
    assert 0 < len(calls) < len(dict.fromkeys((type(e), e) for e in exprs))
