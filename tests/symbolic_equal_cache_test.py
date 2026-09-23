# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.symbolic.equal`` asks the assumption solver once per distinct question: the cached verdict must be the
solver's own, and a question asked under an active global assumption must never be served from the cache."""
import pytest
import sympy

from dace import symbolic

N = symbolic.symbol('N', positive=True)
M = symbolic.symbol('M')
K = symbolic.symbol('K', nonnegative=True)

#: ``(a, b, is_length)``; the verdicts cover True, False and inconclusive.
QUESTIONS = [
    (N, N, True),
    (N, 0, False),
    (M, 0, False),
    (M - 1 + 1, M, True),
    (N, M, True),
    (K + 1, 0, False),
    (3, 4, True),
    (M * N, N * M, True),
    (symbolic.pystr_to_symbolic('kfdia - kidia + 1'), 0, False),
]


def solver_verdict(a, b, is_length: bool):
    facts = [q for arg in (a, b) for q in (sympy.Q.integer(arg), sympy.Q.positive(arg))] if is_length else []
    with sympy.assuming(*facts):
        return sympy.ask(sympy.Q.is_true(sympy.Eq(a, b)))


@pytest.mark.parametrize('a, b, is_length', QUESTIONS)
def test_a_cached_verdict_is_the_solvers(a, b, is_length: bool) -> None:
    symbolic.ask_equal.cache_clear()
    first = symbolic.equal(a, b, is_length)
    assert first == solver_verdict(sympy.sympify(a), sympy.sympify(b), is_length), (a, b)
    assert symbolic.equal(a, b, is_length) is first, 'the cached answer differs from the first one'


def test_a_repeated_question_reaches_the_solver_once(monkeypatch) -> None:
    """Codegen asks the same zero-size question per copy; each ask costs tens of milliseconds."""
    symbolic.ask_equal.cache_clear()
    asked = []
    real_ask = sympy.ask

    def counting_ask(*args, **kwargs):
        asked.append(args)
        return real_ask(*args, **kwargs)

    monkeypatch.setattr(sympy, 'ask', counting_ask)
    extent = symbolic.pystr_to_symbolic('kfdia - kidia + 1')
    verdicts = {symbolic.equal(extent, 0, is_length=False) for _ in range(20)}
    assert verdicts == {None} and len(asked) == 1, (verdicts, len(asked))


def test_a_global_assumption_bypasses_the_cache() -> None:
    """A verdict taken under ``sympy.assuming`` holds only there; serving it outside, or the reverse, is wrong."""
    symbolic.ask_equal.cache_clear()
    x = symbolic.symbol('x')
    assert symbolic.equal(x, 0, is_length=False) is None
    with sympy.assuming(sympy.Q.eq(x, 0)):
        assert symbolic.equal(x, 0, is_length=False) is True
    assert symbolic.equal(x, 0, is_length=False) is None
