# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest

# Basic test cases
basic_cases = [
    ("", set()),
    ("x + y", {'x', 'y'}),
    ("x + y - z", {'x', 'y', 'z'}),
    ("e + f", {'e', 'f'}),
    ("1e5", set()),
    ("2e-10", set()),
    ("3e+8", set()),
    ("e = 1e5 + 2e-3", {'e'}),
    ("e * 1e10 + exp(e)", {'e', 'exp'}),
    ("x = 1e5 + y * 2e-3", {'x', 'y'}),
    ("min(x, y)", {'min', 'x', 'y'}),
    ("result = min(x, y) + max(a, b)", {'result', 'min', 'x', 'y', 'max', 'a', 'b'}),
    ("y = exp(x) + log(z)", {'y', 'exp', 'x', 'log', 'z'}),
    ("result = exp(e) + log(e)", {'result', 'exp', 'e', 'log'}),
    ("z = min(exp(x), max(log(y), 1e5))", {'z', 'min', 'exp', 'x', 'max', 'log', 'y'}),
    ("exp(1e-5)", {'exp'}),
    ("a = 2e+10 + b", {'a', 'b'}),
    ("log(e) + e", {'log', 'e'}),
    ("max(1e5, 2e-3)", {'max'}),
    ("my_var + _private + __dunder__", {'my_var', '_private', '__dunder__'}),
    ("camelCase + PascalCase + snake_case", {'camelCase', 'PascalCase', 'snake_case'}),
]

# Test cases with symbols_to_ignore
ignore_cases = [
    ("x + y + z", {'y'}, {'x', 'z'}),
    ("a + b + c + d + e", {'b', 'd'}, {'a', 'c', 'e'}),
    ("x = 1e5 + y", {'x'}, {'y'}),
    ("min(x, y)", {'min'}, {'x', 'y'}),
    ("result = min(x, y) + max(a, b)", {'min', 'max'}, {'result', 'x', 'y', 'a', 'b'}),
    ("exp(x) + log(y)", {'exp', 'log'}, {'x', 'y'}),
    ("a + b + c", {'a', 'b', 'c'}, set()),
    ("e + 1e5", {'e'}, set()),
    ("max(a, b) + min(c, d)", {'max', 'min'}, {'a', 'b', 'c', 'd'}),
]

# Test cases with potential_symbols
potential_cases = [
    ("a + b + c + d", {'a', 'c', 'e'}, {'a', 'c'}),
    ("e = 1e5 + x", {'e', 'x', 'y'}, {'e', 'x'}),
    ("x + y + z", {'x', 'y'}, {'x', 'y'}),
    ("a + b + c", {'x', 'y'}, set()),
    ("min(x, y)", {'min', 'x'}, {'min', 'x'}),
    ("min(x, y)", {'x'}, {'x'}),
    ("exp(e) + 1e5", {'exp', 'e', 'log'}, {'exp', 'e'}),
    ("1e5 + 2e-3", {'e'}, set()),
]

# Test cases with both potential_symbols and symbols_to_ignore
both_cases = [
    ("a + b + c + d", {'a', 'b', 'c'}, {'b'}, {'a', 'c'}),
    ("min(x, y) + max(a, b)", {'min', 'x', 'y'}, {'min'}, {'x', 'y'}),
    ("exp(e) + log(e)", {'exp', 'log', 'e'}, {'e'}, {'exp', 'log'}),
    ("x = 1e5 + y", {'x', 'y', 'z'}, {'x'}, {'y'}),
]


@pytest.mark.parametrize("code,expected", basic_cases)
def test_basic_cases(code, expected):
    assert dace.symbolic.symbols_in_code(code) == expected


@pytest.mark.parametrize("code,ignore,expected", ignore_cases)
def test_with_ignore(code, ignore, expected):
    assert dace.symbolic.symbols_in_code(code, symbols_to_ignore=ignore) == expected


@pytest.mark.parametrize("code,potential,expected", potential_cases)
def test_with_potential(code, potential, expected):
    assert dace.symbolic.symbols_in_code(code, potential_symbols=potential) == expected


@pytest.mark.parametrize("code,potential,ignore,expected", both_cases)
def test_with_potential_and_ignore(code, potential, ignore, expected):
    result = dace.symbolic.symbols_in_code(code, potential_symbols=potential, symbols_to_ignore=ignore)
    assert result == expected


def test_empty_potential_symbols():
    """Edge case: empty potential_symbols set should short-circuit."""
    result = dace.symbolic.symbols_in_code("x + y", potential_symbols=set())
    assert result == set()


def test_result_is_not_the_cached_token_set():
    """The tokenization is memoized, so a caller MUST NOT be able to reach the cache entry.

    ``symbols_in_code`` returns a set and several callers in the tree build on it in place
    (``found.update(...)``, ``referenced -= ...``). Handing back the memoized object would let one
    caller's edit rewrite what every later query sees -- a wrong-answer bug with no crash and no
    locality to the caller that caused it. Mutating the first result must leave the second alone.
    """
    code = "alpha + beta * gamma"
    first = dace.symbolic.symbols_in_code(code)
    first.add('injected')
    first.discard('alpha')
    assert dace.symbolic.symbols_in_code(code) == {'alpha', 'beta', 'gamma'}


def test_the_tokenizer_is_memoized_per_code_string():
    """Repeated queries about the same code must not re-tokenize it.

    This is the point of the cache: a pass asking "who reads symbol ``s``" walks every interstate
    condition and tasklet body once PER SYMBOL, and the strings do not change between those
    queries. Asserted on the cache counters, since the observable result is identical either way.
    """
    code = "delta * 3 + epsilon_v - 1e-5"
    dace.symbolic.symbols_in_code(code)  # prime, so the first call below is never the miss
    before = dace.symbolic.name_tokens_in_code.cache_info()
    for name in ('delta', 'epsilon_v', 'zeta'):
        dace.symbolic.symbols_in_code(code, potential_symbols={name})
    after = dace.symbolic.name_tokens_in_code.cache_info()
    assert after.misses == before.misses, 'a repeated query re-tokenized the same string'
    assert after.hits == before.hits + 3
