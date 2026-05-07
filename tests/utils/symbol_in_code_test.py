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
