#!/usr/bin/env python3
# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import numpy as np
import sympy

import dace
from dace import symbolic


def test_undefined_symbol_creation():
    # Test that UndefinedSymbol can be created
    us = symbolic.UndefinedSymbol()
    assert us.name == '?'
    assert isinstance(us, symbolic.symbol)
    assert isinstance(us, symbolic.UndefinedSymbol)


def test_undefined_symbol_operations():
    # Test that operations with UndefinedSymbol result in UndefinedSymbol
    us = symbolic.UndefinedSymbol()
    s = symbolic.symbol('N')

    # Test various operations
    assert isinstance(us + s, symbolic.UndefinedSymbol)
    assert isinstance(s + us, symbolic.UndefinedSymbol)
    assert isinstance(us - s, symbolic.UndefinedSymbol)
    assert isinstance(s - us, symbolic.UndefinedSymbol)
    assert isinstance(us * s, symbolic.UndefinedSymbol)
    assert isinstance(s * us, symbolic.UndefinedSymbol)
    assert isinstance(us / s, symbolic.UndefinedSymbol)
    assert isinstance(s / us, symbolic.UndefinedSymbol)

    # Test more complex expressions
    expr = (us * 2 + s) / 4
    assert isinstance(expr, symbolic.UndefinedSymbol)


def test_undefined_symbol_comparisons():
    # Test that comparisons with UndefinedSymbol using inequal_symbols return True
    us = symbolic.UndefinedSymbol()
    s = symbolic.symbol('N')

    # Test that symbolic.inequal_symbols handles UndefinedSymbol
    assert symbolic.inequal_symbols(us, s) is True
    assert symbolic.inequal_symbols(s, us) is True

    # Test that UndefinedSymbol in equality checks with function correctly
    expr_with_us = us * s + 5
    assert symbolic.issymbolic(expr_with_us)


def test_undefined_symbol_in_issymbolic():
    # Test that issymbolic recognizes UndefinedSymbol
    us = symbolic.UndefinedSymbol()
    s = symbolic.symbol('N')

    assert symbolic.issymbolic(us)
    assert symbolic.issymbolic(us + s)
    assert symbolic.issymbolic(s * us)


def test_undefined_symbol_in_evaluate():
    # Test that evaluate raises TypeError for expressions with UndefinedSymbol
    us = symbolic.UndefinedSymbol()
    s = symbolic.symbol('N')

    with pytest.raises(TypeError):
        symbolic.evaluate(us, {'N': 5})

    with pytest.raises(TypeError):
        symbolic.evaluate(us + s, {'N': 5})

    with pytest.raises(TypeError):
        symbolic.evaluate(sympy.sin(us), {us: 7})


def test_undefined_symbol_propagation():
    """Tests that UndefinedSymbol propagates through symbolic expressions."""

    # Create expressions with undefined symbols
    a = symbolic.symbol('a')
    b = symbolic.symbol('b')
    undefined = symbolic.UndefinedSymbol()

    # Operations directly on undefined should result in undefined
    expr1 = undefined + b
    expr2 = undefined * b
    expr3 = a * undefined
    expr4 = a / undefined

    assert isinstance(expr1, symbolic.UndefinedSymbol)
    assert isinstance(expr2, symbolic.UndefinedSymbol)
    assert isinstance(expr3, symbolic.UndefinedSymbol)
    assert isinstance(expr4, symbolic.UndefinedSymbol)

    # Test symbolic operations that involve UndefinedSymbol
    assert symbolic.issymbolic(expr1) is True
    assert symbolic.issymbolic(expr2) is True
    assert symbolic.inequal_symbols(expr1, expr2) is True


def test_undefined_symbol_math_functions():
    """Tests that math functions with UndefinedSymbol propagate undefined status."""

    from dace.symbolic import UndefinedSymbol, symbol, inequal_symbols, int_ceil
    import sympy

    us = UndefinedSymbol()
    s = symbol('N')

    # Test that any operation with UndefinedSymbol produces undefined results
    # These might not return UndefinedSymbol directly, but should be treated as undefined

    # Test inequal_symbols with expressions containing UndefinedSymbol
    assert inequal_symbols(sympy.Abs(us), s) is True
    assert inequal_symbols(s, sympy.sin(us + 5)) is True

    # Test internal UndefinedSymbol handling in int_ceil - should produce UndefinedSymbol
    result = int_ceil(us, 2)
    assert symbolic.is_undefined(result)

    # Test with sympy function application - should be treated as undefined
    assert symbolic.is_undefined(sympy.sin(us))
    with pytest.raises(TypeError):
        symbolic.evaluate(sympy.sin(us), {})


def test_is_undefined_function():
    """Test the is_undefined function works correctly."""
    us = symbolic.UndefinedSymbol()
    s = symbolic.symbol('N')

    # Test with UndefinedSymbol directly
    assert symbolic.is_undefined(us) is True

    # Test with regular symbol
    assert symbolic.is_undefined(s) is False

    # Test with constants
    assert symbolic.is_undefined(5) is False
    assert symbolic.is_undefined(5.5) is False

    # Test with expressions containing UndefinedSymbol
    assert symbolic.is_undefined(s + us) is True
    assert symbolic.is_undefined(s * us) is True
    assert symbolic.is_undefined(sympy.sin(us)) is True

    # Test with expressions not containing UndefinedSymbol
    assert symbolic.is_undefined(s + 5) is False
    assert symbolic.is_undefined(s * s) is False
    assert symbolic.is_undefined(sympy.sin(s)) is False

    # Test with string inputs
    assert symbolic.is_undefined("?") is True
    assert symbolic.is_undefined("N") is False


if __name__ == '__main__':
    test_undefined_symbol_creation()
    test_undefined_symbol_operations()
    test_undefined_symbol_comparisons()
    test_undefined_symbol_in_issymbolic()
    test_undefined_symbol_in_evaluate()
    test_undefined_symbol_propagation()
    test_undefined_symbol_math_functions()
    test_is_undefined_function()
