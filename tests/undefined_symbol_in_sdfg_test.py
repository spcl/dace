#!/usr/bin/env python3
# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import numpy as np

import dace
from dace import symbolic


def test_undefined_symbol_in_sdfg():
    """Tests that an UndefinedSymbol can be used in an SDFG but raises an error when compiled."""
    
    @dace.program
    def program_with_undefined_symbol(A: dace.float64[20, 20], B: dace.float64[20]):
        # Create a transient with an undefined dimension
        undefined_dim = symbolic.UndefinedSymbol()
        tmp = dace.define_local([20, undefined_dim], dace.float64)
        for i in range(20):
            for j in range(20):
                tmp[i, 0] = A[i, j]
            B[i] = tmp[i, 0]
    
    sdfg = program_with_undefined_symbol.to_sdfg()
    
    # Validate SDFG - it should pass validation at this point
    sdfg.validate()
    
    # Try to generate code - this should fail
    with pytest.raises(TypeError, match="undefined symbol"):
        sdfg.compile()


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


if __name__ == '__main__':
    test_undefined_symbol_propagation()
    # test_undefined_symbol_in_sdfg() - This would fail as expected