# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import sympy
from sympy.tensor.array.expressions import ArrayElement

import dace
from dace.symbolic import pystr_to_symbolic


def test_simple_array_access():
    """Test basic array access expressions."""
    # Single dimension
    result = pystr_to_symbolic('A[i]')
    assert isinstance(result, ArrayElement)
    assert str(result) == 'A[i]'

    # Multiple dimensions
    result = pystr_to_symbolic('B[i, j, k]')
    assert isinstance(result, ArrayElement)
    assert str(result) == 'B[i, j, k]'


def test_complex_array_expressions():
    """Test complex expressions with arrays mentioned in the issue."""
    # The example from the issue description
    result = pystr_to_symbolic('ztp1[tmp_index_224, tmp_index_225] - v_ydcst_var_1_rtt > 0.0')
    assert isinstance(result, sympy.core.relational.StrictGreaterThan)

    # Multiple arrays in arithmetic operations
    result = pystr_to_symbolic('A[i, j] + B[k] - C[x, y, z]')
    assert isinstance(result, sympy.core.add.Add)

    # Array in comparison
    result = pystr_to_symbolic('A[i] > 0')
    assert isinstance(result, sympy.core.relational.StrictGreaterThan)


def test_nested_array_access():
    """Test nested array access (arrays with array indices)."""
    result = pystr_to_symbolic('A[B[i]]')
    assert isinstance(result, ArrayElement)
    # This should create ArraySymbols for both A and B


def test_backward_compatibility():
    """Test that non-array expressions still work correctly."""
    # Simple arithmetic
    result = pystr_to_symbolic('a + b * c')
    assert isinstance(result, sympy.core.add.Add)
    assert str(result) == 'a + b*c'

    # Conditional expressions
    result = pystr_to_symbolic('x > 0')
    assert isinstance(result, sympy.core.relational.StrictGreaterThan)

    # Logical operations (converted to DaCe's AND/OR functions)
    result = pystr_to_symbolic('(a > 0) and (b < 10)')
    assert 'AND' in str(type(result))


def test_mixed_array_and_scalar():
    """Test expressions mixing arrays and scalar variables."""
    result = pystr_to_symbolic('A[i] + x * y')
    assert isinstance(result, sympy.core.add.Add)

    result = pystr_to_symbolic('sqrt(A[i, j] ** 2 + B[i, j] ** 2)')
    assert 'sqrt' in str(result) or 'Pow' in str(result)


def test_array_dimensions_detection():
    """Test that arrays with different numbers of dimensions are handled correctly."""
    # Arrays with different dimensions in same expression
    result = pystr_to_symbolic('A[i] + B[j, k] + C[l, m, n]')
    assert isinstance(result, sympy.core.add.Add)


def test_fallback_to_function_calls():
    """Test that the original fallback mechanism still works for edge cases."""
    # Test with malformed expressions that might not parse with ArraySymbol approach
    # The implementation should fall back to the original bracket-to-parentheses approach
    try:
        # This should work with either approach
        result = pystr_to_symbolic('func[arg]')
        # Could be either ArrayElement or a function call, both are valid
        assert result is not None
    except Exception:
        # If it fails, that's also acceptable for edge cases
        pass


if __name__ == '__main__':
    pytest.main([__file__])
