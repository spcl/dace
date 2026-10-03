# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import sympy
from dace.symbolic import sympy_numeric_fix, pystr_to_symbolic, symstr


def test_float_zero_stays_float():
    """sympy.Float(0.0) must not be demoted to int(0)."""
    result = sympy_numeric_fix(sympy.Float(0.0))
    assert isinstance(result, sympy.Float), \
        f"Float(0.0) demoted to {type(result).__name__}"
    assert float(result) == 0.0


def test_float_one_stays_float():
    """sympy.Float(1.0) must not be demoted to int(1)."""
    result = sympy_numeric_fix(sympy.Float(1.0))
    assert isinstance(result, sympy.Float)
    assert float(result) == 1.0


def test_float_five_stays_float():
    """sympy.Float(5.0) must not be demoted to int(5)."""
    result = sympy_numeric_fix(sympy.Float(5.0))
    assert isinstance(result, sympy.Float)
    assert float(result) == 5.0


def test_fractional_float_preserved():
    """sympy.Float(0.7) must stay Float(0.7)."""
    result = sympy_numeric_fix(sympy.Float(0.7))
    assert isinstance(result, sympy.Float)
    assert abs(float(result) - 0.7) < 1e-15


def test_float_prints_clean():
    """5.0 should print as '5.0', not '5.00000000000000'."""
    result = sympy_numeric_fix(sympy.Float(5.0))
    s = symstr(result)
    assert s == '5.0', f"Expected '5.0', got '{s}'"


def test_huge_python_int_becomes_oo():
    """Python int beyond float64 range must map to sympy.oo.
    Original comment: int(1.8e308) == expr is True because Python
    has variable-bit integers, but numpy.float64() overflows."""
    result = sympy_numeric_fix(10**309)
    assert result == sympy.oo


def test_huge_negative_python_int_becomes_neg_oo():
    """Negative Python int beyond float64 range must map to -sympy.oo."""
    result = sympy_numeric_fix(-(10**309))
    assert result == -sympy.oo


def test_max_float_literal_roundtrip():
    """Parsing 'max(a, 0.0)' and printing via symstr must preserve '0.0',
    not demote to '0'. Demotion causes Max<int>(a, 0) in C++ codegen."""
    expr = pystr_to_symbolic("max(a, 0.0)")
    result = symstr(expr, cpp_mode=True)

    # Must contain 0.0 (the float literal), not bare 0 (int literal)
    assert "0.0" in result, f"Float literal 0.0 was demoted to int: '{result}'. "


def test_max_float_literal_not_int():
    """Complement: ensure the printed string does NOT match 'Max(a, 0)'
    where 0 is an integer literal (no decimal point)."""
    expr = pystr_to_symbolic("max(a, 0.0)")
    result = symstr(expr, cpp_mode=True)

    # Strip spaces for robust matching
    clean = result.replace(" ", "")
    # Should not end with ,0)
    assert not clean.endswith(",0)"), f"Got integer literal in Max call: '{result}'"


def test_max_int_literal_stays_int():
    """Parsing 'max(a, 0)' with an explicit int should keep it as int.
    This is the correct behavior when the user wrote 0, not 0.0."""
    expr = pystr_to_symbolic("max(a, 0)")
    result = symstr(expr, cpp_mode=True)

    # This one SHOULD have bare 0, not 0.0
    clean = result.replace(" ", "")
    assert "0.0" not in clean, f"Integer literal 0 was promoted to float: '{result}'"
