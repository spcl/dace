# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
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


def test_cpp_floor_of_fraction_difference_recombines_to_integer_division():
    """sympy normalises ``(LEN - 1) // 8`` on integer symbols to
    ``floor(LEN/8 - 1/8)``. The C++ printer must recombine the common-denominator
    sum and emit a single ``((LEN - 1) / (8))`` integer division -- not
    ``floor(LEN/8 - 1/8)`` (which in C++ collapses ``1/8`` to ``0`` and
    overshoots the loop bound) and not the literal ``LEN/8 - 1/8`` string."""
    from dace.symbolic import DaceSympyPrinter
    LEN = sympy.Symbol('LEN', integer=True)
    expr = (LEN - 1) // 8
    out = DaceSympyPrinter(arrays={}, cpp_mode=True).doprint(expr)

    clean = out.replace(' ', '')
    assert 'floor' not in clean, f'C++ printer must not emit floor(...); got {out!r}'
    assert '1/8' not in clean, f'literal Rational 1/8 leaked into C++ output: {out!r}'
    assert 'LEN-1' in clean, f'expected combined numerator (LEN - 1); got {out!r}'


@pytest.mark.parametrize(
    'value',
    [
        1.0 / 21.0,  # FFT ifft factor; needs 17 sig digits to round-trip
        0.1 + 0.2,
        1e-300,
        1e300,
        1.7976931348623157e308,  # near DBL_MAX
        1.0,
        5.0,
        3.14,
        -0.0476190476190476,
        1234567890.1234567,
    ])
def test_format_float_is_idempotent_under_parse_and_reformat(value):
    """The float-to-string serializer must be idempotent: ``f -> str -> f -> str``
    yields the same string as ``f -> str``. Otherwise SDFG save -> load -> save
    fails the round-trip equality check the framework runs on every serialization
    (e.g. ``tests/library/fft_test.py::test_ifft[backward]`` regressed because
    ``1/21`` was emitted as 17 digits in one save and 15 in the next)."""
    from dace.symbolic import _format_float
    s1 = _format_float(value)
    s2 = _format_float(float(s1))
    assert s1 == s2, f'_format_float not idempotent: {value!r} -> {s1!r} -> {s2!r}'
    assert float(s1) == float(value), (f'_format_float loses precision for {value!r}: parsed back as {float(s1)!r}')


@pytest.mark.parametrize('value', [1.0 / 21.0, 0.1 + 0.2, 1e-300, 1e300, 5.0, 3.14])
def test_serialize_symbolic_float_path_is_idempotent(value):
    """``serialize_symbolic`` dispatches on type: ``isinstance(expr, float)`` is a
    distinct branch from ``isinstance(expr, sympy.Basic)``. Both must produce the
    SAME 17-sig-digit shortest-round-trip form -- otherwise a SymbolicProperty
    that was set as a Python float gets a 15-digit string on save 1 (sympy's
    default sstr) and a 17-digit string on save 2 (DaceSympySerializer), breaking
    the SDFG save -> load -> save equality check (e.g. FFT/IFFT
    ``factor = 1/21`` regressed this way).
    """
    from dace.symbolic import serialize_symbolic, deserialize_symbolic
    s1 = serialize_symbolic(value)
    loaded = deserialize_symbolic(s1)
    s2 = serialize_symbolic(loaded)
    assert s1 == s2, (f'serialize_symbolic not idempotent across the float/sympy.Basic branches: '
                      f'save 1 (float)={s1!r}, save 2 (sympy.Float)={s2!r}')


@pytest.mark.parametrize("numerator, denominator", [("(N + 1) * 4", 8), ("i - 1", 2), ("2 * i + 3", 4)])
def test_int_floor_survives_codegen_where_floordiv_does_not(numerator, denominator):
    """Symbolic index/shape arithmetic must use ``int_floor``, never ``//``.

    ``expr // d`` builds ``sympy.floor(expr / d)``; sympy then distributes the division over the sum
    INSIDE the floor, and ``sym2cpp`` prints the argument without the floor. Each term is left to
    truncate on its own, so ``((N+1)*4)//8`` emits ``N/2 + 1/2`` -- at N=3 that is 1, not 2.
    """
    from dace.codegen.targets.cpp import sym2cpp
    from dace.symbolic import int_floor

    expr = pystr_to_symbolic(numerator)
    floored = sym2cpp(int_floor(expr, denominator))
    assert f"/ {denominator})" in floored, f"int_floor lost its divisor in codegen: {floored}"
    assert "1 / 2" not in floored, f"a rational leaked into an integer index: {floored}"

    for value in range(0, 8):
        substituted = {s: value for s in expr.free_symbols}
        expected = int(expr.subs(substituted)) // denominator
        actual = int(int_floor(expr, denominator).subs(substituted))
        assert actual == expected, f"int_floor({expr}, {denominator}) at {value}: {actual} != {expected}"


def test_floordiv_does_not_survive_codegen():
    """Pins WHY int_floor is mandatory: the ``//`` spelling loses its divisor on the way to C, so a
    well-meaning revert to ``//`` fails here rather than in a wrong number."""
    from dace.codegen.targets.cpp import sym2cpp
    from dace.symbolic import int_floor

    n = pystr_to_symbolic("N")
    assert "/ 8)" not in sym2cpp((n + 1) * 4 // 8)
    assert "/ 8)" in sym2cpp(int_floor((n + 1) * 4, 8))
