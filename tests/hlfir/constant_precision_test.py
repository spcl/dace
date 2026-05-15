"""Tests that the bridge stringifies float constants at IEEE-754 round-trip
precision (17 digits for binary64).

Fortran ``parameter`` constants and ``arith.constant`` literals are
folded by Flang into ``arith.constant`` ops; the bridge formats them as
strings for the tasklet code.  At default ``ostream`` precision (6
digits) the bottom of the mantissa is lost  --  multiplying a folded
``3.14159265358979d0`` by 2 produces ``6.28319`` instead of
``6.283185307179586`` and the result has effective f32 precision in a
nominally-f64 SDFG.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_parameter_keeps_f64_precision(tmp_path: Path):
    """Module ``parameter`` real(8) constant flows through Flang's
    constant folding into the tasklet expression.  The bridge must emit
    enough digits to round-trip  --  17 for double  --  so the tasklet's
    folded multiplication matches the f64 reference exactly."""
    src = """
module pi_consts
  implicit none
  real(8), parameter :: pi_val = 3.14159265358979323846d0
end module

subroutine main(out)
  use pi_consts, only: pi_val
  real(8), intent(out) :: out
  out = pi_val * 2.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    expected = 3.14159265358979323846 * 2.0
    # 1e-15 rtol: with 17-digit serialisation we round-trip binary64
    # exactly; with the old default 6-digit serialisation this would
    # only match at ~1e-6.
    np.testing.assert_allclose(out[0], expected, rtol=1e-15, atol=0)


def test_inline_double_literal_keeps_precision(tmp_path: Path):
    """Inline double-precision literal (``1.0d0 / 3.0d0`` style) folded
    by Flang at compile time into a single constant  --  same precision
    requirement, no module ``parameter`` involved."""
    src = """
subroutine main(out)
  real(8), intent(out) :: out
  out = 1.0d0 / 3.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    expected = 1.0 / 3.0
    # Folded constant should be exactly representable as the IEEE-754
    # binary64 of 1/3.  6-digit serialisation would give 0.333333,
    # different from the actual binary64 by ~1e-7.
    np.testing.assert_allclose(out[0], expected, rtol=1e-15, atol=0)


def test_compound_constant_expression_keeps_precision(tmp_path: Path):
    """Multiple folded constants in a single expression  --  each
    ``arith.constant`` carries its own FloatAttr through buildExpr; the
    serialisation must be precise for every one of them."""
    src = """
subroutine main(out)
  real(8), intent(out) :: out
  ! Folded by Flang: 1.234567890123456d0 * 2.0d0 + 0.987654321098765d0
  out = 1.234567890123456d0 * 2.0d0 + 0.987654321098765d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    expected = 1.234567890123456 * 2.0 + 0.987654321098765
    np.testing.assert_allclose(out[0], expected, rtol=1e-15, atol=0)


def _tasklet_code(sdfg) -> str:
    """Concatenate every tasklet's code string in ``sdfg``.

    :param sdfg: a built SDFG.
    :returns: all tasklet code joined by newlines.
    """
    import dace

    return "\n".join(
        str(n.code.as_string) for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.nodes.Tasklet))


_SRC_F32 = """
subroutine cst32(y)
  implicit none
  real(4), intent(out) :: y
  real(4) :: a
  a = 0.1
  y = a + 0.2
end subroutine cst32
"""

_SRC_F64 = """
subroutine cst64(y)
  implicit none
  real(8), intent(out) :: y
  real(8) :: a
  a = 0.1_8
  y = a + 0.2_8
end subroutine cst64
"""


def test_double_constant_stays_double(tmp_path: Path):
    """``real(8)`` literals: no ``float32`` cast, double value kept,
    output matches an f2py reference of the same source exactly."""
    d = tmp_path / "f64"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_F64, d, name="cst64", entry="_QPcst64").build()
    sdfg.validate()

    code = _tasklet_code(sdfg)
    assert "float32" not in code, f"unexpected fp32 cast in f64 kernel: {code}"
    assert sdfg.arrays["y"].dtype.type == np.float64

    ref = f2py_compile(_SRC_F64, d / "ref", "cst64_ref")
    y_ref = np.float64(ref.cst64())

    y = np.zeros(1, dtype=np.float64, order="F")
    sdfg(y=y)
    np.testing.assert_array_equal(np.float64(y[0]), y_ref)


def test_single_constant_emits_float32_cast(tmp_path: Path):
    """``real(4)`` literals: emitted wrapped in ``dace.float32(...)`` so
    they round to single precision exactly as gfortran does; SDFG
    result matches the f2py reference and differs from the naive
    double evaluation."""
    d = tmp_path / "f32"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_F32, d, name="cst32", entry="_QPcst32").build()
    sdfg.validate()

    code = _tasklet_code(sdfg)
    assert "dace.float32(" in code, (f"fp32 constant must be wrapped in dace.float32(...) for correct "
                                     f"rounding; tasklet code was: {code}")
    assert sdfg.arrays["y"].dtype.type == np.float32

    ref = f2py_compile(_SRC_F32, d / "ref", "cst32_ref")
    y_ref = np.float32(ref.cst32())

    y = np.zeros(1, dtype=np.float32, order="F")
    sdfg(y=y)
    np.testing.assert_array_equal(np.float32(y[0]), y_ref)
    # Genuinely the fp32 result, not the fp64 one.
    assert np.float32(y[0]) == np.float32(np.float32(0.1) + np.float32(0.2))
    assert float(y[0]) != (0.1 + 0.2)


def test_single_constant_uses_shortest_roundtrip_form(tmp_path: Path):
    """The f32 literal is stringified in its SHORTEST round-tripping
    form (``dace.float32(0.1)``), not the f64-widened expansion
    (``dace.float32(0.10000000149011612)``).  Both are bit-identical
    once cast, but the short form stays close to the Fortran source."""
    d = tmp_path / "f32short"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_F32, d, name="cst32s", entry="_QPcst32").build()
    sdfg.validate()

    code = _tasklet_code(sdfg)
    assert "dace.float32(0.1)" in code, (f"expected shortest-roundtrip f32 literal dace.float32(0.1); "
                                         f"got: {code}")
    assert "dace.float32(0.2)" in code, code
    # The long f64-widened form must NOT appear.
    assert "0.10000000149011612" not in code, (f"f32 constant widened to f64 17-digit form: {code}")
    assert "0.20000000298023224" not in code, code
