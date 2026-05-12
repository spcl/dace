"""Tests that the bridge stringifies float constants at IEEE-754 round-trip
precision (17 digits for binary64).

Fortran ``parameter`` constants and ``arith.constant`` literals are
folded by Flang into ``arith.constant`` ops; the bridge formats them as
strings for the tasklet code.  At default ``ostream`` precision (6
digits) the bottom of the mantissa is lost — multiplying a folded
``3.14159265358979d0`` by 2 produces ``6.28319`` instead of
``6.283185307179586`` and the result has effective f32 precision in a
nominally-f64 SDFG.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_parameter_keeps_f64_precision(tmp_path: Path):
    """Module ``parameter`` real(8) constant flows through Flang's
    constant folding into the tasklet expression.  The bridge must emit
    enough digits to round-trip — 17 for double — so the tasklet's
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
    by Flang at compile time into a single constant — same precision
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
    """Multiple folded constants in a single expression — each
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
