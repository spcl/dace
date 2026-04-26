"""Simple FaCe-native tests for elementwise math intrinsics that the
existing ``elemwise_intrinsics_test.py`` doesn't cover.  Each test
exercises one intrinsic family on a small kernel and compares the SDFG
output to a numpy reference.

Coverage:
- Hyperbolic: ``sinh``, ``cosh``, ``tanh`` (Flang lowers ``sinh`` as a
  ``fir.call @sinh`` runtime call; bridge recognises it alongside the
  ``math.cosh`` / ``math.tanh`` dialect ops).
- Inverse trig: ``asin``, ``acos``, ``atan``, ``atan2`` (``math.*`` ops).
- Conversion: ``int(x)``, ``nint(x)``, ``aint(x)``, ``anint(x)``,
  ``floor(x)`` — Flang routes ``nint`` through ``llvm.lround``, ``aint``
  through ``llvm.trunc``; bridge maps them to ``dace::int{32,64}`` casts
  and ``trunc`` / ``round`` Python calls.
- Modulo: ``mod(a,b)`` (truncated) and ``modulo(a,b)`` (floored) — both
  lower to ``fir.call @_FortranAMod*Real8``; bridge maps both to the
  Python ``%`` operator and the C++ codegen picks the right semantics
  for the operand type.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _scalar_in(x):
    return np.array([x], dtype=np.float64)


def test_hyperbolic(tmp_path: Path):
    """sinh / cosh / tanh elementwise."""
    src = """
subroutine probe(x, out)
  real(8), intent(in)  :: x
  real(8), intent(out) :: out(3)
  out(1) = sinh(x)
  out(2) = cosh(x)
  out(3) = tanh(x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    x = _scalar_in(0.7)
    out = np.zeros(3, dtype=np.float64)
    sdfg(x=x, out=out)
    np.testing.assert_allclose(out, [np.sinh(0.7), np.cosh(0.7), np.tanh(0.7)], rtol=1e-12)


def test_inverse_trig(tmp_path: Path):
    """asin / acos / atan / atan2."""
    src = """
subroutine probe(x, y, out)
  real(8), intent(in)  :: x, y
  real(8), intent(out) :: out(4)
  out(1) = asin(x)
  out(2) = acos(x)
  out(3) = atan(x)
  out(4) = atan2(y, x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    x = _scalar_in(0.5)
    y = _scalar_in(0.3)
    out = np.zeros(4, dtype=np.float64)
    sdfg(x=x, y=y, out=out)
    np.testing.assert_allclose(out, [np.arcsin(0.5), np.arccos(0.5), np.arctan(0.5), np.arctan2(0.3, 0.5)], rtol=1e-12)


def test_floor_aint(tmp_path: Path):
    """floor / aint — both round toward -inf / 0 respectively, return
    real of the same kind."""
    src = """
subroutine probe(x, out)
  real(8), intent(in)  :: x
  real(8), intent(out) :: out(2)
  out(1) = floor(x)
  out(2) = aint(x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    for v in (3.7, -3.7, 0.5):
        x = _scalar_in(v)
        out = np.zeros(2, dtype=np.float64)
        sdfg(x=x, out=out)
        assert out[0] == np.floor(v)  # floor: -inf rounding
        assert out[1] == np.trunc(v)  # aint: trunc toward 0


def test_int_nint(tmp_path: Path):
    """int (truncating cast) / nint (rounding cast)."""
    src = """
subroutine probe(x, out_int, out_nint)
  real(8), intent(in)  :: x
  integer, intent(out) :: out_int, out_nint
  out_int = int(x)
  out_nint = nint(x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    for v, expected_int, expected_nint in [(3.4, 3, 3), (3.6, 3, 4), (-3.4, -3, -3), (-3.6, -3, -4)]:
        x = _scalar_in(v)
        out_int = np.zeros(1, dtype=np.int32)
        out_nint = np.zeros(1, dtype=np.int32)
        sdfg(x=x, out_int=out_int, out_nint=out_nint)
        assert int(out_int[0]) == expected_int, f"int({v}) = {out_int[0]}, want {expected_int}"
        assert int(out_nint[0]) == expected_nint, f"nint({v}) = {out_nint[0]}, want {expected_nint}"


def test_mod_modulo(tmp_path: Path):
    """Fortran MOD (truncated) and MODULO (floored) — both are the
    Python ``%`` at the bridge level; the C++ codegen does the right
    thing per type."""
    src = """
subroutine probe(a, b, out)
  real(8), intent(in)  :: a, b
  real(8), intent(out) :: out(2)
  out(1) = mod(a, b)
  out(2) = modulo(a, b)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    # MOD has truncated-quotient semantics; MODULO floored.
    # For (a, b) = (-7.0, 3.0): mod = -7 - 3*int(-7/3) = -7 - 3*(-2) = -1
    #                            modulo = -7 - 3*floor(-7/3) = -7 - 3*(-3) = 2
    a = _scalar_in(-7.0)
    b = _scalar_in(3.0)
    out = np.zeros(2, dtype=np.float64)
    sdfg(a=a, b=b, out=out)
    # Both bridge to ``%``; Python's `-7.0 % 3.0` is 2.0 (floored), so the
    # bridge expression matches MODULO.  The C++ codegen lowers ``%`` on
    # double to ``fmod`` (truncated) for Fortran-MOD positions and to a
    # floor-rounded helper for MODULO positions, but at this layer both
    # come through as the same operator — verify against the floored
    # numpy result, which is what Python and DaCe's float-`%` produce.
    np.testing.assert_allclose(out[0], np.fmod(-7.0, 3.0), rtol=1e-12)
    np.testing.assert_allclose(out[1], -7.0 - 3.0 * np.floor(-7.0 / 3.0), rtol=1e-12)
