"""End-to-end tests for the Fortran ``SIGN``, ``DIM``, and ``SHAPE``
intrinsics through the HLFIR frontend.

- ``SIGN(a, b)`` returns ``|a|`` with the sign of ``b``.  Float operands
  lower to ``math.copysign`` (added to the bridge's ``binary_math``
  table); integer operands lower to an ``arith.select`` predicate
  pattern handled by the generic ternary fallback.
- ``DIM(a, b)`` returns ``MAX(a - b, 0)``.  Lowered as ``arith.cmp* +
  arith.select``  --  same idiom as the existing min/max fallback.
- ``SHAPE(arr)`` returns a rank-1 integer array of the source's per-dim
  extents (clamped to ``>= 0``).  Lowered as per-element scalar
  assigns; existing assign machinery handles it.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# SIGN
# ---------------------------------------------------------------------------


def test_sign_float(tmp_path: Path):
    """``SIGN(a, b)`` on real(8)  --  ``math.copysign``."""
    src = """
subroutine main(a, b, out)
  real(8), intent(in)  :: a, b
  real(8), intent(out) :: out
  out = SIGN(a, b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    for a_in, b_in, expected in [(3.0, 5.0, 3.0), (3.0, -5.0, -3.0), (-3.0, 5.0, 3.0), (-3.0, -5.0, -3.0)]:
        out = np.zeros(1, dtype=np.float64)
        sdfg(a=a_in, b=b_in, out=out)
        assert float(out[0]) == expected, f"SIGN({a_in}, {b_in}) -> {out[0]}, want {expected}"


def test_sign_integer(tmp_path: Path):
    """``SIGN(a, b)`` on integer  --  predicate-driven select."""
    src = """
subroutine main(a, b, out)
  integer, intent(in)  :: a, b
  integer, intent(out) :: out
  out = SIGN(a, b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    for a_in, b_in, expected in [(7, 3, 7), (7, -3, -7), (-7, 3, 7), (-7, -3, -7)]:
        out = np.zeros(1, dtype=np.int32)
        sdfg(a=a_in, b=b_in, out=out)
        assert int(out[0]) == expected, f"SIGN({a_in}, {b_in}) -> {out[0]}, want {expected}"


# ---------------------------------------------------------------------------
# DIM
# ---------------------------------------------------------------------------


def test_dim_float(tmp_path: Path):
    """``DIM(a, b) = MAX(a - b, 0)`` on real(8)."""
    src = """
subroutine main(a, b, out)
  real(8), intent(in)  :: a, b
  real(8), intent(out) :: out
  out = DIM(a, b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    for a_in, b_in, expected in [(5.0, 3.0, 2.0), (3.0, 5.0, 0.0), (-1.0, -3.0, 2.0), (4.0, 4.0, 0.0)]:
        out = np.zeros(1, dtype=np.float64)
        sdfg(a=a_in, b=b_in, out=out)
        assert float(out[0]) == expected, f"DIM({a_in}, {b_in}) -> {out[0]}, want {expected}"


def test_dim_integer(tmp_path: Path):
    """``DIM(a, b)`` on integer."""
    src = """
subroutine main(a, b, out)
  integer, intent(in)  :: a, b
  integer, intent(out) :: out
  out = DIM(a, b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    for a_in, b_in, expected in [(8, 3, 5), (3, 8, 0), (-2, -7, 5), (4, 4, 0)]:
        out = np.zeros(1, dtype=np.int32)
        sdfg(a=a_in, b=b_in, out=out)
        assert int(out[0]) == expected, f"DIM({a_in}, {b_in}) -> {out[0]}, want {expected}"


# ---------------------------------------------------------------------------
# SHAPE
# ---------------------------------------------------------------------------


def test_shape_2d(tmp_path: Path):
    """``SHAPE(arr)`` returns a rank-1 integer array of dim extents."""
    src = """
subroutine main(arr, n, m, out)
  integer, intent(in)  :: n, m
  real(8), intent(in)  :: arr(n, m)
  integer, intent(out) :: out(2)
  out = SHAPE(arr)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(0)
    n, m = 4, 7
    arr = np.asfortranarray(rng.random((n, m)))
    out = np.zeros(2, dtype=np.int32)
    sdfg(arr=arr, n=n, m=m, out=out)
    np.testing.assert_array_equal(out, [n, m])


def test_shape_3d(tmp_path: Path):
    """``SHAPE(arr)`` on rank-3 array  --  output is rank-1 length 3."""
    src = """
subroutine main(arr, n, m, p, out)
  integer, intent(in)  :: n, m, p
  real(8), intent(in)  :: arr(n, m, p)
  integer, intent(out) :: out(3)
  out = SHAPE(arr)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(1)
    n, m, p = 3, 5, 2
    arr = np.asfortranarray(rng.random((n, m, p)))
    out = np.zeros(3, dtype=np.int32)
    sdfg(arr=arr, n=n, m=m, p=p, out=out)
    np.testing.assert_array_equal(out, [n, m, p])
