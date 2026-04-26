"""Simple FaCe-native tests for Fortran bitwise intrinsics.

Flang lowers each of these to a small ``arith.*`` op tree on the integer
operand types:

- ``IBSET(x, p)``  → ``x | (1 << p)``         (set bit p).
- ``IBCLR(x, p)``  → ``x & ~(1 << p)``         (clear bit p; the ``~``
  comes out as ``arith.xori a, -1``).
- ``IEOR(a, b)``   → ``a ^ b``                 (bitwise XOR).
- ``ISHFT(x, n)``  → ``x << n`` for ``n>0``,
                     ``x >> -n`` for ``n<0``   (Flang inlines via shli /
                                                shrsi + a sign select).
- ``IAND(a, b)``   → ``a & b``.
- ``IOR(a, b)``    → ``a | b``.
- ``BTEST(x, p)``  → ``(x >> p) & 1`` (returned as i1 / .true.).
- ``IBITS(x, p, n)`` → ``(x >> p) & ((1 << n) - 1)``.

The bridge's ``buildExpr`` recognises the underlying ``arith.shli`` /
``arith.shrsi`` / ``arith.andi`` / ``arith.ori`` / ``arith.xori`` ops on
non-i1 operands; the test below verifies the full chain end-to-end.
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


def test_bitwise_set_clear_xor_shift_and(tmp_path: Path):
    src = """
subroutine probe(x, y, out)
  integer, intent(in)  :: x, y
  integer, intent(out) :: out(6)
  out(1) = ibset(x, 2)
  out(2) = ibclr(x, 2)
  out(3) = ieor(x, y)
  out(4) = ishft(x, 3)
  out(5) = iand(x, y)
  out(6) = ior(x, y)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    x_in, y_in = 0b1010, 0b1100
    x = np.array([x_in], dtype=np.int32)
    y = np.array([y_in], dtype=np.int32)
    out = np.zeros(6, dtype=np.int32)
    sdfg(x=x, y=y, out=out)
    assert int(out[0]) == x_in | (1 << 2)  # ibset
    assert int(out[1]) == x_in & ~(1 << 2)  # ibclr
    assert int(out[2]) == x_in ^ y_in  # ieor
    assert int(out[3]) == x_in << 3  # ishft (positive)
    assert int(out[4]) == x_in & y_in  # iand
    assert int(out[5]) == x_in | y_in  # ior


def test_bit_query_and_extract(tmp_path: Path):
    """``btest(x, p)`` and ``ibits(x, p, n)`` — bit query + slice extract."""
    src = """
subroutine probe(x, out_ibits, out_btest)
  integer, intent(in)  :: x
  integer, intent(out) :: out_ibits, out_btest
  out_ibits = ibits(x, 2, 3)
  if (btest(x, 1)) then
    out_btest = 1
  else
    out_btest = 0
  end if
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    x_in = 0b101110
    x = np.array([x_in], dtype=np.int32)
    out_ibits = np.zeros(1, dtype=np.int32)
    out_btest = np.zeros(1, dtype=np.int32)
    sdfg(x=x, out_ibits=out_ibits, out_btest=out_btest)
    assert int(out_ibits[0]) == (x_in >> 2) & ((1 << 3) - 1)
    assert int(out_btest[0]) == 1 if (x_in & (1 << 1)) else 0
