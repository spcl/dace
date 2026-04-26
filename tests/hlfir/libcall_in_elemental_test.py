"""Bridge support for libcall results consumed by an enclosing
``hlfir.elemental`` body — the ``2.0 - matmul(a, b)`` /
``1.0 - transpose(a)`` pattern.

Flang's HLFIR lowers ``res = expr_with_libcall(args)`` to:

  * an ``hlfir.matmul`` / ``hlfir.transpose`` / ``hlfir.dot_product``
    op producing an ``hlfir.expr`` *value* (no memory backing),
  * an ``hlfir.elemental`` whose body reads that value via
    ``hlfir.apply`` and weaves the surrounding arithmetic
    (``2.0 - apply(matmul, i, j)``),
  * an ``hlfir.assign`` of the elemental result to the destination.

Without bridge support, ``buildExpr`` returns ``?`` for the apply, the
tasklet body becomes ``2 - ?`` which can't be parsed.  The bridge now
materialises each libcall expr-producer into a synthetic
``_libtmp_<gid>`` transient (declared with Fortran column-major
strides), emits the libcall to write that transient, and rewrites the
apply as a regular array read of it.

These tests are deliberately the smallest possible programs that
exercise that path so a regression here points at the bridge logic
rather than at any other surrounding feature.
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


def test_one_minus_transpose(tmp_path: Path):
    """``res = 1.0 - transpose(a)`` — exercise the libcall-in-elemental
    materialisation for ``hlfir.transpose``."""
    src = """
subroutine main(a, res)
  double precision, dimension(5,4) :: a
  double precision, dimension(4,5) :: res
  res = 1.0 - transpose(a)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.empty((5, 4), order="F", dtype=np.float64)
    a[:] = np.arange(20).reshape(5, 4)
    res = np.zeros((4, 5), order="F", dtype=np.float64)
    sdfg(a=a, res=res)
    np.testing.assert_array_equal(res, 1.0 - a.T)


def test_two_minus_matmul(tmp_path: Path):
    """``res = 2.0 - matmul(a, b)`` — same pattern, ``hlfir.matmul``."""
    src = """
subroutine main(a, b, res)
  double precision, dimension(5,3) :: a
  double precision, dimension(3,7) :: b
  double precision, dimension(5,7) :: res
  res = 2.0 - matmul(a, b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.empty((5, 3), order="F", dtype=np.float64)
    a[:] = np.arange(15).reshape(5, 3)
    b = np.empty((3, 7), order="F", dtype=np.float64)
    b[:] = np.arange(21).reshape(3, 7)
    res = np.zeros((5, 7), order="F", dtype=np.float64)
    sdfg(a=a, b=b, res=res)
    np.testing.assert_array_equal(res, 2.0 - a @ b)
