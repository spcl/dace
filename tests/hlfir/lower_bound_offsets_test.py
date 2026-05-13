"""Per-axis offset symbols make non-1 Fortran lower bounds work uniformly.

The bridge declares ``offset_<arr>_d<i>`` symbols for every array (one
per dim) and emits memlets in the form ``A[(idx) - offset_A_d<i>]``.
``SDFGBuilder.build()`` calls ``sdfg.specialize`` at the end with the
known compile-time values (default ``1``; ``dimension(20:24)`` gets
``20``; etc.).  Sympy folds the substitution everywhere automatically.

Tests pin three Fortran shapes the design unblocks:

  * ``dimension(20:24)``   --  constant non-1 lower bound.
  * ``dimension(0:4)``     --  constant zero lower bound.
  * Whole-array section assignment with a non-1 lb on both sides  --
    section-parent + array offset interact through the symbol layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_constant_nonone_lower_bound(tmp_path: Path):
    """``real :: a(20:24)``  --  lbound 20.  Writing ``a(22) = 7`` must
    land on the third element (offset 2 from the start of storage)."""
    src = """
subroutine main(a)
  real(8), intent(inout) :: a(20:24)
  a(22) = 7.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.zeros(5, dtype=np.float64, order='F')
    sdfg(a=a)
    np.testing.assert_array_equal(a, [0.0, 0.0, 7.0, 0.0, 0.0])


def test_constant_zero_lower_bound(tmp_path: Path):
    """``real :: a(0:4)``  --  lbound 0.  Writing ``a(0) = 1`` and
    ``a(4) = 5`` lands on the first and last element of storage."""
    src = """
subroutine main(a)
  real(8), intent(inout) :: a(0:4)
  a(0) = 1.0d0
  a(4) = 5.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.zeros(5, dtype=np.float64, order='F')
    sdfg(a=a)
    np.testing.assert_array_equal(a, [1.0, 0.0, 0.0, 0.0, 5.0])


def test_section_assignment_with_default_lb(tmp_path: Path):
    """Section assign over default-lb arrays  --  ``res(2:4) = src(2:4)``
    exercises the section-parent's ``(lo - 1)`` contribution alongside
    the array's offset symbol."""
    src = """
subroutine main(src, res)
  real(8), intent(in)    :: src(5)
  real(8), intent(inout) :: res(5)
  res(2:4) = src(2:4) * 2.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    src = np.array([10., 20., 30., 40., 50.], dtype=np.float64, order='F')
    res = np.full(5, -1.0, dtype=np.float64, order='F')
    sdfg(src=src, res=res)
    # Only res[1..3] should change; res[0] and res[4] stay at -1.0.
    np.testing.assert_array_equal(res, [-1.0, 40.0, 60.0, 80.0, -1.0])
