"""Multiple reductions writing to *different* elements of the same array.

Regression coverage for the bridge bug where ``buildReduceNode`` only
captured the destination's array name and dropped the
``hlfir.designate`` index, so two ``MINVAL``s in a row both emitted a
Reduce that wrote through the whole destination  --  last one won.

Each test wires ``res(i) = REDUCE(...)`` for two distinct ``i`` values
and checks both elements end up correctly populated.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_minval_two_targets(tmp_path: Path):
    """``res(1) = MINVAL(d); res(2) = MINVAL(d(:))``  --  both emit a
    whole-array Reduce; without the destination-index fix both Reduces
    write to ``res[0:2]`` and ``res[1]`` ends up at the identity (inf).
    """
    src = """
subroutine main(d, res)
  double precision, dimension(7), intent(in)  :: d
  double precision, dimension(2), intent(out) :: res
  res(1) = MINVAL(d)
  res(2) = MINVAL(d(:))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.array([3.0, 1.0, 5.0, 2.0, 4.0, 7.0, 6.0], dtype=np.float64)
    res = np.zeros(2, dtype=np.float64)
    sdfg(d=d, res=res)
    np.testing.assert_array_equal(res, [1.0, 1.0])


def test_sum_two_targets(tmp_path: Path):
    """``res(1) = SUM(d); res(2) = SUM(d(:))``  --  same shape, different
    LHS index.  SUM has wcr ``a+b`` and identity ``0``."""
    src = """
subroutine main(d, res)
  double precision, dimension(7), intent(in)  :: d
  double precision, dimension(2), intent(out) :: res
  res(1) = SUM(d)
  res(2) = SUM(d(:))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.array([3.0, 1.0, 5.0, 2.0, 4.0, 7.0, 6.0], dtype=np.float64)
    res = np.zeros(2, dtype=np.float64)
    sdfg(d=d, res=res)
    np.testing.assert_array_equal(res, [d.sum(), d.sum()])


def test_product_three_targets(tmp_path: Path):
    """Three distinct destination indices, three independent Reduces."""
    src = """
subroutine main(d, res)
  integer, dimension(6), intent(in)  :: d
  integer, dimension(3), intent(out) :: res
  res(1) = PRODUCT(d(1:2))
  res(2) = PRODUCT(d(3:4))
  res(3) = PRODUCT(d(5:6))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.array([2, 3, 4, 5, 6, 7], dtype=np.int32)
    res = np.zeros(3, dtype=np.int32)
    sdfg(d=d, res=res)
    np.testing.assert_array_equal(res, [d[0] * d[1], d[2] * d[3], d[4] * d[5]])


def test_mixed_reductions(tmp_path: Path):
    """Min and max into distinct elements  --  different wcr lambdas /
    identities, so any leakage between them shows up immediately."""
    src = """
subroutine main(d, res)
  double precision, dimension(7), intent(in)  :: d
  double precision, dimension(2), intent(out) :: res
  res(1) = MINVAL(d)
  res(2) = MAXVAL(d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.array([3.0, 1.0, 5.0, 2.0, 4.0, 7.0, 6.0], dtype=np.float64)
    res = np.zeros(2, dtype=np.float64)
    sdfg(d=d, res=res)
    np.testing.assert_array_equal(res, [d.min(), d.max()])
