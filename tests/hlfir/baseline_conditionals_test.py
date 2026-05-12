"""Baseline HLFIR coverage — IF / ELSE, ``IF (cond) CYCLE`` inside a DO.
Pulled out of the original ``ported_from_f2dace_windmill_test.py``
per-feature split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def test_empty_if_else(tmp_path):
    src = """
subroutine pick(d, flag)
  implicit none
  logical, intent(in)    :: flag
  real(8), intent(inout) :: d(2)
  if (flag) then
    d(1) = 7.0d0
  else
    d(1) = -7.0d0
  end if
end subroutine pick
"""
    mod = f2py_compile(src, tmp_path / "ref", "pick")
    sdfg = _build(src, tmp_path / "sdfg", name="pick")

    d_ref = np.zeros(2, order="F")
    mod.pick(d_ref, True)
    d_sdfg = np.zeros(2, dtype=np.float64)
    # ``flag`` reads into a branch condition, so the classifier promotes
    # it to an SDFG symbol — pass a Python bool (DaCe casts it to int).
    sdfg(d=d_sdfg, flag=True)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_cond_array(tmp_path):
    """IF guard on a derived value (``s + 5.5 > 5.0``) drives an array
    write — exercises the boolean-expression path through tasklets."""
    src = """
subroutine cond_arr(d)
  implicit none
  real(4), intent(inout) :: d(5, 5)
  real(4) :: s
  s = d(2, 1) + 1.0
  if (s + 5.5 > 5.0) then
    d(2, 1) = 11.0
  else
    d(2, 1) = 12.0
  end if
end subroutine cond_arr
"""
    mod = f2py_compile(src, tmp_path / "ref", "cond_arr")
    sdfg = _build(src, tmp_path / "sdfg", name="cond_arr")

    d_ref = np.full((5, 5), 42.0, order="F", dtype=np.float32)
    mod.cond_arr(d_ref)
    d_sdfg = np.full((5, 5), 42.0, dtype=np.float32, order="F")
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_if_cycle(tmp_path):
    """``IF (i == 2) CYCLE`` inside a DO — branches over a write."""
    src = """
subroutine ifcyc(d)
  implicit none
  real(8), intent(inout) :: d(4)
  integer :: i
  do i = 1, 4
    if (i == 2) cycle
    d(i) = 5.5d0
  end do
end subroutine ifcyc
"""
    mod = f2py_compile(src, tmp_path / "ref", "ifcyc")
    sdfg = _build(src, tmp_path / "sdfg", name="ifcyc")

    d_ref = np.full(4, 42.0, order="F")
    mod.ifcyc(d_ref)
    d_sdfg = np.full(4, 42.0, dtype=np.float64)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)
