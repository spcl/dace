"""Logical reductions  --  ``ANY`` / ``ALL`` / ``COUNT`` on whole arrays.

These are now in ``kRedTable`` and work for the whole-array shape
Flang emits for ``ANY(mask)`` / ``ALL(mask)`` / ``COUNT(mask)``
without any section slicing.  Reductions over dynamic sections
(``ANY(mask(lo:hi, jk))``) hit a separate gap in ``emit_reduce``
and stay xfailed via ``test_loopnest_6_sdfg_matches_f2py``.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_any_whole_array(tmp_path: Path):
    """``result = ANY(mask)`` on a 1D logical array."""
    src = """
subroutine kernel_any(mask, result, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: mask(n)
  logical, intent(out) :: result
  result = ANY(mask)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="kernel_any", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(42)
    # Fortran ``LOGICAL`` -> ``np.bool_`` on the SDFG signature.
    mask = np.asfortranarray(rng.random(8) > 0.5)
    result = np.zeros(1, dtype=np.bool_)
    sdfg(mask=mask, result=result, n=8)
    expected = bool(mask.any())
    assert bool(result[0]) == expected, f"ANY({mask.tolist()}) -> {bool(result[0])}, want {expected}"


def test_all_whole_array(tmp_path: Path):
    """``result = ALL(mask)`` on a 1D logical array."""
    src = """
subroutine kernel_all(mask, result, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: mask(n)
  logical, intent(out) :: result
  result = ALL(mask)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="kernel_all", pipeline="hlfir-propagate-shapes").build()

    # All-true case
    mask_all = np.asfortranarray(np.ones(5, dtype=np.bool_))
    res_all = np.zeros(1, dtype=np.bool_)
    sdfg(mask=mask_all, result=res_all, n=5)
    assert bool(res_all[0]) is True

    # One-false case
    mask_mixed = np.asfortranarray(np.array([True, True, False, True, True], dtype=np.bool_))
    res_mixed = np.zeros(1, dtype=np.bool_)
    sdfg(mask=mask_mixed, result=res_mixed, n=5)
    assert bool(res_mixed[0]) is False
