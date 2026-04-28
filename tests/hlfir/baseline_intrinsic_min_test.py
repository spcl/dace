"""Baseline HLFIR coverage — ``MIN`` intrinsic with mixed scalar /
array reads.  Pulled out of the original
``ported_from_f2dace_windmill_test.py`` per-feature split.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def test_min_intrinsic(tmp_path):
    src = """
subroutine min_res(d, res)
  implicit none
  real(4), intent(inout) :: d(2)
  real(4), intent(inout) :: res(2)
  real(4) :: temp
  temp = 88.0
  d(1) = d(1) * 2.0
  temp = min(d(1), temp)
  res(1) = temp + 10.0
end subroutine min_res
"""
    mod = f2py_compile(src, tmp_path / "ref", "min_res")
    sdfg = _build(src, tmp_path / "sdfg", name="min_res")

    d_ref = np.full(2, 42.0, order="F", dtype=np.float32)
    r_ref = np.full(2, 42.0, order="F", dtype=np.float32)
    mod.min_res(d_ref, r_ref)

    d_sdfg = np.full(2, 42.0, dtype=np.float32)
    r_sdfg = np.full(2, 42.0, dtype=np.float32)
    sdfg(d=d_sdfg, res=r_sdfg)
    np.testing.assert_allclose(r_sdfg, r_ref)
