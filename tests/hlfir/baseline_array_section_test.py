"""Baseline HLFIR coverage — array-section assignment ``res(a:b) = 42``.
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


def test_array_section_assign(tmp_path):
    """``res(a:b) = 42`` — section assign with symbolic bounds."""
    src = """
subroutine fill_range(res, a, b)
  implicit none
  integer, intent(in)    :: a, b
  integer, intent(inout) :: res(6)
  res(a:b) = 42
end subroutine fill_range
"""
    mod = f2py_compile(src, tmp_path / "ref", "fill_range")
    sdfg = _build(src, tmp_path / "sdfg", name="fill_range")

    r_ref = np.zeros(6, order="F", dtype=np.int32)
    mod.fill_range(r_ref, 2, 5)
    r_sdfg = np.zeros(6, dtype=np.int32)
    sdfg(res=r_sdfg, a=2, b=5)
    np.testing.assert_array_equal(r_sdfg, r_ref)
