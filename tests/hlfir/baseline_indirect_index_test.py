"""Baseline HLFIR coverage  --  indirect / nested array indexing
(``out(i) = src(idx(i))``).  Pulled out of the original
``ported_from_f2dace_windmill_test.py`` per-feature split.
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


def test_nested_array_indirect(tmp_path):
    """An index array feeds another array read  --  classic indirect access."""
    src = """
subroutine nested_idx(out, idx, src, n)
  implicit none
  integer, intent(in)    :: n
  integer, intent(in)    :: idx(n)
  real(8), intent(in)    :: src(n)
  real(8), intent(inout) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = src(idx(i))
  end do
end subroutine nested_idx
"""
    mod = f2py_compile(src, tmp_path / "ref", "nested_idx")
    sdfg = _build(src, tmp_path / "sdfg", name="nested_idx")

    rng = np.random.default_rng(3)
    n = 7
    src_data = rng.standard_normal(n)
    idx = rng.integers(1, n + 1, size=n, dtype=np.int32)

    out_ref = np.zeros(n, order="F")
    mod.nested_idx(out_ref, np.asfortranarray(idx), np.asfortranarray(src_data))

    out_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(idx=np.ascontiguousarray(idx), src=np.ascontiguousarray(src_data), out=out_sdfg, n=n)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)
