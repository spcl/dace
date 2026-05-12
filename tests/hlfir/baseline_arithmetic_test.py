"""Baseline HLFIR coverage — straight-line scalar / array arithmetic.

Pulled out of the original ``ported_from_f2dace_windmill_test.py`` so
each feature lives in a focused file.  Every test here builds the SDFG
through the HLFIR frontend AND a gfortran-via-f2py reference from the
same source, then asserts numerical agreement on random inputs.
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


def test_elementwise_scalar_arithmetic(tmp_path):
    src = """
subroutine daxpy_lite(x, y, z, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: x(n), y(n)
  real(8), intent(inout) :: z(n)
  integer :: i
  do i = 1, n
    z(i) = 2.0d0 * x(i) + y(i) - 0.5d0 * x(i)
  end do
end subroutine daxpy_lite
"""
    mod = f2py_compile(src, tmp_path / "ref", "daxpy_lite")
    sdfg = _build(src, tmp_path / "sdfg", name="daxpy_lite")

    rng = np.random.default_rng(1)
    n = 16
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    z_ref = np.zeros(n, order="F")
    mod.daxpy_lite(np.asfortranarray(x), np.asfortranarray(y), z_ref)

    z_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(x=np.ascontiguousarray(x), y=np.ascontiguousarray(y), z=z_sdfg, n=n)
    np.testing.assert_allclose(z_sdfg, z_ref, rtol=1e-12, atol=1e-12)


def test_two_tasklets_raw(tmp_path):
    """Two statements in the same body, the second consuming the first —
    exercises the single-access-node-per-state invariant numerically."""
    src = """
subroutine raw_chain(a, out, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: out(n)
  real(8) :: tmp(n)
  integer :: i
  do i = 1, n
    tmp(i) = a(i) * 2.0d0
    out(i) = tmp(i) + 1.0d0
  end do
end subroutine raw_chain
"""
    mod = f2py_compile(src, tmp_path / "ref", "raw_chain")
    sdfg = _build(src, tmp_path / "sdfg", name="raw_chain")

    rng = np.random.default_rng(2)
    n = 8
    a = rng.standard_normal(n)
    out_ref = np.zeros(n, order="F")
    mod.raw_chain(np.asfortranarray(a), out_ref)
    out_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), out=out_sdfg, n=n)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)
