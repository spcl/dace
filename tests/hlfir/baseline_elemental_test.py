"""Baseline HLFIR coverage — ``ELEMENTAL`` subroutine called on arrays.
Pulled out of the original ``ported_from_f2dace_windmill_test.py``
per-feature split.

The SDFG is built from the ``ELEMENTAL`` form (the feature under test).
f2py can't compile a module-contained elemental (upstream bug in its
Fortran parser), so the reference uses an explicit per-element DO loop
that implements the same scalar body.  A numpy-only check would work
too, but going through gfortran gives us a real Fortran reference.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_elemental(tmp_path):
    sdfg_src = """
module elemod
  implicit none
contains
  elemental subroutine delta(od, scat_od, g)
    real(8), intent(inout) :: od, scat_od, g
    real(8) :: f
    f = g * g
    od = od - scat_od * f
    scat_od = scat_od * (1.0d0 - f)
    g = g / (1.0d0 + g)
  end subroutine delta
end module elemod

subroutine apply_delta(od, scat_od, g)
  use elemod
  implicit none
  real(8), intent(inout) :: od(14), scat_od(14), g(14)
  call delta(od, scat_od, g)
end subroutine apply_delta
"""
    ref_src = """
subroutine apply_delta(od, scat_od, g)
  implicit none
  real(8), intent(inout) :: od(14), scat_od(14), g(14)
  real(8) :: f
  integer :: i
  do i = 1, 14
     f           = g(i) * g(i)
     od(i)       = od(i) - scat_od(i) * f
     scat_od(i)  = scat_od(i) * (1.0d0 - f)
     g(i)        = g(i) / (1.0d0 + g(i))
  end do
end subroutine apply_delta
"""
    mod = f2py_compile(ref_src, tmp_path / "ref", "apply_delta")
    # ELEMENTAL lowering needs inline-all + fold-element-aliases +
    # symbol-dce, all in the default pipeline.  ``entry`` marks the
    # public module-scope ``delta`` private so its dummies don't leak
    # into extract_vars alongside ``apply_delta``'s own dummies.
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(sdfg_src, tmp_path / "sdfg", name="apply_delta", entry="_QPapply_delta").build()

    rng = np.random.default_rng(7)
    od = rng.standard_normal(14)
    s = rng.standard_normal(14)
    g = rng.standard_normal(14)

    od_ref, s_ref, g_ref = (np.asfortranarray(od.copy()), np.asfortranarray(s.copy()), np.asfortranarray(g.copy()))
    mod.apply_delta(od_ref, s_ref, g_ref)

    od_sdfg = np.ascontiguousarray(od.copy())
    s_sdfg = np.ascontiguousarray(s.copy())
    g_sdfg = np.ascontiguousarray(g.copy())
    sdfg(od=od_sdfg, scat_od=s_sdfg, g=g_sdfg)
    np.testing.assert_allclose(od_sdfg, od_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_sdfg, s_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_sdfg, g_ref, rtol=1e-12, atol=1e-12)
