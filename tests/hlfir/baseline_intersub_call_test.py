"""Baseline HLFIR coverage  --  inter-subroutine calls (caller->callee
inlining) and ``OPTIONAL`` scalar dummies with ``PRESENT()`` companion.
Pulled out of the original ``ported_from_f2dace_windmill_test.py``
per-feature split.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def test_intersub_call(tmp_path):
    """Two subroutines in one file, outer calls inner  --  exercises
    ``hlfir-inline-all`` writeback.  f2py wraps every subroutine in
    the file; we pick ``outer``."""
    src = """
subroutine inner(d)
  implicit none
  real(8), intent(inout) :: d(4)
  d(2) = 4.2d0
end subroutine inner

subroutine outer(d)
  implicit none
  real(8), intent(inout) :: d(4)
  d(2) = 5.5d0
  call inner(d)
end subroutine outer
"""
    mod = f2py_compile(src, tmp_path / "ref", "outer_mod")
    sdfg = _build(src, tmp_path / "sdfg", name="outer")

    d_ref = np.zeros(4, order="F")
    mod.outer(d_ref)
    d_sdfg = np.zeros(4, dtype=np.float64)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_optional_arg(tmp_path):
    """``PRESENT()`` on a scalar OPTIONAL resolves to a companion
    ``<name>_present`` symbol on the SDFG ABI.  Caller passes the
    flag alongside the dummy: non-zero = present, zero = absent.
    Covers both branches of the Fortran ``if present()``."""
    src = """
subroutine opt_sum(res, a)
  implicit none
  integer, intent(inout) :: res(2)
  integer, optional      :: a
  if (present(a)) then
    res(1) = a
  else
    res(1) = 0
  end if
end subroutine opt_sum
"""
    mod = f2py_compile(src, tmp_path / "ref", "opt_sum")
    sdfg = _build(src, tmp_path / "sdfg", name="opt_sum")

    # Present branch: caller supplies a and sets a_present=1.  Per
    # the Scalar I/O convention an OPTIONAL scalar dummy lands as a
    # plain Scalar on the SDFG signature.
    r_ref = np.zeros(2, order="F", dtype=np.int32)
    mod.opt_sum(r_ref, 5)
    r_sdfg = np.zeros(2, dtype=np.int32)
    sdfg(res=r_sdfg, a=5, a_present=1)
    np.testing.assert_array_equal(r_sdfg, r_ref)

    # Absent branch: reference call omits the argument entirely; SDFG
    # caller passes any placeholder value for ``a`` and sets the flag
    # to zero.  Fortran guarantees the callee doesn't read ``a`` in
    # that branch, so the placeholder's value is immaterial.
    r_ref_absent = np.zeros(2, order="F", dtype=np.int32)
    mod.opt_sum(r_ref_absent)
    r_sdfg_absent = np.zeros(2, dtype=np.int32)
    sdfg(res=r_sdfg_absent, a=0, a_present=0)
    np.testing.assert_array_equal(r_sdfg_absent, r_ref_absent)
