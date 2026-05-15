"""Verbatim port of f2dace/dev:tests/fortran/fortran_language_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_real_kind_selector(tmp_path):
    src = """
subroutine main(d)
  implicit none
  integer, parameter :: JPRB = selected_real_kind(13, 300)
  integer, parameter :: JPIM = selected_int_kind(9)
  real(KIND=JPRB) d(4)
  integer(KIND=JPIM) i

  i = 7
  d(2) = 5.5 + i
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0)
    assert (a[0] == 42)
    assert (a[1] == 12.5)
    assert (a[2] == 42)


def test_fortran_frontend_if1(tmp_path):
    src = """
subroutine main(d)
  implicit none
  double precision d(3, 4, 5), ZFAC(10)
  integer JK, JL, RTT, NSSOPT
  integer ZTP1(10, 10)
  JL = 1
  JK = 1
  ZTP1(JL, JK) = 1.0
  RTT = 2
  NSSOPT = 1

  if (ZTP1(JL, JK) >= RTT .or. NSSOPT == 0) then
    ZFAC(1) = 1.0
  else
    ZFAC(1) = 2.0
  end if
  d(1, 1, 1) = ZFAC(1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d, jk=0, jl=0, rtt=0, nssopt=0)
    assert (d[0, 0, 0] == 2)


def test_fortran_frontend_loop1(tmp_path):
    src = """
subroutine main(d)
  logical d(3, 4, 5), ZFAC(10)
  integer :: a, JK, JL, JM
  integer, parameter :: KLEV = 10, N = 10, NCLV = 3

  double precision :: RLMIN, ZVQX(NCLV)
  logical :: LLCOOLJ, LLFALL(NCLV)
  LLFALL(:) = .false.
  ZVQX(:) = 0.d0
  ZVQX(2) = 1.d0
  do JM = 1, NCLV
    if (ZVQX(JM) > 0.d0) LLFALL(JM) = .true. ! falling species
  end do

  d(1, 1, 1) = LLFALL(1)
  d(1, 1, 2) = LLFALL(2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    # ``d`` is Fortran ``LOGICAL`` -- pass ``np.bool_`` (1 byte / elem)
    # so the C ABI dtype matches the SDFG's ``bool *`` declaration.
    # Initial fill of ``True`` so the assertions can distinguish
    # SDFG writes (False / True) from the initial value.
    d = np.full([3, 4, 5], True, order="F", dtype=np.bool_)
    sdfg(d=d, a=0, jk=0, jl=0, jm=0)
    # LLFALL(1) == .false. (no positive ZVQX entry at JM=1) ->
    # ``d(1,1,1) = .false.``; LLFALL(2) == .true. (ZVQX(2)=1.0).
    assert d[0, 0, 0] == False
    assert bool(d[0, 0, 1])


def test_fortran_frontend_function_statement(tmp_path):
    # All literals carry the ``d0`` suffix so the source is fp64 end-to-end.
    # A bare ``5.1`` is fp32 default-real; widening to fp64 yields
    # ``5.099999904632568``, which would break the exact-equality
    # assertion against Python's fp64 ``5.1`` below.  The bridge promotes
    # constants to fp64 by design.
    src = """
subroutine main(d)
  double precision d(3, 4, 5)
  double precision :: PTARE, RTT(2), FOEDELTA, FOELDCP
  double precision :: RALVDCP(2), RALSDCP(2), RES

  FOEDELTA(PTARE) = max(0.d0, sign(1.d0, PTARE - RTT(1)))
  FOELDCP(PTARE) = FOEDELTA(PTARE)*RALVDCP(1) + (1.d0 - FOEDELTA(PTARE))*RALSDCP(1)

  RTT(1) = 4.5d0
  RALVDCP(1) = 4.9d0
  RALSDCP(1) = 5.1d0
  d(1, 1, 1) = FOELDCP(3.d0)
  RES = FOELDCP(3.d0)
  d(1, 1, 2) = RES
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 5.1)
    assert (d[0, 0, 1] == 5.1)


def test_internal_subprograms(tmp_path):
    src = """
module lib
contains
  real function fn2()
    fn2 = 2.
  contains
    subroutine subr
    end subroutine subr
  end function fn2
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: f(5)
  real, intent(out) :: d(1)
  call fn(f, d(1))
contains
  subroutine fn(f, d)
    real, intent(inout) :: f(:)
    real, intent(out) :: d
    f(1) = 7
    d = fn2()
  end subroutine fn
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.full([1], 42, order="F", dtype=np.float32)
    sdfg(d=d)
    assert np.allclose(d, [2])


def test_fortran_frontend_pow1(tmp_path):
    src = """
subroutine main(d)
  implicit none
  double precision d(3, 4, 5)
  double precision :: ZSIGK(2), ZHRC(2), RAMID(2)

  ZSIGK(1) = 4.8
  RAMID(1) = 0.0
  ZHRC(1) = 12.34
  if (ZSIGK(1) > 0.8) then
    ZHRC(1) = RAMID(1) + (1.0 - RAMID(1))*((ZSIGK(1) - 0.8)/0.2)**2
  end if
  d(1, 1, 2) = ZHRC(1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "pow1_ref")
    d_ref = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    mod.main(d_ref)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='main').build()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    np.testing.assert_allclose(d, d_ref, rtol=1e-12, atol=1e-12)


def test_fortran_frontend_pow2(tmp_path):
    src = """
subroutine main(d)
  implicit none
  double precision d(3, 4, 5)
  double precision :: ZSIGK(2), ZHRC(2), RAMID(2)

  ZSIGK(1) = 4.8
  RAMID(1) = 0.0
  ZHRC(1) = 12.34
  if (ZSIGK(1) > 0.8) then
    ZHRC(1) = RAMID(1) + (1.0 - RAMID(1))*((ZSIGK(1) - 0.8)/0.01)**1.5
  end if
  d(1, 1, 2) = ZHRC(1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "pow2_ref")
    d_ref = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    mod.main(d_ref)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='main').build()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    np.testing.assert_allclose(d, d_ref, rtol=1e-12, atol=1e-12)


def test_fortran_frontend_sign1(tmp_path):
    src = """
subroutine main(d)
  implicit none
  double precision d(3, 4, 5)
  double precision :: ZSIGK(2), ZHRC(2), RAMID(2)

  ZSIGK(1) = 4.8
  RAMID(1) = 0.0
  ZHRC(1) = -12.34
  d(1, 1, 2) = sign(ZSIGK(1), ZHRC(1))
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "sign1_ref")
    d_ref = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    mod.main(d_ref)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='main').build()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    np.testing.assert_allclose(d, d_ref, rtol=1e-12, atol=1e-12)
