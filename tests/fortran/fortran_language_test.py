# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_real_kind_selector():
    """
    Tests that the size intrinsics are correctly parsed and translated to DaCe.
    """
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(d)
  implicit none
  integer, parameter :: JPRB = selected_real_kind(13, 300)
  integer, parameter :: JPIM = selected_int_kind(9)
  real(KIND=JPRB) d(4)
  integer(KIND=JPIM) i

  i = 7
  d(2) = 5.5 + i
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 12.5)
    assert (a[2] == 42)


def test_fortran_frontend_if1():
    """
    Tests that the if/else construct is correctly parsed and translated to DaCe.
    """
    sources, main = SourceCodeBuilder().add_file(
        """
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 2)


def test_fortran_frontend_loop1():
    """
    Tests that the loop construct is correctly parsed and translated to DaCe.
    """
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(d)
  logical d(3, 4, 5), ZFAC(10)
  integer :: a, JK, JL, JM
  integer, parameter :: KLEV = 10, N = 10, NCLV = 3

  double precision :: RLMIN, ZVQX(NCLV)
  logical :: LLCOOLJ, LLFALL(NCLV)
  LLFALL(:) = .false.
  ZVQX(:) = 0.0
  ZVQX(2) = 1.0
  do JM = 1, NCLV
    if (ZVQX(JM) > 0.0) LLFALL(JM) = .true. ! falling species
  end do

  d(1, 1, 1) = LLFALL(1)
  d(1, 1, 2) = LLFALL(2)
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 1, order="F", dtype=np.int32)
    sdfg(d=d)
    assert (d[0, 0, 0] == 0)
    assert (d[0, 0, 1] == 1)


def test_fortran_frontend_function_statement():
    """
    Tests that the function statement are correctly removed recursively.
    """
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(d)
  double precision d(3, 4, 5)
  double precision :: PTARE, RTT(2), FOEDELTA, FOELDCP
  double precision :: RALVDCP(2), RALSDCP(2), RES

  FOEDELTA(PTARE) = max(0.0, sign(1.d0, PTARE - RTT(1)))
  FOELDCP(PTARE) = FOEDELTA(PTARE)*RALVDCP(1) + (1.0 - FOEDELTA(PTARE))*RALSDCP(1)

  RTT(1) = 4.5
  RALVDCP(1) = 4.9
  RALSDCP(1) = 5.1
  d(1, 1, 1) = FOELDCP(3.d0)
  RES = FOELDCP(3.d0)
  d(1, 1, 2) = RES
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 5.1)
    assert (d[0, 0, 1] == 5.1)


def test_internal_subprograms():
    """
    Tests that the function statement are correctly removed recursively.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([1], 42, order="F", dtype=np.float32)
    sdfg(d=d)
    assert np.allclose(d, [2])


def test_fortran_frontend_pow1():
    """
    Tests that the power intrinsic is correctly parsed and translated to DaCe. (should become a*a)
    """
    sources, main = SourceCodeBuilder().add_file(
        """
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == 400)


def test_fortran_frontend_pow2():
    """
    Tests that the power intrinsic is correctly parsed and translated to DaCe (this time it's p sqrt p).
    """
    sources, main = SourceCodeBuilder().add_file(
        """
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == 8000)


def test_fortran_frontend_sign1():
    """
    Tests that the sign intrinsic is correctly parsed and translated to DaCe.
    """
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(d)
  implicit none
  double precision d(3, 4, 5)
  double precision :: ZSIGK(2), ZHRC(2), RAMID(2)

  ZSIGK(1) = 4.8
  RAMID(1) = 0.0
  ZHRC(1) = -12.34
  d(1, 1, 2) = sign(ZSIGK(1), ZHRC(1))
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == -4.8)


if __name__ == "__main__":
    test_fortran_frontend_real_kind_selector()
    test_fortran_frontend_if1()
    test_fortran_frontend_loop1()
    test_fortran_frontend_function_statement1()
    test_fortran_frontend_pow1()
    test_fortran_frontend_pow2()
    test_fortran_frontend_sign1()
