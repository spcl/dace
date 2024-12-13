# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_real_kind_selector():
    """
    Tests that the size intrinsics are correctly parsed and translated to DaCe.
    """
    test_string = """
program real_kind_selector_test
  implicit none
  integer, parameter :: JPRB = selected_real_kind(13, 300)
  real(KIND=JPRB) d(4)
  call real_kind_selector_test_function(d)
end

subroutine real_kind_selector_test_function(d)
  implicit none
  integer, parameter :: JPRB = selected_real_kind(13, 300)
  integer, parameter :: JPIM = selected_int_kind(9)
  real(KIND=JPRB) d(4)
  integer(KIND=JPIM) i

  i = 7
  d(2) = 5.5 + i

end subroutine real_kind_selector_test_function
"""
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "real_kind_selector_test")
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
    test_string = """
                    PROGRAM if1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL if1_test_function(d)
                    end

                    SUBROUTINE if1_test_function(d)
                    double precision d(3,4,5),ZFAC(10)
                    integer JK,JL,RTT,NSSOPT
                    integer ZTP1(10,10)
                    JL=1
                    JK=1
                    ZTP1(JL,JK)=1.0
                    RTT=2
                    NSSOPT=1

                    IF (ZTP1(JL,JK)>=RTT .OR. NSSOPT==0) THEN
                      ZFAC(1)  = 1.0
                    ELSE
                      ZFAC(1)  = 2.0
                    ENDIF
                    d(1,1,1)=ZFAC(1)
                                    
                    END SUBROUTINE if1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "if1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 2)


def test_fortran_frontend_loop1():
    """
    Tests that the loop construct is correctly parsed and translated to DaCe.
    """

    test_string = """
program loop1_test
  implicit none
  logical :: d(3, 4, 5)
  call loop1_test_function(d)
end

subroutine loop1_test_function(d)
  logical :: d(3, 4, 5), ZFAC(10)
  integer :: a, JK, JL, JM
  integer, parameter :: KLEV = 10, N = 10, NCLV = 3
  integer :: tmp

  double precision :: RLMIN, ZVQX(NCLV)
  logical :: LLCOOLJ, LLFALL(NCLV)
  LLFALL(:) = .false.
  ZVQX(:) = 0.0
  ZVQX(2) = 1.0
  do JM = 1, NCLV
    if (ZVQX(JM) > 0.0) LLFALL(JM) = .true. ! falling species
  end do

  do I = 1, 3
    do J = 1, 4
      do K = 1, 5
        tmp = I+J+K-3
        tmp = mod(tmp, 2)
        if (tmp == 1) then
          d(I, J, K) = LLFALL(2)
        else
          d(I, J, K) = LLFALL(1)
        end if
      end do
    end do
  end do
end subroutine loop1_test_function
"""
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "loop1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    sdfg(d=d)
    # Verify the checkerboard pattern.
    assert all(bool(v) == ((i+j+k) % 2 == 1) for (i, j, k), v in np.ndenumerate(d))


def test_fortran_frontend_function_statement1():
    """
    Tests that the function statement are correctly removed recursively.
    """

    test_string = """
program function_statement1_test
  implicit none
  double precision d(3, 4, 5)
  call function_statement1_test_function(d)
end

subroutine function_statement1_test_function(d)
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
end subroutine function_statement1_test_function
"""
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "function_statement1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 5.1)
    assert (d[0, 0, 1] == 5.1)


def test_fortran_frontend_pow1():
    """
    Tests that the power intrinsic is correctly parsed and translated to DaCe. (should become a*a)
    """
    test_string = """
                    PROGRAM pow1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL pow1_test_function(d)
                    end

                    SUBROUTINE pow1_test_function(d)
                   double precision d(3,4,5)
                  double precision :: ZSIGK(2), ZHRC(2),RAMID(2)

                  ZSIGK(1)=4.8
                  RAMID(1)=0.0
                  ZHRC(1)=12.34
                  IF(ZSIGK(1) > 0.8) THEN
                          ZHRC(1)=RAMID(1)+(1.0-RAMID(1))*((ZSIGK(1)-0.8)/0.2)**2
                  ENDIF
                   d(1,1,2)=ZHRC(1)                 
                   END SUBROUTINE pow1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "pow1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == 400)


def test_fortran_frontend_pow2():
    """
    Tests that the power intrinsic is correctly parsed and translated to DaCe (this time it's p sqrt p).
    """

    test_string = """
                    PROGRAM pow2_test
                    implicit none
                    double precision d(3,4,5)
                    CALL pow2_test_function(d)
                    end

                    SUBROUTINE pow2_test_function(d)
                   double precision d(3,4,5)
                  double precision :: ZSIGK(2), ZHRC(2),RAMID(2)

                  ZSIGK(1)=4.8
                  RAMID(1)=0.0
                  ZHRC(1)=12.34
                  IF(ZSIGK(1) > 0.8) THEN
                          ZHRC(1)=RAMID(1)+(1.0-RAMID(1))*((ZSIGK(1)-0.8)/0.01)**1.5
                  ENDIF
                   d(1,1,2)=ZHRC(1)                 
                   END SUBROUTINE pow2_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "pow2_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == 8000)


def test_fortran_frontend_sign1():
    """
    Tests that the sign intrinsic is correctly parsed and translated to DaCe.
    """
    test_string = """
                    PROGRAM sign1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL sign1_test_function(d)
                    end

                    SUBROUTINE sign1_test_function(d)
                   double precision d(3,4,5)
                  double precision :: ZSIGK(2), ZHRC(2),RAMID(2)

                  ZSIGK(1)=4.8
                  RAMID(1)=0.0
                  ZHRC(1)=-12.34
                   d(1,1,2)=SIGN(ZSIGK(1),ZHRC(1))                 
                   END SUBROUTINE sign1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "sign1_test")
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
