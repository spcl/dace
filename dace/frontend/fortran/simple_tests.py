# Copyright 2022 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import *
import sys, os
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dace.frontend.fortran.fortran_parser import *


def test_fortran_frontend_simplify():
    test_string = """
                    PROGRAM symbol_test
                    implicit none
                    double precision d(2,3)
                    CALL symbol_test_function(d)
                    end

                    SUBROUTINE symbol_test_function(d)
                    double precision d(2,3)
                    integer a,b

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=5
                    
                    END SUBROUTINE symbol_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "symbol_test")
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d_0=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


def test_fortran_frontend_scalar():
    test_string = """
                    PROGRAM scalar_test
                    implicit none
                    double precision d
                    CALL scalar_test_function(d)
                    end

                    SUBROUTINE scalar_test_function(d)
                    double precision d

                    d=d+1.5
                    
                    END SUBROUTINE scalar_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "scalar_test")
    sdfg.simplify(verbose=True)
    res = np.zeros((1, ), dtype=np.float64)[0]
    res = 42.0
    sdfg(d_0=res)
    #auto generated correct assert !?!
    assert (res == 43.5)


def test_fortran_frontend_input_output_connector():
    test_string = """
                    PROGRAM ioc_test
                    implicit none
                    double precision d(2,3)
                    CALL ioc_test_function(d)
                    end

                    SUBROUTINE ioc_test_function(d)
                    double precision d(2,3)
                    integer a,b

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=d(1,1)+5
                    
                    END SUBROUTINE ioc_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "ioc_test")
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d_0=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


def test_fortran_frontend_view_test():
    test_name = "view_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none
double precision a(10,11,12)
double precision res(1,1,2) 

CALL """ + test_name + """_function(a,res)

end

SUBROUTINE """ + test_name + """_function(aa,res)

double precision aa(10,11,12)
double precision res(1,1,2) 

call viewlens(aa(:,:,1),res)

end SUBROUTINE """ + test_name + """_function

SUBROUTINE viewlens(aa,res)

IMPLICIT NONE

double precision  :: aa(10,11,23) 
double precision :: res(1,1,2)

INTEGER ::  JK, JL

res(1,1,1)=0.0
DO JK=1,10
  DO JL=1,11
    res(1,1,1)=res(1,1,1)+aa(JK,JL)
  ENDDO
ENDDO
aa(1,1)=res(1,1,1)


END SUBROUTINE viewlens
                    """
    sdfg = create_sdfg_from_string(test_string, test_name)
    sdfg.simplify(verbose=True)
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([1, 1, 2], 42, order="F", dtype=np.float64)
    b[0, 0, 0] = 1
    sdfg(aa_0=a, res_0=b)
    assert (a[0, 0, 1] == 42)
    assert (a[0, 0, 0] == 4620)
    assert (b[0, 0, 0] == 4620)


def test_fortran_frontend_view_test_2():
    test_name = "view2_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none
integer, parameter :: n=10
double precision a(n,11,12),b(n,11,12),c(n,11,12)

CALL """ + test_name + """_function(a,b,c,n)

end

SUBROUTINE """ + test_name + """_function(aa,bb,cc,n)

integer, parameter :: n=10
double precision a(n,11,12),b(n,11,12),c(n,11,12)
integer j,k

j=1
    call viewlens(aa(:,:,j),bb(:,:,j),cc(:,:,j))
k=2
    call viewlens(aa(:,:,k),bb(:,:,k),cc(:,:,k))

end SUBROUTINE """ + test_name + """_function

SUBROUTINE viewlens(aa,bb,cc)

IMPLICIT NONE

double precision  :: aa(10,11),bb(10,11),cc(10,11) 

INTEGER ::  JK, JL

DO JK=1,10
  DO JL=1,11
    cc(JK,JL)=bb(JK,JL)+aa(JK,JL)
  ENDDO
ENDDO

END SUBROUTINE viewlens
                    """
    sdfg = create_sdfg_from_string(test_string, test_name)
    sdfg.simplify(verbose=True)
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    c = np.full([10, 11, 12], 42, order="F", dtype=np.float64)

    b[0, 0, 0] = 1
    sdfg(aa_0=a, bb_0=b, cc_0=c, n=10)
    assert (c[0, 0, 0] == 43)
    assert (c[1, 1, 1] == 84)


def test_fortran_frontend_view_test_3():
    test_name = "view3_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none
integer, parameter :: n=10
double precision a(n,n+1,12),b(n,n+1,12)

CALL """ + test_name + """_function(a,b,n)

end

SUBROUTINE """ + test_name + """_function(aa,bb,n)

integer, parameter :: n=10
double precision a(n,n+1,12),b(n,n+1,12)
integer j,k

j=1
    call viewlens(aa(:,:,j),bb(:,:,j),bb(:,:,j+1))

end SUBROUTINE """ + test_name + """_function

SUBROUTINE viewlens(aa,bb,cc)

IMPLICIT NONE

double precision  :: aa(10,11),bb(10,11),cc(10,11) 

INTEGER ::  JK, JL

DO JK=1,10
  DO JL=1,11
    cc(JK,JL)=bb(JK,JL)+aa(JK,JL)
  ENDDO
ENDDO

END SUBROUTINE viewlens
                    """
    sdfg = create_sdfg_from_string(test_string, test_name)
    sdfg.simplify(verbose=True)
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([10, 11, 12], 42, order="F", dtype=np.float64)

    b[0, 0, 0] = 1
    sdfg(aa_0=a, bb_0=b, n=10)
    assert (b[0, 0, 0] == 1)
    assert (b[0, 0, 1] == 43)


def test_fortran_frontend_array_access():
    test_string = """
                    PROGRAM access_test
                    implicit none
                    double precision d(4)
                    CALL array_access_test_function(d)
                    end

                    SUBROUTINE array_access_test_function(d)
                    double precision d(4)

                    d(2)=5.5
                    
                    END SUBROUTINE array_access_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "array_access_test")
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d_0=a)
    assert (a[0] == 42)
    assert (a[1] == 5.5)


def test_fortran_frontend_array_ranges():
    test_string = """
                    PROGRAM ranges_test
                    implicit none
                    double precision d(3,4,5)
                    CALL array_ranges_test_function(d)
                    end

                    SUBROUTINE array_ranges_test_function(d)
                    double precision d(3,4,5),e(3,4,5),f(3,4,5)

                    e(:,:,:)=1.0
                    f(:,:,:)=2.0
                    f(:,2:4,:)=3.0
                    f(1,1,:)=4.0
                    d(:,:,:)=e(:,:,:)+f(:,:,:)
                    d(1,2:4,1)=e(1,2:4,1)*10.0
                    d(1,1,1)=SUM(e(:,1,:))
                    
                    END SUBROUTINE array_ranges_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "array_access_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 0] == 15)
    assert (d[0, 1, 0] == 10)
    assert (d[1, 0, 0] == 3)
    assert (d[1, 1, 0] == 4)
    assert (d[0, 0, 2] == 5)


def test_fortran_frontend_if1():
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
    sdfg = create_sdfg_from_string(test_string, "if1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 0] == 2)


def test_fortran_frontend_loop1():
    test_string = """
                    PROGRAM loop1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL loop1_test_function(d)
                    end

                    SUBROUTINE loop1_test_function(d)
                   double precision d(3,4,5),ZFAC(10)
                   INTEGER :: a, JK, JL,JM
                   INTEGER, PARAMETER :: KLEV=10, N=10,NCLV=3

                   double precision :: RLMIN,ZVQX(NCLV)
                   LOGICAL :: LLCOOLJ,LLFALL(NCLV)
                   LLFALL(:)= .FALSE.
                   ZVQX(:)= 0.0
                   ZVQX(2)= 1.0
                   DO JM=1,NCLV
                    IF (ZVQX(JM)>0.0) LLFALL(JM)=.TRUE. ! falling species
                   ENDDO

                   d(1,1,1)=LLFALL(1)
                   d(1,1,2)=LLFALL(2)                 
                   END SUBROUTINE loop1_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "loop1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 0] == 0)
    assert (d[0, 0, 1] == 1)


def test_fortran_frontend_function_statement1():
    test_string = """
                    PROGRAM function_statement1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL function_statement1_test_function(d)
                    end

                    SUBROUTINE function_statement1_test_function(d)
                   double precision d(3,4,5)
                   double precision :: PTARE,RTT(2),FOEDELTA,FOELDCP
                   double precision :: RALVDCP(2),RALSDCP(2),RES

                    FOEDELTA (PTARE) = MAX (0.0,SIGN(1.0,PTARE-RTT(1)))
                    FOELDCP ( PTARE ) = FOEDELTA(PTARE)*RALVDCP(1) + (1.0-FOEDELTA(PTARE))*RALSDCP(1)

                    RTT(1)=4.5
                    RALVDCP(1)=4.9
                    RALSDCP(1)=5.1
                    d(1,1,1)=FOELDCP(3.0)
                    RES=FOELDCP(3.0)
                   d(1,1,2)=RES                 
                   END SUBROUTINE function_statement1_test_function
                    """
    sdfg = create_sdfg_from_string(test_string, "function_statement1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 0] == 5.1)
    assert (d[0, 0, 1] == 5.1)


def test_fortran_frontend_pow1():
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
    sdfg = create_sdfg_from_string(test_string, "pow1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 1] == 400)


def test_fortran_frontend_pow2():
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
    sdfg = create_sdfg_from_string(test_string, "pow2_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 1] == 8000)


def test_fortran_frontend_sign1():
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
    sdfg = create_sdfg_from_string(test_string, "sign1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d_0=d)
    assert (d[0, 0, 1] == -4.8)


if __name__ == "__main__":

    # test_fortran_frontend_array_access()
    # test_fortran_frontend_simplify()
    # test_fortran_frontend_input_output_connector()
    # test_fortran_frontend_view_test()
    # test_fortran_frontend_view_test_2()
    test_fortran_frontend_view_test_3()
    # test_fortran_frontend_array_ranges()
    # test_fortran_frontend_if1()
    # test_fortran_frontend_loop1()
    # test_fortran_frontend_function_statement1()

    # test_fortran_frontend_pow1()
    # test_fortran_frontend_pow2()
    # test_fortran_frontend_sign1()
    # test_fortran_frontend_scalar()
    print("All tests passed")
