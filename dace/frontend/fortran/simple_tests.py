from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import *
import sys, os
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from fcdc import *


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
double precision a(10,11,12),b(10,11,12),c(10,11,12)

CALL """ + test_name + """_function(a,b,c)

end

SUBROUTINE """ + test_name + """_function(aa,bb,cc)

double precision aa(10,11,12),bb(10,11,12),cc(10,11,12)
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
    sdfg(aa_0=a, bb_0=b, cc_0=c)
    assert (c[0, 0, 0] == 43)
    assert (c[1, 1, 1] == 84)


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
                    !f(:,:,:)=2.0
                    !f(:,2:4,:)=3.0
                    !f(1,1,:)=4.0
                    !d(:,:,:)=e(:,:,:)+f(:,:,:)
                    !d(1,2:4,1)=e(1,2:4,1)*10.0
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


if __name__ == "__main__":

    #test_fortran_frontend_array_access()
    #test_fortran_frontend_simplify()
    #test_fortran_frontend_input_output_connector()
    #Breaks due to codegen if scalars are used for res instead an array
    #test_fortran_frontend_view_test()
    #test_fortran_frontend_view_test_2()
    #break due to codegen if scalars are used (redefined scalar in tasklet -> unitialised)
    #test_fortran_frontend_array_ranges()
    #breaks if array has exactly one element or scalars are used.
    test_fortran_frontend_if1()

    print("All tests passed")
