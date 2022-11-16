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
                    real d(2,3)
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
                    real d(2,3)
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


if __name__ == "__main__":

    test_fortran_frontend_simplify()
    test_fortran_frontend_input_output_connector()
    test_fortran_frontend_view_test()
    print("All tests passed")
