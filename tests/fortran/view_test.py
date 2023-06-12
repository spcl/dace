# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
import sys, os
import numpy as np
import pytest

from dace import SDFG, SDFGState, nodes, dtypes, data, subsets, symbolic
from dace.frontend.fortran import fortran_parser
from fparser.two.symbol_table import SymbolTable
from dace.sdfg import utils as sdutil

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes


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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name)
    sdfg.simplify(verbose=True)
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([1, 1, 2], 42, order="F", dtype=np.float64)
    b[0, 0, 0] = 1
    sdfg(aa=a, res=b)
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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name)
    sdfg.simplify(verbose=True)
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    c = np.full([10, 11, 12], 42, order="F", dtype=np.float64)

    b[0, 0, 0] = 1
    sdfg(aa=a, bb=b, cc=c, n=10)
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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name)
    sdfg.simplify(verbose=True)
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([10, 11, 12], 42, order="F", dtype=np.float64)

    b[0, 0, 0] = 1
    sdfg(aa=a, bb=b, n=10)
    assert (b[0, 0, 0] == 1)
    assert (b[0, 0, 1] == 43)


if __name__ == "__main__":

    test_fortran_frontend_view_test()
    test_fortran_frontend_view_test_2()
    test_fortran_frontend_view_test_3()
