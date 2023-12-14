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


def test_fortran_frontend_function_test():
    """
    Tests to check whether Fortran array slices are correctly translates to DaCe views.
    """
    test_name = "function_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none
INTEGER a
INTEGER lon(10)
INTEGER lat(10)

a=function_test_function(1,lon,lat,10)

end


  INTEGER FUNCTION function_test_function (lonc, lon, lat, n)
     INTEGER, INTENT(in) :: n
     REAL, INTENT(in) :: lonc
     REAL, INTENT(in) :: lon(n), lat(n)
     REAL :: pi=3.14 
     REAL :: lonl(n), latl(n)

     REAL :: area

     INTEGER :: i,j

     lonl(:) = lon(:)
     latl(:) = lat(:)

     DO i = 1, n
       lonl(i) = lonl(i) - lonc
       IF (lonl(i) < -pi) THEN
         lonl(i) =  pi+MOD(lonl(i), pi)
       ENDIF
       IF (lonl(i) >  pi) THEN
         lonl(i) = -pi+MOD(lonl(i), pi)
       ENDIF
     ENDDO

     area = 0.0
     DO i = 1, n
       j = MOD(i,n)+1
       area = area+lonl(i)*latl(j)
       area = area-latl(i)*lonl(j)
     ENDDO

     IF (area >= 0.0) THEN
       function_test_function = +1
     ELSE
       function_test_function = -1
     END IF

   END FUNCTION function_test_function
  
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name,False,False)
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if node.sdfg is not None:
                if 'test_function' in node.sdfg.name:
                    sdfg = node.sdfg
                    break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()                
    sdfg.simplify(verbose=True)
    sdfg.view()




def test_fortran_frontend_function_test2():
    """
    Tests to check whether Fortran array slices are correctly translates to DaCe views.
    """
    test_name = "function2_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none
REAL x(3)
REAL y(3)
REAL z 

z=function2_test_function(x,y)

end


  
PURE FUNCTION function2_test_function (p_x, p_y)  result (p_arc)
    REAL, INTENT(in) :: p_x(3), p_y(3)  ! endpoints

    REAL            :: p_arc          ! length of geodesic arc

    REAL            :: z_lx,  z_ly    ! length of vector p_x and p_y
    REAL            :: z_cc           ! cos of angle between endpoints

    !-----------------------------------------------------------------------

    !z_lx = SQRT(DOT_PRODUCT(p_x,p_x))
    !z_ly = SQRT(DOT_PRODUCT(p_y,p_y))

    !z_cc = DOT_PRODUCT(p_x, p_y)/(z_lx*z_ly)

    ! in case we get numerically incorrect solutions

    !IF (z_cc > 1._wp )  z_cc =  1.0
    !IF (z_cc < -1._wp ) z_cc = -1.0
    z_cc= p_x(1)*p_y(1)+p_x(2)*p_y(2)+p_x(3)*p_y(3)
    p_arc = ACOS(z_cc)

  END FUNCTION function2_test_function

  
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name,False,False)
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if node.sdfg is not None:
                if 'test_function' in node.sdfg.name:
                    sdfg = node.sdfg
                    break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()                
    sdfg.simplify(verbose=True)
    sdfg.view()



def test_fortran_frontend_function_test3():
    """
    Tests to check whether Fortran array slices are correctly translates to DaCe views.
    """
    test_name = "function3_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none

REAL z

! cartesian coordinate class
    TYPE t_cartesian_coordinates
        REAL :: x(3)
    END TYPE t_cartesian_coordinates

    ! geographical coordinate class
    TYPE t_geographical_coordinates
        REAL :: lon
        REAL :: lat
    END TYPE t_geographical_coordinates

    ! the two coordinates on the tangent plane
    TYPE t_tangent_vectors
        REAL :: v1
        REAL :: v2
    END TYPE t_tangent_vectors

    ! line class
    TYPE t_line
        TYPE(t_geographical_coordinates) :: p1
        TYPE(t_geographical_coordinates) :: p2
    END TYPE t_line

TYPE(t_cartesian_coordinates) :: v    

v%x(1)=1.0
v%x(2)=2.0
v%x(3)=3.0

z=function3_test_function(v)

end


  
   ELEMENTAL FUNCTION function3_test_function (v) result(length)
     TYPE(t_cartesian_coordinates), INTENT(in) :: v
     REAL :: length

     !length = SQRT(DOT_PRODUCT(v%x,v%x))
     length = v%x(1) *v%x(2)*v%x(3)

  END FUNCTION function3_test_function

  
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name,False,False)
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if node.sdfg is not None:
                if 'test_function' in node.sdfg.name:
                    sdfg = node.sdfg
                    break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()                
    sdfg.simplify(verbose=True)
    sdfg.view()
    sdfg.compile()


if __name__ == "__main__":

    #test_fortran_frontend_function_test()
    #test_fortran_frontend_function_test2()
    test_fortran_frontend_function_test3()
    #test_fortran_frontend_view_test_2()
    #test_fortran_frontend_view_test_3()
