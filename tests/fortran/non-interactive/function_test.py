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


@pytest.mark.skip(reason="Interactive test (opens SDFG).")
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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name, False)
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


@pytest.mark.skip(reason="Interactive test (opens SDFG).")
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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name, False)
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


@pytest.mark.skip(reason="Interactive test (opens SDFG).")
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
        TYPE(t_geographical_coordinates) :: p1(10)
        TYPE(t_geographical_coordinates) :: p2
    END TYPE t_line

    TYPE(t_line) :: v
    TYPE(t_geographical_coordinates) :: gp1_1
    TYPE(t_geographical_coordinates) :: gp1_2
    TYPE(t_geographical_coordinates) :: gp1_3
    TYPE(t_geographical_coordinates) :: gp1_4
    TYPE(t_geographical_coordinates) :: gp1_5
    TYPE(t_geographical_coordinates) :: gp1_6
    TYPE(t_geographical_coordinates) :: gp1_7
    TYPE(t_geographical_coordinates) :: gp1_8
    TYPE(t_geographical_coordinates) :: gp1_9
    TYPE(t_geographical_coordinates) :: gp1_10

    gp1_1%lon = 1.0
    gp1_1%lat = 1.0
    gp1_2%lon = 2.0
    gp1_2%lat = 2.0
    gp1_3%lon = 3.0
    gp1_3%lat = 3.0
    gp1_4%lon = 4.0
    gp1_4%lat = 4.0
    gp1_5%lon = 5.0
    gp1_5%lat = 5.0
    gp1_6%lon = 6.0
    gp1_6%lat = 6.0
    gp1_7%lon = 7.0
    gp1_7%lat = 7.0
    gp1_8%lon = 8.0
    gp1_8%lat = 8.0
    gp1_9%lon = 9.0
    gp1_9%lat = 9.0
    gp1_10%lon = 10.0
    gp1_10%lat = 10.0

    v%p1(1) = gp1_1
    v%p1(2) = gp1_2
    v%p1(3) = gp1_3
    v%p1(4) = gp1_4
    v%p1(5) = gp1_5
    v%p1(6) = gp1_6
    v%p1(7) = gp1_7
    v%p1(8) = gp1_8
    v%p1(9) = gp1_9
    v%p1(10) = gp1_10

    z = function3_test_function(v)

END PROGRAM """ + test_name + """_program

ELEMENTAL FUNCTION function3_test_function (v) result(length)
    TYPE(t_line), INTENT(in) :: v
    REAL :: length
    REAL :: segment
    REAL :: dlon
    REAL :: dlat

    length = 0
    DO i = 1, 9
        segment = 0
        dlon = 0
        dlat = 0
        dlon = v%p1(i + 1)%lon - v%p1(i)%lon
        dlat = v%p1(i + 1)%lat - v%p1(i)%lat
        segment = dlon * dlon + dlat * dlat
        length = length + SQRT(segment)
    ENDDO

END FUNCTION function3_test_function
"""

    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name, False)
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


@pytest.mark.skip(reason="Interactive test (opens SDFG).")
def test_fortran_frontend_function_test4():
    """
    Test for elemental functions
    """
    test_name = "function4_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none

REAL b
REAL v
REAL z(10)
z(:)=4.0

b=function4_test_function(v,z)

end



   FUNCTION function4_test_function (v,z) result(length)
     REAL, INTENT(in) :: v
     REAL z(10)
     REAL :: length


REAL a(10)
REAL b



a=norm(z)
length=norm(v)+a

  END FUNCTION function4_test_function

 ELEMENTAL FUNCTION norm (v) result(length)
     REAL, INTENT(in) :: v
     REAL :: length


     length = v*v

  END FUNCTION norm




                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name, False)
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


@pytest.mark.skip(reason="Interactive test (opens SDFG).")
def test_fortran_frontend_function_test5():
    """
    Test for elemental functions
    """
    test_name = "function5_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none

REAL b
REAL v
REAL z(10)
REAL y(10)
INTEGER proc(10)
INTEGER keyval(10)
z(:)=4.0

CALL function5_test_function(z,y,10,1,2,proc,keyval,3,0)

end



  SUBROUTINE function5_test_function(in_field, out_field, n, op, loc_op, &
       proc_id, keyval, comm, root)
    INTEGER, INTENT(in) :: n, op, loc_op
    REAL, INTENT(in) :: in_field(n)
    REAL, INTENT(out) :: out_field(n)

    INTEGER, OPTIONAL, INTENT(inout) :: proc_id(n)
    INTEGER, OPTIONAL, INTENT(inout) :: keyval(n)
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm


    out_field = in_field

  END SUBROUTINE function5_test_function



                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name, False)
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
    #test_fortran_frontend_function_test3()
    test_fortran_frontend_function_test4()
    #test_fortran_frontend_function_test5()
    #test_fortran_frontend_view_test_2()
    #test_fortran_frontend_view_test_3()
