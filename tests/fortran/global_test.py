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
from dace.sdfg.nodes import AccessNode

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes


def test_fortran_frontend_global():
    """
    Tests that the Fortran frontend can parse complex global includes.
    """
    test_string = """
        PROGRAM global_test
            implicit none
            USE global_test_module_subroutine, ONLY: global_test_function
            
            REAL :: d(4), a(4,4,4)

            CALL global_test_function(d)

        end
                    

                    """
    sources={}
    sources["global_test"]=test_string
    sources["global_test_module_subroutine.f90"]="""
                    MODULE global_test_module_subroutine
                    
                    CONTAINS
                    
                    SUBROUTINE global_test_function(d)
                    USE global_test_module, ONLY: outside_init,simple_type
                    USE nested_one, ONLY: nested
                    double precision d(4)
                    double precision :: a(4,4,4)
                    integer :: i
                   
                 

            
                    TYPE(simple_type) :: ptr_patch
            
                    double precision d(4)

                    ptr_patch%w(:,:,:)=5.5

                    i=outside_init
                    CALL nested(i,ptr_patch%w)
                    d(i+1)=5.5
                    
                    END SUBROUTINE global_test_function
                    END MODULE global_test_module_subroutine
                    """
    sources["global_test_module.f90"]="""
                    MODULE global_test_module
                    IMPLICIT NONE
                    TYPE simple_type
                        double precision,POINTER :: w(:,:,:)
                        integer a
                
                    END TYPE simple_type
                    integer outside_init=1
                    END MODULE global_test_module
                    """
    
    sources["nested_one.f90"]="""
                    MODULE nested_one
                    IMPLICIT NONE
                    CONTAINS
                    SUBROUTINE nested(i,a)
                    USE nested_two, ONLY: nestedtwo
                    integer :: i
                    double precision :: a(:,:,:)

                    i=0
                    CALL nestedtwo(i)
                    a(i+1,i+1,i+1)=5.5
                    END SUBROUTINE nested
                    
                    END MODULE nested_one
                    """
    sources["nested_two.f90"]="""
                    MODULE nested_two
                    IMPLICIT NONE
                    CONTAINS
                    SUBROUTINE nestedtwo(i)
                    USE global_test_module, ONLY: outside_init
                    integer :: i
                    
                    i = outside_init+1
                    
                    END SUBROUTINE nestedtwo
                    
                    END MODULE nested_two
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "global_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    a2 = np.full([4,4,4], 42, order="F", dtype=np.float64)
    sdfg(d=a,a=a2)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)

if __name__ == "__main__":

    test_fortran_frontend_global()
