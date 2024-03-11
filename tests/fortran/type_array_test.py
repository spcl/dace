# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

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

from dace.transformation.passes.lift_struct_views import LiftStructViews
from dace.transformation import pass_pipeline as ppl

def test_fortran_frontend_type_array():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_array_test
            implicit none

          
            TYPE simple_type
                REAL :: w(5,5)
            END TYPE simple_type

             TYPE simple_type2
                type(simple_type) :: pprog(10)
            END TYPE simple_type2

            REAL :: d(5,5)
            CALL type_array_test_function(d)
            print *, d(1,1)
        end

        SUBROUTINE type_array_test_function(d)
            REAL :: d(5,5)
            TYPE(simple_type2) :: p_prog

            CALL type_array_test_f2(p_prog%pprog(1))
            d(1,1) = p_prog%pprog(1)%w(1,1)
        END SUBROUTINE type_array_test_function

        SUBROUTINE type_array_test_f2(stuff)
            TYPE(simple_type) :: stuff
            CALL deepest(stuff%w)
            
        END SUBROUTINE type_array_test_f2

        SUBROUTINE deepest(my_arr)
            REAL :: my_arr(:,:)

            my_arr(1,1) = 42
        END SUBROUTINE deepest

    """
    sources={}
    sources["type_array_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_array_test",sources=sources, normalize_offsets=True)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)



if __name__ == "__main__":
  
   test_fortran_frontend_type_array()
