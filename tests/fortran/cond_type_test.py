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


def test_fortran_frontend_cond_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: id
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL cond_type_test_function(d)
        end

        SUBROUTINE cond_type_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: ptr_patch
            LOGICAL :: bla=.TRUE.
            ptr_patch%w(1,1,1) = 5.5
            ptr_patch%id = 6
            if (ptr_patch%id .GT. 5) then
            d(2,1) = 5.5 + ptr_patch%w(1,1,1)
            else
            d(2,1) = 12
            endif
        END SUBROUTINE cond_type_test_function
    """
    sources={}
    sources["type_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    test_fortran_frontend_cond_type()
