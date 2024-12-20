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


def test_fortran_frontend_rename():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    test_string = """
                    PROGRAM rename_test
                    implicit none
                    USE rename_test_module_subroutine, ONLY: rename_test_function
                    double precision d(4)
                    CALL rename_test_function(d)
                    end
                    

                    """
    sources = {}
    sources["rename_test"] = test_string
    sources["rename_test_module_subroutine.f90"] = """
                    MODULE rename_test_module_subroutine
                    CONTAINS
                    SUBROUTINE rename_test_function(d)
                    USE rename_test_module, ONLY: ik4=>i4
                    integer(ik4) :: i

                    i=4
                    d(2)=5.5 +i
                    
                    END SUBROUTINE rename_test_function
                    END MODULE rename_test_module_subroutine
                    """
    sources["rename_test_module.f90"] = """
                    MODULE rename_test_module
                    IMPLICIT NONE
                    INTEGER, PARAMETER :: pi4 =  9
                    INTEGER, PARAMETER :: i4 = SELECTED_INT_KIND(pi4)  
                    END MODULE rename_test_module
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "rename_test", sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 9.5)
    assert (a[2] == 42)


if __name__ == "__main__":

    test_fortran_frontend_rename()
