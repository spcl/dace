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


def test_fortran_frontend_basic_allocate():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM allocate_test
                    implicit none
                    double precision, allocatable :: d(:,:)
                    allocate(d(4,5))
                    CALL allocate_test_function(d)
                    end

                    SUBROUTINE allocate_test_function(d)
                    double precision d(4,5)
                    
                    d(2,1)=5.5
                    
                    END SUBROUTINE allocate_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "allocate_test")
    sdfg.simplify(verbose=True)
    a = np.full([4,5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0,0] == 42)
    assert (a[1,0] == 5.5)
    assert (a[2,0] == 42)


if __name__ == "__main__":

    test_fortran_frontend_basic_allocate()
