# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
import sys, os
import numpy as np
import pytest

import dace
from dace import SDFG, SDFGState, instrument, nodes, dtypes, data, subsets, symbolic
from dace.frontend.fortran import fortran_parser
from fparser.two.symbol_table import SymbolTable
from dace.sdfg import utils as sdutil

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes

def test_fortran_frontend_index_offset_attributes():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(50:54) :: d
                    !double precision, dimension(5) :: d
                    !double precision d(50:54)
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    !double precision d(50:54)
                    !double precision d(5)
                    double precision, dimension(50:54) :: d
                    !double precision, intent(inout) :: d(50:54)

                    do i=50,54
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(50,54):
        # offset -1 is already added
        assert a[i-1] == i * 2

def test_fortran_frontend_index_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision d(50:54)
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision d(50:54)

                    do i=50,54
                       d(i) = i * 2.0
                    end do
                    
                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(50,54):
        # offset -1 is already added
        assert a[i-1] == i * 2


if __name__ == "__main__":

    test_fortran_frontend_index_offset()
    test_fortran_frontend_index_offset_attributes()
