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


def test_fortran_frontend_empty():
    """ 
    Test that empty subroutines and functions are correctly parsed.
    """
    sources={}
    test_string = """
                    PROGRAM empty_test
                    implicit none
                    use module_mpi, only: my_process_is_mpi_all_seq_test_function
                    double precision d(2,3)
                    logical bla=False

                    bla=my_process_is_mpi_all_seq_test_function()
                    if (bla == .TRUE.) then
                        d(1,1)=0
                        d(1,2)=5
                        d(2,3)=0
                        else
                        d(1,2)=1
                        endif
                    end

                    
                    """
    sources["module_mpi.f90"]="""
                    MODULE module_mpi
                    integer process_mpi_all_size=0
                    CONTAINS
                    LOGICAL FUNCTION my_process_is_mpi_all_seq_test_function()
                        my_process_is_mpi_all_seq = (process_mpi_all_size <= 1)
                    END FUNCTION my_process_is_mpi_all_seq_test_function
                    END MODULE module_mpi
                    """
    sources["empty_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "empty_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


if __name__ == "__main__":
    test_fortran_frontend_empty()
