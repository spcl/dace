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


def test_fortran_frontend_nested_array_access():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM access_test
                    implicit none
                    double precision d(4)
                    CALL nested_array_access_test_function(d)
                    end

                    SUBROUTINE nested_array_access_test_function(d)
                    double precision d(4)
                    integer test(3,3,3)
                    integer indices(3,3,3)
                    indices(1,1,1)=2
                    indices(1,1,2)=3
                    indices(1,1,3)=1
                    test(indices(1,1,1),indices(1,1,2),indices(1,1,3))=2
                    d(test(2,3,1))=5.5
                    
                    END SUBROUTINE nested_array_access_test_function
                    """
    sources = {"nested_array_access_test_function": test_string}
    sdfg = fortran_parser.create_sdfg_from_string(test_string,
                                                  "nested_array_access_test",
                                                  normalize_offsets=True,
                                                  multiple_sdfgs=False,
                                                  sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


def test_fortran_frontend_nested_array_access2():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM access2_test
                    implicit none
                    double precision d(4)
                     simple_type
                    integer test1(3,3,3)
                    integer indices1(3,3,3)
                    

                    CALL nested_array_access2_test_function(d,test1,indices1)
                    end

                    SUBROUTINE nested_array_access2_test_function(d,test1,indices1)
                    
                    integer,pointer test1(:,:,:)
                    integer,pointer indices1(:,:,:)
                    double precision d(4)
                    integer,pointer test(:,:,:)
                    integer,pointer indices(:,:,:)

                    test1=>test
                    indices1=>indices
                    indices(1,1,1)=2
                    indices(1,1,2)=3
                    indices(1,1,3)=1
                    test(indices(1,1,1),indices(1,1,2),indices(1,1,3))=2
                    d(test(2,3,1))=5.5
                    
                    END SUBROUTINE nested_array_access2_test_function
                    """
    sources = {"nested_array_access2_test_function": test_string}
    sdfg = fortran_parser.create_sdfg_from_string(test_string,
                                                  "nested_array_access2_test",
                                                  normalize_offsets=True,
                                                  multiple_sdfgs=False,
                                                  sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


if __name__ == "__main__":

    test_fortran_frontend_nested_array_access2()
