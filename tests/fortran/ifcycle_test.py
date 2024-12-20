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


def test_fortran_frontend_if_cycle():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM if_cycle_test
                    implicit none
                    double precision :: d(4)
                    CALL if_cycle_test_function(d)
                    end

                    SUBROUTINE if_cycle_test_function(d)
                    double precision d(4,5)
                    integer :: i
                    DO i=1,4              
                     if (i .eq. 2) CYCLE
                     d(i)=5.5
                    ENDDO             
                    if (d(2) .eq. 42) d(2)=6.5
                    

                    END SUBROUTINE if_cycle_test_function
                    """
    sources={}
    sources["if_cycle"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "if_cycle",normalize_offsets=True,multiple_sdfgs=False,sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 5.5)
    assert (a[1] == 6.5)
    assert (a[2] == 5.5)



def test_fortran_frontend_if_nested_cycle():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM if_nested_cycle_test
                    implicit none
                    double precision :: d(4,4)
                    
                    CALL if_nested_cycle_test_function(d)
                    end

                    SUBROUTINE if_nested_cycle_test_function(d)
                    double precision d(4,4)
                    double precision :: tmp
                    integer :: i,limit,start,count
                    limit=4
                    start=1
                    DO i=start,limit
                        count=0
                        
                        DO j=start,limit             
                            if (j .eq. 2) count=count+2
                        ENDDO
                        if (count .eq. 2) CYCLE
                        if (count .eq. 3) CYCLE
                        DO j=start,limit
                             
                            d(i,j)=d(i,j)+1.5
                        ENDDO
                        d(i,1)=5.5    
                    ENDDO             
                    
                    if (d(2,1) .eq. 42.0) d(2,1)=6.5
                    

                    END SUBROUTINE if_nested_cycle_test_function
                    """
    sources={}
    sources["if_nested_cycle"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "if_nested_cycle",normalize_offsets=True,multiple_sdfgs=False,sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4,4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0,0] == 42)
    assert (a[1,0] == 6.5)
    assert (a[2,0] == 42)    


if __name__ == "__main__":

    test_fortran_frontend_if_nested_cycle()
