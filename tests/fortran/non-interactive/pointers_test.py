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
def test_fortran_frontend_pointer_test():
    """
    Tests to check whether Fortran array slices are correctly translates to DaCe views.
    """
    test_name = "pointer_test"
    test_string = """
                    PROGRAM """ + test_name + """_program
implicit none
REAL lon(10)
REAL lout(10)
TYPE simple_type
                        REAL:: w(5,5,5),z(5)
                        INTEGER:: a         
END TYPE simple_type

lon(:) = 1.0
CALL pointer_test_function(lon,lout)

end


  SUBROUTINE pointer_test_function (lon,lout)
     REAL, INTENT(in) :: lon(10)
     REAL, INTENT(out) :: lout(10)
     TYPE(simple_type) :: s
     REAL :: area
     REAL, POINTER, CONTIGUOUS :: p_area
     INTEGER :: i,j
     
     s%w(1,1,1)=5.5
     lout(:)=0.0
     p_area  => s%w
    
     lout(1)=p_area(1,1,1)+lon(1)

     
   END SUBROUTINE pointer_test_function
  
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name, False, False)
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
    sdfg.validate()
    sdfg.simplify(verbose=True)
    sdfg.view()


if __name__ == "__main__":

    test_fortran_frontend_pointer_test()
