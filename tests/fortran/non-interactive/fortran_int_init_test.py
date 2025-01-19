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

def test_fortran_frontend_int_init():
    """
    Tests that the power intrinsic is correctly parsed and translated to DaCe. (should become a*a)
    """
    test_string = """
                    PROGRAM int_init_test
                    implicit none
                    integer d(2)
                    CALL int_init_test_function(d)
                    end

                    SUBROUTINE int_init_test_function(d)
                    integer d(2)
                   d(1)=INT(z'000000ffffffffff',i8)               
                   END SUBROUTINE int_init_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "int_init_test",True,False)
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
    sdfg.simplify(verbose=True)
    sdfg.compile()
    d = np.full([2], 42, order="F", dtype=np.int64)
    sdfg(d=d)
    assert (d[0] == 400)



if __name__ == "__main__":

 

    test_fortran_frontend_int_init()

