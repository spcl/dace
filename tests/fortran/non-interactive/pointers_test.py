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

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from dace.sdfg import utils as sdutil

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_pointer_test():
    """
    Tests to check whether Fortran array slices are correctly translates to DaCe views.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(lon, lout)
  real, intent(in) :: lon(10)
  real, intent(out) :: lout(10)
  type simple_type
    real:: w(5, 5, 5), z(5)
    integer:: a
  end type simple_type
  type(simple_type), target :: s
  real :: area
  real, pointer, contiguous :: p_area(:, :, :)
  integer :: i, j
  s%w(1, 1, 1) = 5.5
  lout(:) = 0.0
  p_area => s%w
  lout(1) = p_area(1, 1, 1) + lon(1)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
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





if __name__ == "__main__":

    test_fortran_frontend_pointer_test()
