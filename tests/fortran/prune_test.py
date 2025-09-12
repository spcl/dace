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


def test_fortran_frontend_prune_simple():
    test_string = """
                    PROGRAM init_test
                    implicit none
                    double precision d(4)
                    double precision dx(4)
                    CALL init_test_function(d, dx)
                    end

                    SUBROUTINE init_test_function(d, dx)

                    double precision dx(4)
                    double precision d(4)

                    d(2) = d(1) + 3.14

                    END SUBROUTINE init_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "init_test", False)
    print('a', flush=True)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    b = np.full([4], 42, order="F", dtype=np.float64)
    print(a)
    sdfg(d=a, dx=b)
    print(a)
    assert (a[0] == 42)
    assert (a[1] == 42 + 3.14)
    assert (a[2] == 42)


if __name__ == "__main__":

    test_fortran_frontend_prune_simple()
