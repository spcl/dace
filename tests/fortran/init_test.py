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


def test_fortran_frontend_init():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    test_string = """
                    PROGRAM init_test
                    implicit none
                    USE init_test_module_subroutine, ONLY: init_test_function
                    double precision d(4)
                    CALL init_test_function(d)
                    end
                    

                    """
    sources = {}
    sources["init_test"] = test_string
    sources["init_test_module_subroutine.f90"] = """
                    MODULE init_test_module_subroutine
                    CONTAINS
                    SUBROUTINE init_test_function(d)
                    USE init_test_module, ONLY: outside_init
                    double precision d(4)
                    REAL bob=EPSILON(1.0)


                    d(2)=5.5 +bob +outside_init
                    
                    END SUBROUTINE init_test_function
                    END MODULE init_test_module_subroutine
                    """
    sources["init_test_module.f90"] = """
                    MODULE init_test_module
                    IMPLICIT NONE
                    REAL outside_init=EPSILON(1.0)
                    END MODULE init_test_module
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "init_test", sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a, outside_init=0)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


def test_fortran_frontend_init2():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    test_string = """
                    PROGRAM init2_test
                    implicit none
                    USE init2_test_module_subroutine, ONLY: init2_test_function
                    double precision d(4)
                    CALL init2_test_function(d)
                    end
                    

                    """
    sources = {}
    sources["init2_test"] = test_string
    sources["init2_test_module_subroutine.f90"] = """
                    MODULE init2_test_module_subroutine
                    CONTAINS
                    SUBROUTINE init2_test_function(d)
                    USE init2_test_module, ONLY: TORUS_MAX_LAT
                    double precision d(4)
                    

                    d(2)=5.5 + TORUS_MAX_LAT
                    
                    END SUBROUTINE init2_test_function
                    END MODULE init2_test_module_subroutine
                    """
    sources["init2_test_module.f90"] = """
                    MODULE init2_test_module
                    IMPLICIT NONE
                    REAL, PARAMETER :: TORUS_MAX_LAT = 4.0 / 18.0 * ATAN(1.0)
                    END MODULE init2_test_module
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "init2_test", sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a, torus_max_lat=4.0 / 18.0 * np.arctan(1.0))
    assert (a[0] == 42)
    assert (a[1] == 5.674532920122147)
    assert (a[2] == 42)


if __name__ == "__main__":

    test_fortran_frontend_init()
    test_fortran_frontend_init2()
