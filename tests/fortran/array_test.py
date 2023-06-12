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


def test_fortran_frontend_array_access():
    test_string = """
                    PROGRAM access_test
                    implicit none
                    double precision d(4)
                    CALL array_access_test_function(d)
                    end

                    SUBROUTINE array_access_test_function(d)
                    double precision d(4)

                    d(2)=5.5
                    
                    END SUBROUTINE array_access_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "array_access_test")
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


def test_fortran_frontend_array_ranges():
    test_string = """
                    PROGRAM ranges_test
                    implicit none
                    double precision d(3,4,5)
                    CALL array_ranges_test_function(d)
                    end

                    SUBROUTINE array_ranges_test_function(d)
                    double precision d(3,4,5),e(3,4,5),f(3,4,5)

                    e(:,:,:)=1.0
                    f(:,:,:)=2.0
                    f(:,2:4,:)=3.0
                    f(1,1,:)=4.0
                    d(:,:,:)=e(:,:,:)+f(:,:,:)
                    d(1,2:4,1)=e(1,2:4,1)*10.0
                    d(1,1,1)=SUM(e(:,1,:))
                    
                    END SUBROUTINE array_ranges_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "array_access_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 15)
    assert (d[0, 1, 0] == 10)
    assert (d[1, 0, 0] == 3)
    assert (d[2, 3, 3] == 4)
    assert (d[0, 0, 2] == 5)


def test_fortran_frontend_array_3dmap():
    test_string = """
                    PROGRAM array_3dmap_test
                    implicit none
                    double precision d(4,4,4)
                    CALL array_3dmap_test_function(d)
                    end

                    SUBROUTINE array_3dmap_test_function(d)
                    double precision d(4,4,4)

                    d(:,:,:)=7
                    
                    END SUBROUTINE array_3dmap_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "array_3dmap_test")
    sdfg.simplify(verbose=True)
    sdutil.normalize_offsets(sdfg)
    from dace.transformation.auto import auto_optimize as aopt
    aopt.auto_optimize(sdfg, dtypes.DeviceType.CPU)
    a = np.full([4, 4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0, 0] == 7)
    assert (a[3, 3, 3] == 7)


def test_fortran_frontend_twoconnector():
    test_string = """
                    PROGRAM twoconnector_test
                    implicit none
                    double precision d(4)
                    CALL twoconnector_test_function(d)
                    end

                    SUBROUTINE twoconnector_test_function(d)
                    double precision d(4)

                    d(2)=d(1)+d(3)
                    
                    END SUBROUTINE twoconnector_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "twoconnector_test")
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 84)
    assert (a[2] == 42)


def test_fortran_frontend_input_output_connector():
    test_string = """
                    PROGRAM ioc_test
                    implicit none
                    double precision d(2,3)
                    CALL ioc_test_function(d)
                    end

                    SUBROUTINE ioc_test_function(d)
                    double precision d(2,3)
                    integer a,b

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=d(1,1)+5
                    
                    END SUBROUTINE ioc_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "ioc_test")
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


if __name__ == "__main__":

    test_fortran_frontend_array_3dmap()
    test_fortran_frontend_array_access()
    test_fortran_frontend_input_output_connector()
    test_fortran_frontend_array_ranges()
    test_fortran_frontend_twoconnector()
