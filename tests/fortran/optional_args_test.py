# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_optional():

    sources, main = SourceCodeBuilder().add_file(
        """

    MODULE intrinsic_optional_test
        INTERFACE
            SUBROUTINE intrinsic_optional_test_function2(res, a)
                integer, dimension(2) :: res
                integer, optional :: a
            END SUBROUTINE intrinsic_optional_test_function2
        END INTERFACE
    END MODULE

    SUBROUTINE intrinsic_optional_test_function(res, res2, a)
    USE intrinsic_optional_test
    implicit none
    integer, dimension(4) :: res
    integer, dimension(4) :: res2
    integer :: a

    CALL intrinsic_optional_test_function2(res, a)
    CALL intrinsic_optional_test_function2(res2)

    END SUBROUTINE intrinsic_optional_test_function

    SUBROUTINE intrinsic_optional_test_function2(res, a)
    integer, dimension(2) :: res
    integer, optional :: a

    res(1) = a

    END SUBROUTINE intrinsic_optional_test_function2
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'intrinsic_optional_test_function', normalize_offsets=True)
    #sdfg.simplify()
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5)

    assert res[0] == 5
    assert res2[0] == 0


def test_fortran_frontend_optional_complex():

    sources, main = SourceCodeBuilder().add_file(
        """

    MODULE intrinsic_optional_test
        INTERFACE
            SUBROUTINE intrinsic_optional_test_function2(res, a, b, c)
                integer, dimension(5) :: res
                integer, optional :: a
                double precision, optional :: b
                logical, optional :: c
            END SUBROUTINE intrinsic_optional_test_function2
        END INTERFACE
    END MODULE

    SUBROUTINE intrinsic_optional_test_function(res, res2, a, b, c)
    USE intrinsic_optional_test
    implicit none
    integer, dimension(5) :: res
    integer, dimension(5) :: res2
    integer :: a
    double precision :: b
    logical :: c

    CALL intrinsic_optional_test_function2(res, a, b)
    CALL intrinsic_optional_test_function2(res2)

    END SUBROUTINE intrinsic_optional_test_function

    SUBROUTINE intrinsic_optional_test_function2(res, a, b, c)
    integer, dimension(5) :: res
    integer, optional :: a
    double precision, optional :: b
    logical, optional :: c

    res(1) = a
    res(2) = b
    res(3) = c

    END SUBROUTINE intrinsic_optional_test_function2
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'intrinsic_optional_test_function', normalize_offsets=True)
    #sdfg.simplify()
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5, b=7, c=1)

    assert res[0] == 5
    assert res[1] == 7
    assert res[2] == 0

    assert res2[0] == 0
    assert res2[1] == 0
    assert res2[2] == 0


if __name__ == "__main__":

    test_fortran_frontend_optional()
    test_fortran_frontend_optional_complex()
