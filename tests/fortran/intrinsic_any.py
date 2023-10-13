# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_any_array():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM intrinsic_any_test
                    implicit none
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res
                    CALL intrinsic_any_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_any_test_function(d, res)
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res

                    res(1) = ANY(d)

                    !res(1) = ANY(d == .True.)
                    !d(3) = .False.
                    !res(2) = ANY(d == .True.)

                    !res(1) = ANY(d == e)
                    !d(3) = .False.
                    !res(2) = ANY(d == 

                    END SUBROUTINE intrinsic_any_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_any_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    d = np.full([size], False, order="F", dtype=np.int32)
    res = np.full([2], 42, order="F", dtype=np.int32)

    d[2] = True
    sdfg(d=d, res=res)
    assert res[0] == True

    d[2] = False
    sdfg(d=d, res=res)
    assert res[0] == False


def test_fortran_frontend_any_array_dim():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM intrinsic_any_test
                    implicit none
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res
                    CALL intrinsic_any_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_any_test_function(d, res)
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res

                    res(1) = ANY(d, 1)

                    END SUBROUTINE intrinsic_any_test_function
                    """

    with pytest.raises(NotImplementedError):
        fortran_parser.create_sdfg_from_string(test_string, "intrinsic_any_test", False)


def test_fortran_frontend_any_array_comparison():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM intrinsic_any_test
                    implicit none
                    integer, dimension(5) :: first
                    integer, dimension(5) :: second
                    logical, dimension(6) :: res
                    CALL intrinsic_any_test_function(first, second, res)
                    end

                    SUBROUTINE intrinsic_any_test_function(first, second, res)
                    integer, dimension(5) :: first
                    integer, dimension(5) :: second
                    logical, dimension(6) :: res

                    res(1) = ANY(first .eq. second)
                    !res(2) = ANY(first(:) .eq. second)
                    !res(3) = ANY(first .eq. second(:))
                    !res(4) = ANY(first(:) .eq. second(:))
                    !res(5) = any(first(1:5) .eq. second(1:5))
                    !res(6) = any(first(1:3) .eq. second(3:5))

                    END SUBROUTINE intrinsic_any_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_any_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    first = np.full([size], 1, order="F", dtype=np.int32)
    second = np.full([size], 2, order="F", dtype=np.int32)
    second[3] = 1
    res = np.full([6], 1, order="F", dtype=np.int32)

    sdfg(first=first, second=second, res=res)
    for val in res:
        assert val == True

    second = np.full([size], 2, order="F", dtype=np.int32)
    res = np.full([6], 0, order="F", dtype=np.int32)
    sdfg(first=first, second=second, res=res)
    for val in res:
        assert val == False


if __name__ == "__main__":

    test_fortran_frontend_any_array()
    test_fortran_frontend_any_array_dim()
    test_fortran_frontend_any_array_comparison()
