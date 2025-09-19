# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser


def test_fortran_frontend_sum2loop_1d_without_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(7) :: d
                    double precision, dimension(3) :: res
                    CALL index_offset_test_function(d, res)
                    end

                    SUBROUTINE index_offset_test_function(d, res)
                    double precision, dimension(7) :: d
                    double precision, dimension(3) :: res

                    res(1) = SUM(d(:))
                    res(2) = SUM(d)
                    res(3) = SUM(d(2:6))

                    END SUBROUTINE index_offset_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify()
    sdfg.compile()

    size = 7
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == (1 + size) * size / 2
    assert res[1] == (1 + size) * size / 2
    assert res[2] == (2 + size - 1) * (size - 2) / 2


def test_fortran_frontend_sum2loop_1d_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(2:6) :: d
                    double precision, dimension(3) :: res
                    CALL index_offset_test_function(d,res)
                    end

                    SUBROUTINE index_offset_test_function(d, res)
                    double precision, dimension(2:6) :: d
                    double precision, dimension(3) :: res

                    res(1) = SUM(d)
                    res(2) = SUM(d(:))
                    res(3) = SUM(d(3:5))

                    END SUBROUTINE index_offset_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == (1 + size) * size / 2
    assert res[1] == (1 + size) * size / 2
    assert res[2] == (2 + size - 1) * (size - 2) / 2


def test_fortran_frontend_arr2loop_2d():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,3) :: d
                    double precision, dimension(4) :: res
                    CALL index_offset_test_function(d,res)
                    end

                    SUBROUTINE index_offset_test_function(d, res)
                    double precision, dimension(5,3) :: d
                    double precision, dimension(4) :: res

                    res(1) = SUM(d)
                    res(2) = SUM(d(:,:))
                    res(3) = SUM(d(2:4, 2))
                    res(4) = SUM(d(2:4, 2:3))

                    END SUBROUTINE index_offset_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.simplify()
    sdfg.compile()

    sizes = [5, 3]
    d = np.full(sizes, 42, order="F", dtype=np.float64)
    cnt = 0
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            d[i, j] = cnt
            cnt += 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == 105
    assert res[1] == 105
    assert res[2] == 21
    assert res[3] == 45


def test_fortran_frontend_arr2loop_2d_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(2:6,7:10) :: d
                    double precision, dimension(3) :: res
                    CALL index_offset_test_function(d,res)
                    end

                    SUBROUTINE index_offset_test_function(d, res)
                    double precision, dimension(2:6,7:10) :: d
                    double precision, dimension(3) :: res

                    res(1) = SUM(d)
                    res(2) = SUM(d(:,:))
                    res(3) = SUM(d(3:5, 8:9))

                    END SUBROUTINE index_offset_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.simplify()
    sdfg.compile()

    sizes = [5, 4]
    d = np.full(sizes, 42, order="F", dtype=np.float64)
    cnt = 0
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            d[i, j] = cnt
            cnt += 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == 190
    assert res[1] == 190
    assert res[2] == 57


if __name__ == "__main__":

    test_fortran_frontend_sum2loop_1d_without_offset()
    test_fortran_frontend_sum2loop_1d_offset()
    test_fortran_frontend_arr2loop_2d()
    test_fortran_frontend_arr2loop_2d_offset()
