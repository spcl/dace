# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

from tests.fortran.fortran_test_helper import SourceCodeBuilder
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string


def test_fortran_frontend_merge_1d():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, mask, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, mask, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    double precision, dimension(7) :: res

                    res = MERGE(input1, input2, mask)

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, res=res)
    for val in res:
        assert val == 42

    for i in range(int(size / 2)):
        mask[i] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    for i in range(int(size / 2)):
        assert res[i] == 13
    for i in range(int(size / 2), size):
        assert res[i] == 42

    mask[:] = 0
    for i in range(size):
        if i % 2 == 1:
            mask[i] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    for i in range(size):
        if i % 2 == 1:
            assert res[i] == 13
        else:
            assert res[i] == 42


def test_fortran_frontend_merge_comparison_scalar():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: res

                    res = MERGE(input1, input2, input1 .eq. 3)

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, res=res)
    for val in res:
        assert val == 42

    for i in range(int(size / 2)):
        first[i] = 3
    sdfg(input1=first, input2=second, res=res)
    for i in range(int(size / 2)):
        assert res[i] == 3
    for i in range(int(size / 2), size):
        assert res[i] == 42

    first[:] = 13
    for i in range(size):
        if i % 2 == 1:
            first[i] = 3
    sdfg(input1=first, input2=second, res=res)
    for i in range(size):
        if i % 2 == 1:
            assert res[i] == 3
        else:
            assert res[i] == 42


def test_fortran_frontend_merge_comparison_arrays():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: res

                    res = MERGE(input1, input2, input1 .lt. input2)

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, res=res)
    for val in res:
        assert val == 13

    for i in range(int(size / 2)):
        first[i] = 45
    sdfg(input1=first, input2=second, res=res)
    for i in range(int(size / 2)):
        assert res[i] == 42
    for i in range(int(size / 2), size):
        assert res[i] == 13

    first[:] = 13
    for i in range(size):
        if i % 2 == 1:
            first[i] = 45
    sdfg(input1=first, input2=second, res=res)
    for i in range(size):
        if i % 2 == 1:
            assert res[i] == 42
        else:
            assert res[i] == 13


def test_fortran_frontend_merge_comparison_arrays_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(14) :: mask1
                    double precision, dimension(14) :: mask2
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, mask1, mask2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, mask1, mask2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(14) :: mask1
                    double precision, dimension(14) :: mask2
                    double precision, dimension(7) :: res

                    res = MERGE(input1, input2, mask1(3:9) .lt. mask2(5:11))

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask1 = np.full([size * 2], 30, order="F", dtype=np.float64)
    mask2 = np.full([size * 2], 0, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    mask1[2:9] = 3
    mask2[4:11] = 4
    sdfg(input1=first, input2=second, mask1=mask1, mask2=mask2, res=res)
    for val in res:
        assert val == 13


def test_fortran_frontend_merge_array_shift():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(21) :: input2
                    double precision, dimension(14) :: mask1
                    double precision, dimension(14) :: mask2
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, mask1, mask2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, mask1, mask2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(21) :: input2
                    double precision, dimension(14) :: mask1
                    double precision, dimension(14) :: mask2
                    double precision, dimension(7) :: res

                    res = MERGE(input1, input2(13:19), mask1(3:9) .gt. mask2(5:11))

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    # sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size * 3], 42, order="F", dtype=np.float64)
    mask1 = np.full([size * 2], 30, order="F", dtype=np.float64)
    mask2 = np.full([size * 2], 0, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    second[12:19] = 100
    mask1[2:9] = 3
    mask2[4:11] = 4
    sdfg(input1=first, input2=second, mask1=mask1, mask2=mask2, res=res)
    for val in res:
        assert val == 100


def test_fortran_frontend_merge_nonarray():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    logical :: val(2)
                    double precision :: res(2)
                    CALL merge_test_function(val, res)
                    end

                    SUBROUTINE merge_test_function(val, res)
                    logical :: val(2)
                    double precision :: res(2)
                    double precision :: input1
                    double precision :: input2

                    input1 = 1
                    input2 = 5

                    res(1) = MERGE(input1, input2, val(1))

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    val = np.full([1], 1, order="F", dtype=np.int32)
    res = np.full([1], 40, order="F", dtype=np.float64)

    sdfg(val=val, res=res)
    assert res[0] == 1

    val[0] = 0
    sdfg(val=val, res=res)
    assert res[0] == 5


def test_fortran_frontend_merge_recursive():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: input3
                    integer, dimension(7) :: mask1
                    integer, dimension(7) :: mask2
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, input3, mask1, mask2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, input3, mask1, mask2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: input3
                    integer, dimension(7) :: mask1
                    integer, dimension(7) :: mask2
                    double precision, dimension(7) :: res

                    res = MERGE(MERGE(input1, input2, mask1), input3, mask2)

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    third = np.full([size], 43, order="F", dtype=np.float64)
    mask1 = np.full([size], 0, order="F", dtype=np.int32)
    mask2 = np.full([size], 1, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    for i in range(int(size / 2)):
        mask1[i] = 1

    mask2[-1] = 0

    sdfg(input1=first, input2=second, input3=third, mask1=mask1, mask2=mask2, res=res)

    assert np.allclose(res, [13, 13, 13, 42, 42, 42, 43])


def test_fortran_frontend_merge_scalar():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, mask, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, mask, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    double precision, dimension(7) :: res

                    res(1) = MERGE(input1(1), input2(1), mask(1))

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    # sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, res=res)

    assert res[0] == 42
    for val in res[1:]:
        assert val == 40

    mask[0] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    assert res[0] == 13
    for val in res[1:]:
        assert val == 40


def test_fortran_frontend_merge_scalar2():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, mask, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, mask, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    double precision, dimension(7) :: res

                    res(1) = MERGE(input1(1), 0.0, mask(1))

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    # sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, res=res)
    assert res[0] == 0

    mask[:] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    assert res[0] == 13


def test_fortran_frontend_merge_scalar3():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    integer, dimension(7) :: mask2
                    double precision, dimension(7) :: res
                    CALL merge_test_function(input1, input2, mask, mask2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, mask, mask2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    integer, dimension(7) :: mask
                    integer, dimension(7) :: mask2
                    double precision, dimension(7) :: res

                    res(1) = MERGE(input1(1), 0.0, mask(1) > mask2(1) .AND. mask2(2) == 0)

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    # sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    mask2 = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, mask2=mask2, res=res)
    assert res[0] == 0

    mask[:] = 1
    sdfg(input1=first, input2=second, mask=mask, mask2=mask2, res=res)
    assert res[0] == 13


def test_fortran_frontend_merge_literal():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM merge_test
                    implicit none
                    double precision :: input1
                    double precision :: input2
                    double precision, dimension(1) :: res
                    CALL merge_test_function(input1, input2, res)
                    end

                    SUBROUTINE merge_test_function(input1, input2, res)
                    double precision :: input1
                    double precision :: input2
                    double precision, dimension(1) :: res

                    res = MERGE(1.0D0, input2, input2 .lt. 3)

                    END SUBROUTINE merge_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    # Minimum is in the beginning
    res = np.full([1], 40, order="F", dtype=np.float64)

    sdfg(input1=13, input2=42, res=res)
    assert res[0] == 42

    sdfg(input1=13, input2=10, res=res)
    assert res[0] == 10

    sdfg(input1=13, input2=2, res=res)
    assert res[0] == 1


def test_fortran_frontend_merge_dataref():
    sources, main = (
        SourceCodeBuilder()
        .add_file(
            """
                    module lib
                      implicit none
                      type test_type
                          double precision, dimension(3) :: input_data
                          double precision, dimension(3) :: input_data_second
                      end type

                      type test_type2
                          type(test_type) :: var
                      end type
                    end module lib

                    MODULE test_merge

                        contains

                        SUBROUTINE merge_test(input1, input2, res)
                        use lib, only: test_type2
                        implicit none

                        type(test_type2) :: data
                        double precision, dimension(3) :: input1
                        double precision, dimension(3) :: input2
                        double precision, dimension(3) :: res

                        data%var%input_data = input1
                        data%var%input_data_second = input2

                        CALL merge_test_function(data, res)
                        end SUBROUTINE merge_test

                        SUBROUTINE merge_test_function(data, res)
                        use lib, only: test_type2
                        implicit none

                        double precision, dimension(3) :: res
                        type(test_type2) :: data

                        res = MERGE(data%var%input_data, data%var%input_data_second, data%var%input_data .lt. 3)
                        !res = MERGE(data%var%input_data, data%var%input_data_second, 1 .lt. 3)

                        END SUBROUTINE merge_test_function

                    END MODULE
    """,
            "main",
        )
        .check_with_gfortran()
        .get()
    )

    sdfg = create_singular_sdfg_from_string(sources, "test_merge.merge_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    # Minimum is in the beginning
    data1 = np.full([3], 42, order="F", dtype=np.float64)
    data2 = np.full([3], 40, order="F", dtype=np.float64)
    res = np.full([3], 0, order="F", dtype=np.float64)

    sdfg(input1=data1, input2=data2, res=res)
    print(res)
    assert res[0] == 42

    sdfg(input1=13, input2=10, res=res)
    assert res[0] == 10

    sdfg(input1=13, input2=2, res=res)
    assert res[0] == 1


if __name__ == "__main__":
    test_fortran_frontend_merge_1d()
    test_fortran_frontend_merge_comparison_scalar()
    test_fortran_frontend_merge_comparison_arrays()
    test_fortran_frontend_merge_comparison_arrays_offset()
    test_fortran_frontend_merge_array_shift()
    test_fortran_frontend_merge_nonarray()
    test_fortran_frontend_merge_recursive()
    test_fortran_frontend_merge_scalar()
    test_fortran_frontend_merge_scalar2()
    test_fortran_frontend_merge_scalar3()
    test_fortran_frontend_merge_literal()
    test_fortran_frontend_merge_dataref()
