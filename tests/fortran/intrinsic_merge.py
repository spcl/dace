# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

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

    for i in range(int(size/2)):
        mask[i] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    for i in range(int(size/2)):
        assert res[i] == 13
    for i in range(int(size/2), size):
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

    for i in range(int(size/2)):
        first[i] = 3
    sdfg(input1=first, input2=second, res=res)
    for i in range(int(size/2)):
        assert res[i] == 3
    for i in range(int(size/2), size):
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

    for i in range(int(size/2)):
        first[i] = 45
    sdfg(input1=first, input2=second, res=res)
    for i in range(int(size/2)):
        assert res[i] == 42
    for i in range(int(size/2), size):
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

    # mask comparison on array participating
    # mask comparison on two arrays participating
    # mask comparison - second array with shift
    # mask comparison - both arrays wiht a shift
    # second array - shift!
    # merge 2d

if __name__ == "__main__":

    test_fortran_frontend_merge_1d()
    test_fortran_frontend_merge_comparison_scalar()
    test_fortran_frontend_merge_comparison_arrays()
