# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

# We test for the following patterns:
# * Range 'ALL'
# * selecting one element by constant
# * selecting one element by variable
# * selecting a subset (proper range) through constants (WiP)
# * selecting a subset (proper range) through variables (WiP)
# * ECRAD patterns (WiP)
# * Arrays with offsets (WiP)

def test_fortran_frontend_multiple_ranges_all():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges 
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: res
                    CALL multiple_ranges_function(input1, input2, res)
                    end

                    SUBROUTINE multiple_ranges_function(input1, input2, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(7) :: input2
                    double precision, dimension(7) :: res

                    res(:) = input1(:) - input2(:)

                    END SUBROUTINE multiple_ranges_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_function", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    input2 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 1
        input2[i] = i
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, input2=input2, res=res)
    print(res)
    for val in res:
        assert val == 1.0

def test_fortran_frontend_multiple_ranges_selection():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_selection
                    implicit none
                    double precision, dimension(7,2) :: input1
                    double precision, dimension(7) :: res
                    CALL multiple_ranges_selection_function(input1, res)
                    end

                    SUBROUTINE multiple_ranges_selection_function(input1, res)
                    double precision, dimension(7,2) :: input1
                    double precision, dimension(7) :: res

                    res(:) = input1(:, 1) - input1(:, 2)

                    END SUBROUTINE multiple_ranges_selection_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_selection", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 2
    input1 = np.full([size, size2], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i][0] = i + 1
        input1[i][1] = 0
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res, outside_init=False)
    for idx, val in enumerate(res):
        assert val == idx + 1.0

def test_fortran_frontend_multiple_ranges_selection_var():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_selection
                    implicit none
                    double precision, dimension(7,2) :: input1
                    double precision, dimension(7) :: res
                    integer :: pos1
                    integer :: pos2
                    CALL multiple_ranges_selection_function(input1, res, pos1, pos2)
                    end

                    SUBROUTINE multiple_ranges_selection_function(input1, res, pos1, pos2)
                    double precision, dimension(7,2) :: input1
                    double precision, dimension(7) :: res
                    integer :: pos1
                    integer :: pos2

                    res(:) = input1(:, pos1) - input1(:, pos2)

                    END SUBROUTINE multiple_ranges_selection_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_selection", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 2
    input1 = np.full([size, size2], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i][1] = i + 1
        input1[i][0] = 0
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res, pos1=2, pos2=1, outside_init=False)
    for idx, val in enumerate(res):
        assert val == idx + 1.0

    sdfg(input1=input1, res=res, pos1=1, pos2=2, outside_init=False)
    for idx, val in enumerate(res):
        assert -val == idx + 1.0

def test_fortran_frontend_multiple_ranges_subset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_subset
                    implicit none
                    double precision, dimension(7) :: input1
                    double precision, dimension(3) :: res
                    CALL multiple_ranges_subset_function(input1, res)
                    end

                    SUBROUTINE multiple_ranges_subset_function(input1, res)
                    double precision, dimension(7) :: input1
                    double precision, dimension(3) :: res

                    res(:) = input1(1:3) - input1(4:6)

                    END SUBROUTINE multiple_ranges_subset_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_subset", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()
    sdfg.save('test.sdfg')

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res, outside_init=False)
    print(res)
    for idx, val in enumerate(res):
        assert val == -3.0

if __name__ == "__main__":

    test_fortran_frontend_multiple_ranges_all()
    test_fortran_frontend_multiple_ranges_selection()
    test_fortran_frontend_multiple_ranges_selection_var()
    test_fortran_frontend_multiple_ranges_subset()
