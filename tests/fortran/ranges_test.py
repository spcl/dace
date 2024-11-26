# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

"""
We test for the following patterns:
* Range 'ALL'
* selecting one element by constant
* selecting one element by variable
* selecting a subset (proper range) through constants
* selecting a subset (proper range) through variables
* ECRAD patterns (WiP)
  flux_dn(:,1:i_cloud_top) = flux_dn_clear(:,1:i_cloud_top)
* Extended ECRAD pattern with different loop starting positions.
* Arrays with offsets
* Arrays with ALL range and offsets (WiP)
"""

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
    for idx, val in enumerate(res):
        assert val == -3.0

def test_fortran_frontend_multiple_ranges_subset_var():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_subset_var
                    implicit none
                    double precision, dimension(9) :: input1
                    double precision, dimension(3) :: res
                    integer, dimension(4) :: pos
                    CALL multiple_ranges_subset_var_function(input1, res, pos)
                    end

                    SUBROUTINE multiple_ranges_subset_var_function(input1, res, pos)
                    double precision, dimension(9) :: input1
                    double precision, dimension(3) :: res
                    integer, dimension(4) :: pos

                    res(:) = input1(pos(1):pos(2)) - input1(pos(3):pos(4))

                    END SUBROUTINE multiple_ranges_subset_var_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_subset_var", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 9
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = 2 ** i

    pos = np.full([4], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 4
    pos[2] = 6
    pos[3] = 8

    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res, outside_init=False)

    for i in range(len(res)):
        assert res[i] == input1[pos[0] - 1 + i] - input1[pos[2] - 1 + i]

def test_fortran_frontend_multiple_ranges_ecrad_pattern():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_ecrad
                    implicit none
                    double precision, dimension(7, 7) :: input1
                    double precision, dimension(7, 7) :: res
                    integer, dimension(2) :: pos
                    CALL multiple_ranges_ecrad_function(input1, res, pos)
                    end

                    SUBROUTINE multiple_ranges_ecrad_function(input1, res, pos)
                    double precision, dimension(7, 7) :: input1
                    double precision, dimension(7, 7) :: res
                    integer, dimension(2) :: pos

                    res(:, pos(1):pos(2)) = input1(:, pos(1):pos(2)) 

                    END SUBROUTINE multiple_ranges_ecrad_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_ecrad", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2 ** j

    pos = np.full([2], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 5

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res, outside_init=False)

    for i in range(size):
        for j in range(pos[0], pos[1] + 1):

            print(i , j, res[i - 1, j - 1], input1[i - 1, j - 1])
            assert res[i - 1, j - 1] == input1[i - 1, j - 1]

def test_fortran_frontend_multiple_ranges_ecrad_pattern_complex():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_ecrad
                    implicit none
                    double precision, dimension(7, 7) :: input1
                    double precision, dimension(7, 7) :: res
                    integer, dimension(6) :: pos
                    CALL multiple_ranges_ecrad_function(input1, res, pos)
                    end

                    SUBROUTINE multiple_ranges_ecrad_function(input1, res, pos)
                    double precision, dimension(7, 7) :: input1
                    double precision, dimension(7, 7) :: res
                    integer, dimension(6) :: pos

                    res(:, pos(1):pos(2)) = input1(:, pos(3):pos(4)) + input1(:, pos(5):pos(6))

                    END SUBROUTINE multiple_ranges_ecrad_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_ecrad", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2 ** j

    pos = np.full([6], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 5
    pos[2] = 1
    pos[3] = 4
    pos[4] = 4
    pos[5] = 7

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res, outside_init=False)

    iter_1 = pos[0]
    iter_2 = pos[2]
    iter_3 = pos[4]
    length = pos[1] - pos[0] + 1

    for i in range(size):
        for j in range(length):
            assert res[i - 1, iter_1 + j - 1] == input1[i - 1, iter_2 + j - 1] + input1[i - 1, iter_3 + j - 1]

def test_fortran_frontend_multiple_ranges_ecrad_pattern_complex_offsets():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_ecrad_offset
                    implicit none
                    double precision, dimension(7, 21:27) :: input1
                    double precision, dimension(7, 31:37) :: res
                    integer, dimension(6) :: pos
                    CALL multiple_ranges_ecrad_function(input1, res, pos)
                    end

                    SUBROUTINE multiple_ranges_ecrad_function(input1, res, pos)
                    double precision, dimension(7, 21:27) :: input1
                    double precision, dimension(7, 31:37) :: res
                    integer, dimension(6) :: pos

                    res(:, pos(1):pos(2)) = input1(:, pos(3):pos(4)) + input1(:, pos(5):pos(6))

                    END SUBROUTINE multiple_ranges_ecrad_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_ecrad_offset", True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2 ** j

    pos = np.full([6], 0, order="F", dtype=np.int32)
    pos[0] = 2 + 30
    pos[1] = 5 + 30
    pos[2] = 1 + 20
    pos[3] = 4 + 20
    pos[4] = 4 + 20
    pos[5] = 7 + 20

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res, outside_init=False)

    iter_1 = pos[0] - 30
    iter_2 = pos[2] - 20
    iter_3 = pos[4] - 20
    length = pos[1] - pos[0] + 1

    for i in range(size):
        for j in range(length):
            assert res[i - 1, iter_1 + j - 1] == input1[i - 1, iter_2 + j - 1] + input1[i - 1, iter_3 + j - 1]

if __name__ == "__main__":

    test_fortran_frontend_multiple_ranges_all()
    test_fortran_frontend_multiple_ranges_selection()
    test_fortran_frontend_multiple_ranges_selection_var()
    test_fortran_frontend_multiple_ranges_subset()
    test_fortran_frontend_multiple_ranges_subset_var()
    test_fortran_frontend_multiple_ranges_ecrad_pattern()
    test_fortran_frontend_multiple_ranges_ecrad_pattern_complex()
    test_fortran_frontend_multiple_ranges_ecrad_pattern_complex_offsets()
