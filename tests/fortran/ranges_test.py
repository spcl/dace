# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

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

def test_fortran_frontend_multiple_ranges_subset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM multiple_ranges_subset
                    implicit none
                    double precision, dimension(7,2) :: input1
                    double precision, dimension(7) :: res
                    CALL multiple_ranges_subset_function(input1, res)
                    end

                    SUBROUTINE multiple_ranges_subset_function(input1, res)
                    double precision, dimension(7,2) :: input1
                    double precision, dimension(7) :: res

                    res(:) = input1(:, 1) - input1(:, 2)

                    END SUBROUTINE multiple_ranges_subset_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "multiple_ranges_subset", True)
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

if __name__ == "__main__":

    test_fortran_frontend_multiple_ranges_all()
    test_fortran_frontend_multiple_ranges_subset()
