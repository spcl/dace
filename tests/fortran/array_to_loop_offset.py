# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

def test_fortran_frontend_arr2loop_without_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,3) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5,3) :: d

                    do i=1,5
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,9], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1,6):
        for j in range(1,4):
            assert a[i-1, j-1] == i * 2

def test_fortran_frontend_arr2loop_with_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,7:9) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5,7:9) :: d

                    do i=1,5
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,9], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1,6):
        for j in range(7,10):
            assert a[i-1, j-1] == i * 2

if __name__ == "__main__":

    test_fortran_frontend_arr2loop_with_offset()
    test_fortran_frontend_arr2loop_without_offset()
