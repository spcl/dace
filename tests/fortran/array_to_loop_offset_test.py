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
                    integer :: i

                    do i=1,5
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", False)
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

def test_fortran_frontend_arr2loop_1d_offset():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(2:6) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(2:6) :: d

                    d(:) = 5

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5

    a = np.full([6], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert a[5] == 42
    for i in range(0,4):
        assert a[i] == 5

def test_fortran_frontend_arr2loop_2d_offset():
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
                    integer :: i

                    do i=1,5
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,9], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1,6):
        for j in range(1,3):
            assert a[i-1, j-1] == i * 2

def test_fortran_frontend_arr2loop_2d_offset2():
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

                    d(:,:) = 43

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,9], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1,6):
        for j in range(1,3):
            assert a[i-1, j-1] == 43

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    a = np.full([5,3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0,5):
        for j in range(0,3):
            assert a[i, j] == 43

def test_fortran_frontend_arr2loop_2d_offset3():
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

                    d(2:4, 7:8) = 43

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,9], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(2,4):
        for j in range(1,3):
            assert a[i-1, j-1] == 43
        for j in range(4,5):
            assert a[i-1, j-1] == 42

    for i in [1, 5]:
        for j in range(4,8):
            assert a[i-1, j-1] == 42

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    a = np.full([5,3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1,4):
        for j in range(0,2):
            assert a[i, j] == 43
        for j in range(2,3):
            assert a[i, j] == 42

    for i in [0, 4]:
        for j in range(0,3):
            assert a[i, j] == 42

if __name__ == "__main__":

    test_fortran_frontend_arr2loop_1d_offset()
    test_fortran_frontend_arr2loop_2d_offset()
    test_fortran_frontend_arr2loop_2d_offset2()
    test_fortran_frontend_arr2loop_2d_offset3()
    test_fortran_frontend_arr2loop_without_offset()
