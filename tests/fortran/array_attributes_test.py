# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_array_attribute_no_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5) :: d

                    do i=1,5
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 5):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_no_offset_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    double precision, dimension(arrsize) :: d
                    CALL index_test_function(d,arrsize)
                    end

                    SUBROUTINE index_test_function(d, arrsize)
                    integer :: arrsize
                    double precision, dimension(arrsize) :: d

                    do i=1,arrsize
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    from dace.symbolic import symbol
    assert isinstance(sdfg.data('d').shape[0], symbol)
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    size = 10
    a = np.full([size], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=size)
    for i in range(1, size):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(50:54) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(50:54) :: d

                    do i=50,54
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(50, 54):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_offset_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    double precision, dimension(arrsize:arrsize+4) :: d
                    CALL index_test_function(d, arrsize)
                    end

                    SUBROUTINE index_test_function(d, arrsize)
                    integer :: arrsize
                    double precision, dimension(arrsize:arrsize+4) :: d

                    do i=arrsize, arrsize+4
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    arrsize = 50
    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize)
    for i in range(arrsize, arrsize + 4):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_offset_symbol2():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    Compared to the previous one, this one should prevent simplification from removing symbols
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    double precision, dimension(arrsize:arrsize2) :: d
                    CALL index_test_function(d, arrsize, arrsize2)
                    end

                    SUBROUTINE index_test_function(d, arrsize, arrsize2)
                    integer :: arrsize
                    integer :: arrsize2
                    double precision, dimension(arrsize:arrsize2) :: d

                    do i=arrsize, arrsize2
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    from dace.symbolic import evaluate

    arrsize = 50
    arrsize2 = 54
    assert len(sdfg.data('d').shape) == 1
    assert evaluate(sdfg.data('d').shape[0], {'arrsize': arrsize, 'arrsize2': arrsize2}) == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(arrsize, arrsize2):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision d(50:54)
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision d(50:54)

                    do i=50,54
                       d(i) = i * 2.0
                    end do
                    
                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(50, 54):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_offset_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    Compared to the previous one, this one should prevent simplification from removing symbols
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    double precision :: d(arrsize:arrsize2)
                    CALL index_test_function(d, arrsize, arrsize2)
                    end

                    SUBROUTINE index_test_function(d, arrsize, arrsize2)
                    integer :: arrsize
                    integer :: arrsize2
                    double precision :: d(arrsize:arrsize2)

                    do i=arrsize, arrsize2
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    from dace.symbolic import evaluate

    arrsize = 50
    arrsize2 = 54
    assert len(sdfg.data('d').shape) == 1
    assert evaluate(sdfg.data('d').shape[0], {'arrsize': arrsize, 'arrsize2': arrsize2}) == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(arrsize, arrsize2):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_arbitrary():
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    double precision :: d(arrsize, arrsize2)
                    CALL index_test_function(d, arrsize, arrsize2)
                    end

                    SUBROUTINE index_test_function(d, arrsize, arrsize2)
                    integer :: arrsize
                    integer :: arrsize2
                    double precision :: d(:,:)

                    do i=1, arrsize
                       d(i, 1) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    arrsize = 5
    arrsize2 = 10
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(arrsize):
        # offset -1 is already added
        assert a[i, 0] == (i + 1) * 2


def test_fortran_frontend_array_arbitrary_attribute():
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    double precision :: d(arrsize, arrsize2)
                    CALL index_test_function(d, arrsize, arrsize2)
                    end

                    SUBROUTINE index_test_function(d, arrsize, arrsize2)
                    integer :: arrsize
                    integer :: arrsize2
                    double precision, dimension(:,:) :: d

                    do i=1, arrsize
                       d(i, 1) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    arrsize = 5
    arrsize2 = 10
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(arrsize):
        # offset -1 is already added
        assert a[i, 0] == (i + 1) * 2


def test_fortran_frontend_array_arbitrary_attribute2():
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: arrsize3
                    integer :: arrsize4
                    double precision :: d(arrsize, arrsize2)
                    double precision :: d2(arrsize3, arrsize4)
                    CALL index_test_function(d, d2)
                    end

                    SUBROUTINE index_test_function(d, d2)
                    double precision, dimension(:,:) :: d, d2

                    CALL index_test_function_second(d,d2)

                    END SUBROUTINE index_test_function

                    SUBROUTINE index_test_function_second(d, d2)
                    double precision, dimension(:,:) :: d, d2

                    d(1,1) = SIZE(d,1)
                    d(1,2) = SIZE(d,2)
                    d(1,3) = SIZE(d2,1)
                    d(1,4) = SIZE(d2,2)

                    END SUBROUTINE index_test_function_second
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    arrsize = 5
    arrsize2 = 10
    arrsize3 = 3
    arrsize4 = 7
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    b = np.full([arrsize3, arrsize4], 42, order="F", dtype=np.float64)
    sdfg(d=a, d2=b, arrsize=arrsize, arrsize2=arrsize2, arrsize3=arrsize3, arrsize4=arrsize4)
    assert a[0, 0] == arrsize
    assert a[0, 1] == arrsize2
    assert a[0, 2] == arrsize3
    assert a[0, 3] == arrsize4


if __name__ == "__main__":
    test_fortran_frontend_array_offset()
    test_fortran_frontend_array_attribute_no_offset()
    test_fortran_frontend_array_attribute_offset()
    test_fortran_frontend_array_attribute_no_offset_symbol()
    test_fortran_frontend_array_attribute_offset_symbol()
    test_fortran_frontend_array_attribute_offset_symbol2()
    test_fortran_frontend_array_offset_symbol()
    test_fortran_frontend_array_arbitrary()
    test_fortran_frontend_array_arbitrary_attribute()
    test_fortran_frontend_array_arbitrary_attribute2()
