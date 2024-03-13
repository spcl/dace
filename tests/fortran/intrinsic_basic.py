# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_bit_size():
    test_string = """
                    PROGRAM intrinsic_math_test_bit_size
                    implicit none
                    integer, dimension(4) :: res
                    CALL intrinsic_math_test_function(res)
                    end

                    SUBROUTINE intrinsic_math_test_function(res)
                    integer, dimension(4) :: res
                    logical :: a = .TRUE.
                    integer :: b = 1
                    real :: c = 1
                    double precision :: d = 1

                    res(1) = BIT_SIZE(a)
                    res(2) = BIT_SIZE(b)
                    res(3) = BIT_SIZE(c)
                    res(4) = BIT_SIZE(d)

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_bit_size", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [32, 32, 32, 64])

def test_fortran_frontend_bit_size_symbolic():
    test_string = """
                    PROGRAM intrinsic_math_test_bit_size
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: arrsize3
                    integer :: res(arrsize)
                    integer :: res2(arrsize, arrsize2, arrsize3)
                    integer :: res3(arrsize+arrsize2, arrsize2 * 5, arrsize3 + arrsize2*arrsize)
                    CALL intrinsic_math_test_function(arrsize, arrsize2, arrsize3, res, res2, res3)
                    end

                    SUBROUTINE intrinsic_math_test_function(arrsize, arrsize2, arrsize3, res, res2, res3)
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: arrsize3
                    integer :: res(arrsize)
                    integer :: res2(arrsize, arrsize2, arrsize3)
                    integer :: res3(arrsize+arrsize2, arrsize2 * 5, arrsize3 + arrsize2*arrsize)

                    res(1) = SIZE(res)
                    res(2) = SIZE(res2)
                    res(3) = SIZE(res3)
                    res(4) = SIZE(res)*2
                    res(5) = SIZE(res)*SIZE(res2)*SIZE(res3)
                    res(6) = SIZE(res2, 1) + SIZE(res2, 2) + SIZE(res2, 3)
                    res(7) = SIZE(res3, 1) + SIZE(res3, 2) + SIZE(res3, 3)

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_bit_size", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 24
    size2 = 5
    size3 = 7
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size, size2, size3], 42, order="F", dtype=np.int32)
    res3 = np.full([size+size2, size2*5, size3 + size*size2], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, res3=res3, arrsize=size, arrsize2=size2, arrsize3=size3)

    assert res[0] == size
    assert res[1] == size*size2*size3
    assert res[2] == (size + size2) * (size2 * 5) * (size3 + size2*size)
    assert res[3] == size * 2 
    assert res[4] == res[0] * res[1] * res[2]
    assert res[5] == size + size2 + size3
    assert res[6] == size + size2 + size2*5 + size3 + size*size2

def test_fortran_frontend_size_arbitrary():
    test_string = """
                    PROGRAM intrinsic_basic_size_arbitrary
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: res(arrsize, arrsize2)
                    CALL intrinsic_basic_size_arbitrary_test_function(res)
                    end

                    SUBROUTINE intrinsic_basic_size_arbitrary_test_function(res)
                    implicit none
                    integer :: res(:, :)

                    res(1,1) = SIZE(res)
                    res(2,1) = SIZE(res, 1)
                    res(3,1) = SIZE(res, 2)

                    END SUBROUTINE intrinsic_basic_size_arbitrary_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_basic_size_arbitrary_test", True,)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 5
    res = np.full([size, size2], 42, order="F", dtype=np.int32)
    sdfg(res=res,arrsize=size,arrsize2=size2)

    assert res[0,0] == size*size2
    assert res[1,0] == size
    assert res[2,0] == size2

def test_fortran_frontend_ubound_assumed_1d():
    test_string = """
                    PROGRAM intrinsic_basic_ubound_arbitrary
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: res(arrsize:arrsize2)
                    CALL intrinsic_basic_ubound_arbitrary_test_function(res)
                    end

                    SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function(res)
                    implicit none
                    integer :: res(:)
                    integer :: lboundvar

                    lboundvar = LBOUND(res, 1)
                    res(lboundvar) = lboundvar
                    res(lboundvar+1) = UBOUND(res, 1)

                    END SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_basic_ubound_arbitrary_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 10
    res = np.full([10], 42, order="F", dtype=np.int32)
    sdfg(res=res,arrsize=size,arrsize2=size2)

    assert res[0] == 1
    assert res[1] == size2 - size + 1

def test_fortran_frontend_ubound_assumed_1d_ptr():
    test_string = """
                    PROGRAM intrinsic_basic_ubound_arbitrary
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: res(arrsize:arrsize2)
                    CALL intrinsic_basic_ubound_arbitrary_test_function(res, arrsize, arrsize2)
                    end

                    SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function(res, arrsize, arrsize2)
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer, target :: res(arrsize:arrsize2)
                    integer, pointer, dimension(:) :: ptr

                    ptr => res

                    CALL intrinsic_basic_ubound_arbitrary_test_function2(ptr)

                    END SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function

                    SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function2(res)
                    implicit none
                    integer, pointer, dimension(:) :: res
                    integer :: lboundvar

                    lboundvar = LBOUND(res, 1)
                    res(lboundvar) = lboundvar
                    res(lboundvar+1) = UBOUND(res, 1)

                    END SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function2
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_basic_ubound_arbitrary_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 10
    res = np.full([10], 42, order="F", dtype=np.int32)
    sdfg(res=res,arrsize=size,arrsize2=size2)
    print(res)

    assert res[0] == size
    assert res[1] == size2

def test_fortran_frontend_ubound_assumed_1d_struct_ptr():
    test_string = """
                    PROGRAM intrinsic_basic_ubound_arbitrary
                    implicit none

                    type a_t
                        integer, pointer, dimension(:) :: w
                    end type a_t

                    integer :: arrsize
                    integer :: arrsize2
                    integer :: res(arrsize:arrsize2)
                    CALL intrinsic_basic_ubound_arbitrary_test_function(res, arrsize, arrsize2)
                    end

                    SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function(res, arrsize, arrsize2)
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer, target :: res(arrsize:arrsize2)
                    integer, pointer :: ptr(:)
                    type(a_t) :: data
                    integer :: lboundvar

                    data%w=>res

                    ptr=>data%w

                    lboundvar = LBOUND(res, 1)
                    !ptr(lboundvar) = lboundvar
                    !ptr(lboundvar+1) = UBOUND(res, 1)

                    !res = data%w

                    END SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_basic_ubound_arbitrary_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 10
    res = np.full([10], 42, order="F", dtype=np.int32)
    sdfg(res=res,arrsize=size,arrsize2=size2)
    print(res)

    assert res[0] == size
    assert res[1] == size2

def test_fortran_frontend_ubound_1d():
    test_string = """
                    PROGRAM intrinsic_basic_ubound_arbitrary
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: res(arrsize:arrsize2)
                    CALL intrinsic_basic_ubound_arbitrary_test_function(res, arrsize, arrsize2)
                    end

                    SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function(res, arrsize, arrsize2)
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: res(arrsize:arrsize2)
                    integer :: lboundvar

                    lboundvar = LBOUND(res, 1)
                    res(lboundvar) = lboundvar
                    res(lboundvar+1) = UBOUND(res, 1)

                    END SUBROUTINE intrinsic_basic_ubound_arbitrary_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_basic_ubound_arbitrary_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    size2 = 10
    res = np.full([10], 42, order="F", dtype=np.int32)
    sdfg(res=res,arrsize=size,arrsize2=size2)

    assert res[0] == size
    assert res[1] == size2

def test_fortran_frontend_present():
    test_string = """
                    PROGRAM intrinsic_basic_present
                    implicit none
                    integer, dimension(4) :: res
                    integer, dimension(4) :: res2
                    integer :: a
                    CALL intrinsic_basic_present_test_function(res, res2, a)
                    end

                    SUBROUTINE intrinsic_basic_present_test_function(res, res2, a)
                    integer, dimension(4) :: res
                    integer, dimension(4) :: res2
                    integer :: a

                    CALL tf2(res, a=a)
                    CALL tf2(res2)

                    END SUBROUTINE intrinsic_basic_present_test_function

                    SUBROUTINE tf2(res, a)
                    integer, dimension(4) :: res
                    integer, optional :: a

                    res(1) = PRESENT(a)

                    END SUBROUTINE tf2
                    """
    #test_string = """
    #                PROGRAM intrinsic_basic_present
    #                implicit none
    #                integer, dimension(4) :: res
    #                integer, dimension(4) :: res2
    #                integer :: a
    #                CALL test_intrinsic_basic_pre2sent_function(res, res2,a)
    #                end

    #                SUBROUTINE test_intrinsic_basic_pre2sent_function(res, res2,a)
    #                integer, dimension(4) :: res
    #                integer, dimension(4) :: res2
    #                integer :: a

    #                res(1) = 1
    #                END SUBROUTINE test_intrinsic_basic_pre2sent_function
    #                !PROGRAM intrinsic_basic_present
    #                !implicit none
    #                !integer, dimension(4) :: res
    #                !integer, dimension(4) :: res2
    #                !integer :: a
    #                !CALL intrinsic_basic_present_function(res, res2, a)
    #                !end

    #                !SUBROUTINE intrinsic_basic_present_function(res, res2, a)
    #                !integer, dimension(4) :: res
    #                !integer, dimension(4) :: res2
    #                !integer :: a

    #                !res(1) = 5
    #                !!CALL intrinsic_basic_present_function2(res, a)
    #                !!CALL intrinsic_basic_present_function2(res2)

    #                !END SUBROUTINE intrinsic_basic_present_function

    #                !SUBROUTINE intrinsic_basic_present_function2(res, a)
    #                !integer, dimension(4) :: res
    #                !integer :: a

    #                !!res(1) = PRESENT(a)
    #                !res(1) = 2
    #                !res(2) = 2

    #                !END SUBROUTINE intrinsic_basic_present_function2
    #                """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_basic_present_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5)

    assert res[0] == 1
    assert res2[0] == 0

if __name__ == "__main__":

    test_fortran_frontend_bit_size()
    test_fortran_frontend_bit_size_symbolic()
    test_fortran_frontend_size_arbitrary()
    test_fortran_frontend_present()

    test_fortran_frontend_ubound_1d()
    test_fortran_frontend_ubound_assumed_1d()

    # This does not work - we do not support all types of pointers
    #test_fortran_frontend_ubound_assumed_1d_ptr()

    # This does not work - we do not support assigning pointer to a local array
    #test_fortran_frontend_ubound_assumed_1d_struct_ptr()

