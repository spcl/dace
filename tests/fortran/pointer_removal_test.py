# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser


@pytest.mark.skip(reason="This must be rewritten to use fparser preprocessing")
def test_fortran_frontend_ptr_assignment_removal():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            INTEGER,POINTER :: tmp
            tmp=>s%a

            tmp = 13
            d(2,1) = max(1.0, tmp)
        END SUBROUTINE type_in_call_test_function
    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test")
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 13)
    assert (a[2, 0] == 42)


@pytest.mark.skip(reason="This must be rewritten to use fparser preprocessing")
def test_fortran_frontend_ptr_assignment_removal_array():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            REAL,POINTER :: tmp(:,:,:)
            tmp=>s%w

            tmp(1,1,1) = 11.0
            d(2,1) = max(1.0, tmp(1,1,1))
        END SUBROUTINE type_in_call_test_function
    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@pytest.mark.skip(reason="This must be rewritten to use fparser preprocessing")
def test_fortran_frontend_ptr_assignment_removal_array_assumed():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            REAL,POINTER :: tmp(:,:,:)
            tmp=>s%w

            tmp(1,1,1) = 11.0
            d(2,1) = max(1.0, tmp(1,1,1))

            CALL type_in_call_test_function2(tmp)
            d(3,1) = max(1.0, tmp(2,1,1))

        END SUBROUTINE type_in_call_test_function

        SUBROUTINE type_in_call_test_function2(tmp)
            REAL,POINTER :: tmp(:,:,:)

            tmp(2,1,1) = 1410
        END SUBROUTINE type_in_call_test_function2
    """
    sources = {}
    sources["type_test"] = test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test")
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 1410)


@pytest.mark.skip(reason="This must be rewritten to use fparser preprocessing")
def test_fortran_frontend_ptr_assignment_removal_array_nested():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type4
                REAL :: w(5,5,5)
            END TYPE simple_type4

            TYPE simple_type3
                type(simple_type4):: val3
            END TYPE simple_type3

            TYPE simple_type2
                type(simple_type3):: val
                REAL :: w(5,5,5)
            END TYPE simple_type2

            TYPE simple_type
                type(simple_type2) :: val1
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            REAL,POINTER :: tmp(:,:,:)
            !tmp=>s%val1%val%w
            tmp=>s%val1%w

            tmp(1,1,1) = 11.0
            d(2,1) = tmp(1,1,1)
        END SUBROUTINE type_in_call_test_function
    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test")
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    # pointers to non-array fields are broken
    test_fortran_frontend_ptr_assignment_removal()
    test_fortran_frontend_ptr_assignment_removal_array()
    # broken - no idea why
    test_fortran_frontend_ptr_assignment_removal_array_assumed()
    # also broken - bug in codegen
    test_fortran_frontend_ptr_assignment_removal_array_nested()
