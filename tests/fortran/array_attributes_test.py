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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1,5):
        # offset -1 is already added
        assert a[i-1] == i * 2

def test_fortran_frontend_array_attribute_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(50:54) :: d
                    !double precision, dimension(5) :: d
                    !double precision d(50:54)
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    !double precision d(50:54)
                    !double precision d(5)
                    double precision, dimension(50:54) :: d
                    !double precision, intent(inout) :: d(50:54)

                    do i=50,54
                       d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(50,54):
        # offset -1 is already added
        assert a[i-1] == i * 2

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
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(50,54):
        # offset -1 is already added
        assert a[i-1] == i * 2


if __name__ == "__main__":

    test_fortran_frontend_array_offset()
    test_fortran_frontend_array_attribute_no_offset()
    test_fortran_frontend_array_attribute_offset()
