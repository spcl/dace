# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

def test_fortran_frontend_minval_double():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM minval_test
                    implicit none
                    double precision, dimension(7) :: d
                    double precision, dimension(4) :: res
                    CALL minval_test_function(d, res)
                    end

                    SUBROUTINE minval_test_function(d, res)
                    double precision, dimension(7) :: d
                    double precision, dimension(0) :: dt
                    double precision, dimension(4) :: res

                    res(1) = MINVAL(d)
                    res(2) = MINVAL(d(:))
                    res(3) = MINVAL(d(3:6))
                    res(4) = MINVAL(dt)

                    END SUBROUTINE minval_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "minval_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    # It should be the dace max for integer
    assert res[3] == np.finfo(np.float64).max

    # Minimum is in the beginning
    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    # It should be the dace max for integer
    assert res[3] == np.finfo(np.float64).max

def test_fortran_frontend_minval_int():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM minval_test
                    implicit none
                    integer, dimension(7) :: d
                    integer, dimension(4) :: res
                    CALL minval_test_function(d, res)
                    end

                    SUBROUTINE minval_test_function(d, res)
                    integer, dimension(7) :: d
                    integer, dimension(0) :: dt
                    integer, dimension(4) :: res

                    res(1) = MINVAL(d)
                    res(2) = MINVAL(d(:))
                    res(3) = MINVAL(d(3:6))
                    res(4) = MINVAL(dt)

                    END SUBROUTINE minval_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "minval_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    d = np.full([size], 0, order="F", dtype=np.int32)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    # It should be the dace max for integer
    assert res[3] == np.iinfo(np.int32).max

    # Minimum is in the beginning
    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    # It should be the dace max for integer
    assert res[3] == np.iinfo(np.int32).max

    # Minimum is in the middle
    d = np.full([size], 0, order="F", dtype=np.int32)
    d[:] = [-5, 10, -6, 4, 32, 42, -1]
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[2]
    assert res[1] == d[2]
    assert res[2] == d[2]
    # It should be the dace max for integer
    assert res[3] == np.iinfo(np.int32).max

def test_fortran_frontend_maxval_double():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM minval_test
                    implicit none
                    double precision, dimension(7) :: d
                    double precision, dimension(4) :: res
                    CALL minval_test_function(d, res)
                    end

                    SUBROUTINE minval_test_function(d, res)
                    double precision, dimension(7) :: d
                    double precision, dimension(0) :: dt
                    double precision, dimension(4) :: res

                    res(1) = MAXVAL(d)
                    res(2) = MAXVAL(d(:))
                    res(3) = MAXVAL(d(3:6))
                    res(4) = MAXVAL(dt)

                    END SUBROUTINE minval_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "minval_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    # It should be the dace max for integer
    assert res[3] == np.finfo(np.float64).min

    # Minimum is in the beginning
    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    # It should be the dace max for integer
    assert res[3] == np.finfo(np.float64).min

def test_fortran_frontend_maxval_int():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM minval_test
                    implicit none
                    integer, dimension(7) :: d
                    integer, dimension(4) :: res
                    CALL minval_test_function(d, res)
                    end

                    SUBROUTINE minval_test_function(d, res)
                    integer, dimension(7) :: d
                    integer, dimension(0) :: dt
                    integer, dimension(4) :: res

                    res(1) = MAXVAL(d)
                    res(2) = MAXVAL(d(:))
                    res(3) = MAXVAL(d(3:6))
                    res(4) = MAXVAL(dt)

                    END SUBROUTINE minval_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "minval_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    size = 7

    # Minimum is in the beginning
    d = np.full([size], 0, order="F", dtype=np.int32)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    # It should be the dace max for integer
    assert res[3] == np.iinfo(np.int32).min

    # Minimum is in the beginning
    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    # It should be the dace max for integer
    assert res[3] == np.iinfo(np.int32).min

    # Minimum is in the middle
    d = np.full([size], 0, order="F", dtype=np.int32)
    d[:] = [41, 10, 42, -5, 32, 41, 40]
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[2]
    assert res[1] == d[2]
    assert res[2] == d[2]
    # It should be the dace max for integer
    assert res[3] == np.iinfo(np.int32).min

if __name__ == "__main__":

    test_fortran_frontend_minval_double()
    test_fortran_frontend_minval_int()
    test_fortran_frontend_maxval_double()
    test_fortran_frontend_maxval_int()
