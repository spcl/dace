# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import ast_transforms, fortran_parser


def test_fortran_frontend_product_array():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(7) :: d
                    double precision, dimension(3) :: res
                    CALL index_test_function(d, res)
                    end

                    SUBROUTINE index_test_function(d, res)
                    double precision, dimension(7) :: d
                    double precision, dimension(3) :: res

                    res(1) = PRODUCT(d)
                    res(2) = PRODUCT(d(:))
                    res(3) = PRODUCT(d(2:5))

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 7
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == np.prod(d)
    assert res[1] == np.prod(d)
    assert res[2] == np.prod(d[1:5])


def test_fortran_frontend_product_array_dim():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res
                    CALL intrinsic_count_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(d, res)
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res

                    res(1) = PRODUCT(d, 1)

                    END SUBROUTINE intrinsic_count_test_function
                    """

    with pytest.raises(NotImplementedError):
        fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test", False)


def test_fortran_frontend_product_2d():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,3) :: d
                    double precision, dimension(4) :: res
                    CALL index_test_function(d,res)
                    end

                    SUBROUTINE index_test_function(d, res)
                    double precision, dimension(5,3) :: d
                    double precision, dimension(4) :: res

                    res(1) = PRODUCT(d)
                    res(2) = PRODUCT(d(:,:))
                    res(3) = PRODUCT(d(2:4, 2))
                    res(4) = PRODUCT(d(2:4, 2:3))

                    END SUBROUTINE index_test_function
                    """

    # Now test to verify it executes correctly with no offset normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    sizes = [5, 3]
    d = np.full(sizes, 42, order="F", dtype=np.float64)
    cnt = 1
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            d[i, j] = cnt
            cnt += 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == np.prod(d)
    assert res[1] == np.prod(d)
    assert res[2] == np.prod(d[1:4, 1])
    assert res[3] == np.prod(d[1:4, 1:3])


if __name__ == "__main__":

    test_fortran_frontend_product_array()
    test_fortran_frontend_product_array_dim()
    test_fortran_frontend_product_2d()
