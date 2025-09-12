# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_count_array():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    logical, dimension(5) :: d
                    integer, dimension(2) :: res
                    CALL intrinsic_count_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(d, res)
                    logical, dimension(5) :: d
                    integer, dimension(2) :: res

                    res(1) = COUNT(d)

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    d = np.full([size], False, order="F", dtype=np.int32)
    res = np.full([2], 42, order="F", dtype=np.int32)

    d[2] = True
    sdfg(d=d, res=res)
    assert res[0] == 1

    d[2] = False
    sdfg(d=d, res=res)
    assert res[0] == 0


def test_fortran_frontend_count_array_dim():
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

                    res(1) = COUNT(d, 1)

                    END SUBROUTINE intrinsic_count_test_function
                    """

    with pytest.raises(NotImplementedError):
        fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")


def test_fortran_frontend_count_array_comparison():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    integer, dimension(5) :: first
                    integer, dimension(5) :: second
                    logical, dimension(7) :: res
                    CALL intrinsic_count_test_function(first, second, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(first, second, res)
                    integer, dimension(5) :: first
                    integer, dimension(5) :: second
                    logical, dimension(7) :: res

                    res(1) = COUNT(first .eq. second)
                    res(2) = COUNT(first(:) .eq. second)
                    res(3) = COUNT(first .eq. second(:))
                    res(4) = COUNT(first(:) .eq. second(:))
                    res(5) = COUNT(first(1:5) .eq. second(1:5))
                    ! This will also be true - the only same
                    ! element is at position 3.
                    res(6) = COUNT(first(1:3) .eq. second(3:5))
                    res(7) = COUNT(first(1:2) .eq. second(4:5))

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    first = np.full([size], 1, order="F", dtype=np.int32)
    second = np.full([size], 1, order="F", dtype=np.int32)
    second[2] = 2
    res = np.full([7], 0, order="F", dtype=np.int32)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [4, 4, 4, 4, 4, 2, 2]

    second = np.full([size], 2, order="F", dtype=np.int32)
    res = np.full([7], 0, order="F", dtype=np.int32)
    sdfg(first=first, second=second, res=res)
    for val in res:
        assert val == 0

    second = np.full([size], 1, order="F", dtype=np.int32)
    res = np.full([7], 0, order="F", dtype=np.int32)
    sdfg(first=first, second=second, res=res)
    assert list(res) == [5, 5, 5, 5, 5, 3, 2]


def test_fortran_frontend_count_array_scalar_comparison():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    integer, dimension(5) :: first
                    logical, dimension(9) :: res
                    CALL intrinsic_count_test_function(first, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(first, res)
                    integer, dimension(5) :: first
                    logical, dimension(9) :: res

                    res(1) = COUNT(first .eq. 42)
                    res(2) = COUNT(first(:) .eq. 42)
                    res(3) = COUNT(first(1:2) .eq. 42)
                    res(4) = COUNT(first(3) .eq. 42)
                    res(5) = COUNT(first(3:5) .eq. 42)
                    res(6) = COUNT(42 .eq. first)
                    res(7) = COUNT(42 .ne. first)
                    res(8) = COUNT(6 .lt. first)
                    res(9) = COUNT(6 .gt. first)

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    first = np.full([size], 1, order="F", dtype=np.int32)
    res = np.full([9], 0, order="F", dtype=np.int32)

    sdfg(first=first, res=res)
    assert list(res) == [0, 0, 0, 0, 0, 0, 5, 0, size]

    first[1] = 42
    sdfg(first=first, res=res)
    assert list(res) == [1, 1, 1, 0, 0, 1, 4, 1, size - 1]

    first[1] = 5
    first[2] = 42
    sdfg(first=first, res=res)
    assert list(res) == [1, 1, 0, 1, 1, 1, 4, 1, size - 1]

    first[2] = 7
    first[3] = 42
    sdfg(first=first, res=res)
    assert list(res) == [1, 1, 0, 0, 1, 1, 4, 2, size - 2]


@pytest.mark.skip("Changing the order of AST transformations prevents the intrinsics from analyzing it")
def test_fortran_frontend_count_array_comparison_wrong_subset():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    logical, dimension(5) :: first
                    logical, dimension(5) :: second
                    logical, dimension(2) :: res
                    CALL intrinsic_count_test_function(first, second, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(first, second, res)
                    logical, dimension(5) :: first
                    logical, dimension(5) :: second
                    logical, dimension(2) :: res

                    res(1) = COUNT(first(1:2) .eq. second(2:5))

                    END SUBROUTINE intrinsic_count_test_function
                    """

    with pytest.raises(TypeError):
        fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")


def test_fortran_frontend_count_array_2d():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    logical, dimension(5,7) :: d
                    logical, dimension(2) :: res
                    CALL intrinsic_count_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(d, res)
                    logical, dimension(5,7) :: d
                    logical, dimension(2) :: res

                    res(1) = COUNT(d)

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    sizes = [5, 7]
    d = np.full(sizes, True, order="F", dtype=np.int32)
    res = np.full([2], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)
    assert res[0] == 35

    d[2, 2] = False
    sdfg(d=d, res=res)
    assert res[0] == 34

    d = np.full(sizes, False, order="F", dtype=np.int32)
    sdfg(d=d, res=res)
    assert res[0] == 0

    d[2, 2] = True
    sdfg(d=d, res=res)
    assert res[0] == 1


def test_fortran_frontend_count_array_comparison_2d():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    integer, dimension(5,4) :: first
                    integer, dimension(5,4) :: second
                    logical, dimension(7) :: res
                    CALL intrinsic_count_test_function(first, second, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(first, second, res)
                    integer, dimension(5,4) :: first
                    integer, dimension(5,4) :: second
                    logical, dimension(7) :: res

                    res(1) = COUNT(first .eq. second)
                    res(2) = COUNT(first(:,:) .eq. second)
                    res(3) = COUNT(first .eq. second(:,:))
                    res(4) = COUNT(first(:,:) .eq. second(:,:))
                    res(5) = COUNT(first(1:5,:) .eq. second(1:5,:))
                    res(6) = COUNT(first(:,1:4) .eq. second(:,1:4))
                    ! Now test subsets.
                    res(7) = COUNT(first(2:3, 3:4) .eq. second(2:3, 3:4))

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    sizes = [5, 4]
    first = np.full(sizes, 1, order="F", dtype=np.int32)
    second = np.full(sizes, 2, order="F", dtype=np.int32)
    second[1, 1] = 1
    res = np.full([7], 0, order="F", dtype=np.int32)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [1, 1, 1, 1, 1, 1, 0]

    second = np.full(sizes, 1, order="F", dtype=np.int32)
    res = np.full([7], 0, order="F", dtype=np.int32)
    sdfg(first=first, second=second, res=res)
    assert list(res) == [20, 20, 20, 20, 20, 20, 4]


def test_fortran_frontend_count_array_comparison_2d_subset():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    integer, dimension(5,4) :: first
                    integer, dimension(5,4) :: second
                    logical, dimension(2) :: res
                    CALL intrinsic_count_test_function(first, second, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(first, second, res)
                    integer, dimension(5,4) :: first
                    integer, dimension(5,4) :: second
                    logical, dimension(2) :: res

                    ! Now test subsets - make sure the equal values are only
                    ! in the tested area.
                    res(1) = COUNT(first(1:2, 3:4) .ne. second(4:5, 2:3))
                    res(2) = COUNT(first(1:2, 3:4) .eq. second(4:5, 2:3))

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test")
    sdfg.simplify(verbose=True)
    sdfg.compile()

    sizes = [5, 4]
    first = np.full(sizes, 1, order="F", dtype=np.int32)
    first[2:5, :] = 2
    first[0:2, 0:2] = 2

    second = np.full(sizes, 1, order="F", dtype=np.int32)
    second[0:3, :] = 2
    second[3:5, 0] = 2
    second[3:5, 3:5] = 2

    res = np.full([2], 0, order="F", dtype=np.int32)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [0, 4]


def test_fortran_frontend_count_array_comparison_2d_subset_offset():
    test_string = """
                    PROGRAM intrinsic_count_test
                    implicit none
                    integer, dimension(20:24,4) :: first
                    integer, dimension(5,7:10) :: second
                    logical, dimension(2) :: res
                    CALL intrinsic_count_test_function(first, second, res)
                    end

                    SUBROUTINE intrinsic_count_test_function(first, second, res)
                    integer, dimension(20:24,4) :: first
                    integer, dimension(5,7:10) :: second
                    logical, dimension(2) :: res

                    ! Now test subsets - make sure the equal values are only
                    ! in the tested area.
                    res(1) = COUNT(first(20:21, 3:4) .ne. second(4:5, 8:9))
                    res(2) = COUNT(first(20:21, 3:4) .eq. second(4:5, 8:9))

                    END SUBROUTINE intrinsic_count_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_count_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    sizes = [5, 4]
    first = np.full(sizes, 1, order="F", dtype=np.int32)
    first[2:5, :] = 2
    first[0:2, 0:2] = 2

    second = np.full(sizes, 1, order="F", dtype=np.int32)
    second[0:3, :] = 2
    second[3:5, 0] = 2
    second[3:5, 3:5] = 2

    res = np.full([2], 0, order="F", dtype=np.int32)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [0, 4]


if __name__ == "__main__":

    test_fortran_frontend_count_array()
    test_fortran_frontend_count_array_dim()
    test_fortran_frontend_count_array_comparison()
    test_fortran_frontend_count_array_scalar_comparison()
    test_fortran_frontend_count_array_comparison_wrong_subset()
    test_fortran_frontend_count_array_2d()
    test_fortran_frontend_count_array_comparison_2d()
    test_fortran_frontend_count_array_comparison_2d_subset()
    test_fortran_frontend_count_array_comparison_2d_subset_offset()
