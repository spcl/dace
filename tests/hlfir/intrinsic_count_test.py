"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_count_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_count_array(tmp_path):
    src = """
SUBROUTINE intrinsic_count_test_function(d, res)
logical, dimension(5) :: d
integer, dimension(2) :: res

res(1) = COUNT(d)

END SUBROUTINE intrinsic_count_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

    size = 5
    d = np.full([size], False, order="F", dtype=np.bool_)
    res = np.full([2], 42, order="F", dtype=np.int32)

    d[2] = True
    sdfg(d=d, res=res)
    assert res[0] == 1

    d[2] = False
    sdfg(d=d, res=res)
    assert res[0] == 0


def test_fortran_frontend_count_array_dim(tmp_path):
    src = """
SUBROUTINE intrinsic_count_test_function(d, res)
logical, dimension(5) :: d
integer, dimension(2) :: res

res(1) = COUNT(d, 1)

END SUBROUTINE intrinsic_count_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()


def test_fortran_frontend_count_array_comparison(tmp_path):
    # ``res`` is declared ``integer, dimension(7)`` so the COUNT result
    # (an INTEGER) round-trips cleanly.  The original f2dace port used
    # ``integer, dimension(7) :: res`` for the same shape, but that
    # made the assertion ambiguous (assigning an INTEGER count to a
    # LOGICAL scalar via Fortran's truthy-cast).  Switching to an
    # integer ``res`` matches the rest of the COUNT tests in this file.
    src = """
SUBROUTINE intrinsic_count_test_function(first, second, res)
integer, dimension(5) :: first
integer, dimension(5) :: second
integer, dimension(7) :: res

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
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

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


def test_fortran_frontend_count_array_scalar_comparison(tmp_path):
    # Original f2dace test had ``COUNT(first(3) .eq. 42)`` -- rank-0
    # mask is invalid Fortran (COUNT needs rank>=1).  Dropped.
    src = """
SUBROUTINE intrinsic_count_test_function(first, res)
integer, dimension(5) :: first
integer, dimension(8) :: res

res(1) = COUNT(first .eq. 42)
res(2) = COUNT(first(:) .eq. 42)
res(3) = COUNT(first(1:2) .eq. 42)
res(4) = COUNT(first(3:5) .eq. 42)
res(5) = COUNT(42 .eq. first)
res(6) = COUNT(42 .ne. first)
res(7) = COUNT(6 .lt. first)
res(8) = COUNT(6 .gt. first)

END SUBROUTINE intrinsic_count_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

    size = 5
    first = np.full([size], 1, order="F", dtype=np.int32)
    res = np.full([8], 0, order="F", dtype=np.int32)

    sdfg(first=first, res=res)
    assert list(res) == [0, 0, 0, 0, 0, 5, 0, size]

    first[1] = 42
    sdfg(first=first, res=res)
    assert list(res) == [1, 1, 1, 0, 1, 4, 1, size - 1]

    first[1] = 5
    first[2] = 42
    sdfg(first=first, res=res)
    assert list(res) == [1, 1, 0, 1, 1, 4, 1, size - 1]

    first[2] = 7
    first[3] = 42
    sdfg(first=first, res=res)
    assert list(res) == [1, 1, 0, 1, 1, 4, 2, size - 2]


def test_fortran_frontend_count_array_2d(tmp_path):
    src = """
SUBROUTINE intrinsic_count_test_function(d, res)
logical, dimension(5,7) :: d
integer, dimension(2) :: res

res(1) = COUNT(d)

END SUBROUTINE intrinsic_count_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

    sizes = [5, 7]
    d = np.full(sizes, True, order="F", dtype=np.bool_)
    res = np.full([2], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)
    assert res[0] == 35

    d[2, 2] = False
    sdfg(d=d, res=res)
    assert res[0] == 34

    d = np.full(sizes, False, order="F", dtype=np.bool_)
    sdfg(d=d, res=res)
    assert res[0] == 0

    d[2, 2] = True
    sdfg(d=d, res=res)
    assert res[0] == 1


def test_fortran_frontend_count_array_comparison_2d(tmp_path):
    src = """
SUBROUTINE intrinsic_count_test_function(first, second, res)
integer, dimension(5,4) :: first
integer, dimension(5,4) :: second
integer, dimension(7) :: res

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
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

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


def test_fortran_frontend_count_array_comparison_2d_subset(tmp_path):
    src = """
SUBROUTINE intrinsic_count_test_function(first, second, res)
integer, dimension(5,4) :: first
integer, dimension(5,4) :: second
integer, dimension(2) :: res

! Now test subsets - make sure the equal values are only
! in the tested area.
res(1) = COUNT(first(1:2, 3:4) .ne. second(4:5, 2:3))
res(2) = COUNT(first(1:2, 3:4) .eq. second(4:5, 2:3))

END SUBROUTINE intrinsic_count_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

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


def test_fortran_frontend_count_array_comparison_2d_subset_offset(tmp_path):
    src = """
SUBROUTINE intrinsic_count_test_function(first, second, res)
integer, dimension(20:24,4) :: first
integer, dimension(5,7:10) :: second
integer, dimension(2) :: res

! Now test subsets - make sure the equal values are only
! in the tested area.
res(1) = COUNT(first(20:21, 3:4) .ne. second(4:5, 8:9))
res(2) = COUNT(first(20:21, 3:4) .eq. second(4:5, 8:9))

END SUBROUTINE intrinsic_count_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_count_test_function').build()

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
