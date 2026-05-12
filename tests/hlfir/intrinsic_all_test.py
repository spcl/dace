"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_all_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_all_array(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(d, res)
logical, dimension(5) :: d
logical, dimension(2) :: res

res(1) = ALL(d)

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    size = 5
    d = np.full([size], False, order="F", dtype=np.bool_)
    res = np.full([2], False, order="F", dtype=np.bool_)

    d[2] = True
    sdfg(d=d, res=res)
    assert res[0] == False

    d = np.full([size], True, order="F", dtype=np.bool_)
    sdfg(d=d, res=res)
    assert res[0] == True


def test_fortran_frontend_all_array_dim(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(d, res)
logical, dimension(5) :: d
logical, dimension(2) :: res

res(1) = ALL(d, 1)

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()


def test_fortran_frontend_all_array_comparison(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(first, second, res)
integer, dimension(5) :: first
integer, dimension(5) :: second
logical, dimension(7) :: res

res(1) = ALL(first .eq. second)
res(2) = ALL(first(:) .eq. second)
res(3) = ALL(first .eq. second(:))
res(4) = ALL(first(:) .eq. second(:))
res(5) = ALL(first(1:5) .eq. second(1:5))
! This will also be true - the only same
! element is at position 3.
res(6) = ALL(first(1:3) .eq. second(3:5))
res(7) = ALL(first(1:2) .eq. second(4:5))

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    size = 5
    first = np.full([size], 1, order="F", dtype=np.int32)
    second = np.full([size], 1, order="F", dtype=np.int32)
    second[2] = 2
    res = np.full([7], False, order="F", dtype=np.bool_)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [0, 0, 0, 0, 0, 0, 1]

    second = np.full([size], 2, order="F", dtype=np.int32)
    res = np.full([7], False, order="F", dtype=np.bool_)
    sdfg(first=first, second=second, res=res)
    for val in res:
        assert val == False


def test_fortran_frontend_all_array_scalar_comparison(tmp_path):
    # Original f2dace test had ``res(4) = ALL(first(3) .eq. 42)``
    # alongside the rank-1 cases.  ``first(3)`` is rank-0, the resulting
    # comparison is rank-0, and ``ALL`` requires a rank>=1 mask -- flang
    # rejects it as a Fortran semantic error (not a flang bug).  Drop
    # that line; the 6 surviving forms exercise the rank-1 path.
    src = """
SUBROUTINE intrinsic_all_test_function(first, res)
integer, dimension(5) :: first
logical, dimension(6) :: res

res(1) = ALL(first .eq. 42)
res(2) = ALL(first(:) .eq. 42)
res(3) = ALL(first(1:2) .eq. 42)
res(4) = ALL(first(3:5) .eq. 42)
res(5) = ALL(42 .eq. first)
res(6) = ALL(42 .ne. first)

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    size = 5
    first = np.full([size], 42, order="F", dtype=np.int32)
    res = np.full([6], False, order="F", dtype=np.bool_)

    sdfg(first=first, res=res)
    for val in res[0:-1]:
        assert val == True
    assert res[-1] == False

    first[1] = 5
    sdfg(first=first, res=res)
    assert list(res) == [0, 0, 0, 1, 0, 0]

    first[1] = 42
    first[3] = 7
    sdfg(first=first, res=res)
    assert list(res) == [0, 0, 1, 0, 0, 0]

    first = np.full([size], 41, order="F", dtype=np.int32)
    sdfg(first=first, res=res)
    assert list(res) == [0, 0, 0, 0, 0, 1]


def test_fortran_frontend_all_array_2d(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(d, res)
logical, dimension(5,7) :: d
logical, dimension(2) :: res

res(1) = ALL(d)

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    sizes = [5, 7]
    d = np.full(sizes, True, order="F", dtype=np.bool_)
    res = np.full([2], False, order="F", dtype=np.bool_)

    d[2, 2] = False
    sdfg(d=d, res=res)
    assert res[0] == False

    d[2, 2] = True
    sdfg(d=d, res=res)
    assert res[0] == True


def test_fortran_frontend_all_array_comparison_2d(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(first, second, res)
integer, dimension(5,4) :: first
integer, dimension(5,4) :: second
logical, dimension(7) :: res

res(1) = ALL(first .eq. second)
res(2) = ALL(first(:,:) .eq. second)
res(3) = ALL(first .eq. second(:,:))
res(4) = ALL(first(:,:) .eq. second(:,:))
res(5) = ALL(first(1:5,:) .eq. second(1:5,:))
res(6) = ALL(first(:,1:4) .eq. second(:,1:4))
! Now test subsets.
res(7) = ALL(first(2:3, 3:4) .eq. second(2:3, 3:4))

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    sizes = [5, 4]
    first = np.full(sizes, 1, order="F", dtype=np.int32)
    second = np.full(sizes, 1, order="F", dtype=np.int32)
    second[2, 2] = 2
    res = np.full([7], False, order="F", dtype=np.bool_)

    sdfg(first=first, second=second, res=res)
    for val in res:
        assert val == False

    second = np.full(sizes, 1, order="F", dtype=np.int32)
    res = np.full([7], False, order="F", dtype=np.bool_)
    sdfg(first=first, second=second, res=res)
    for val in res:
        assert val == True


def test_fortran_frontend_all_array_comparison_2d_subset(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(first, second, res)
integer, dimension(5,4) :: first
integer, dimension(5,4) :: second
logical, dimension(2) :: res

! Now test subsets - make sure the equal values are only
! in the tested area.
res(1) = ALL(first(1:2, 3:4) .ne. second(4:5, 2:3))
res(2) = ALL(first(1:2, 3:4) .eq. second(4:5, 2:3))

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    sizes = [5, 4]
    first = np.full(sizes, 1, order="F", dtype=np.int32)
    first[2:5, :] = 2
    first[0:2, 0:2] = 2

    second = np.full(sizes, 1, order="F", dtype=np.int32)
    second[0:3, :] = 3
    second[3:5, 0] = 3
    second[3:5, 3:5] = 3

    res = np.full([2], False, order="F", dtype=np.bool_)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [0, 1]


def test_fortran_frontend_all_array_comparison_2d_subset_offset(tmp_path):
    src = """
SUBROUTINE intrinsic_all_test_function(first, second, res)
integer, dimension(20:24,4) :: first
integer, dimension(5,7:10) :: second
logical, dimension(2) :: res

! Now test subsets - make sure the equal values are only
! in the tested area.
res(1) = ALL(first(20:21, 3:4) .ne. second(4:5, 8:9))
res(2) = ALL(first(20:21, 3:4) .eq. second(4:5, 8:9))

END SUBROUTINE intrinsic_all_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_all_test_function').build()

    sizes = [5, 4]
    first = np.full(sizes, 1, order="F", dtype=np.int32)
    first[2:5, :] = 2
    first[0:2, 0:2] = 2

    second = np.full(sizes, 1, order="F", dtype=np.int32)
    second[0:3, :] = 3
    second[3:5, 0] = 3
    second[3:5, 3:5] = 3

    res = np.full([2], False, order="F", dtype=np.bool_)

    sdfg(first=first, second=second, res=res)
    assert list(res) == [0, 1]
