"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_basic_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from ported._helpers import xfail

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@xfail('BIT_SIZE on logical/real — flang-new-21 emit-hlfir failure')
def test_fortran_frontend_bit_size(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_bit_size_function(res)
integer, dimension(4) :: res
logical :: a = .TRUE.
integer :: b = 1
real :: c = 1
double precision :: d = 1

res(1) = BIT_SIZE(a)
res(2) = BIT_SIZE(b)
res(3) = BIT_SIZE(c)
res(4) = BIT_SIZE(d)

END SUBROUTINE intrinsic_math_test_bit_size_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_bit_size_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [32, 32, 32, 64])


def test_fortran_frontend_bit_size_symbolic(tmp_path):
    src = """
subroutine main(arrsize, arrsize2, arrsize3, res, res2, res3)
  implicit none
  integer :: arrsize
  integer :: arrsize2
  integer :: arrsize3
  integer :: res(arrsize)
  integer :: res2(arrsize, arrsize2, arrsize3)
  integer :: res3(arrsize + arrsize2, arrsize2*5, arrsize3 + arrsize2*arrsize)

  res(1) = size(res)
  res(2) = size(res2)
  res(3) = size(res3)
  res(4) = size(res)*2
  res(5) = size(res)*size(res2)*size(res3)
  res(6) = size(res2, 1) + size(res2, 2) + size(res2, 3)
  res(7) = size(res3, 1) + size(res3, 2) + size(res3, 3)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 24
    size2 = 5
    size3 = 7
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size, size2, size3], 42, order="F", dtype=np.int32)
    res3 = np.full([size + size2, size2 * 5, size3 + size * size2], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, res3=res3, arrsize=size, arrsize2=size2, arrsize3=size3)

    assert res[0] == size
    assert res[1] == size * size2 * size3
    assert res[2] == (size + size2) * (size2 * 5) * (size3 + size2 * size)
    assert res[3] == size * 2
    assert res[4] == res[0] * res[1] * res[2]
    assert res[5] == size + size2 + size3
    assert res[6] == size + size2 + size2 * 5 + size3 + size * size2


def test_fortran_frontend_size_arbitrary(tmp_path):
    src = """
subroutine main(res)
  implicit none
  integer :: res(:, :)
  res(1, 1) = size(res)
  res(2, 1) = size(res, 1)
  res(3, 1) = size(res, 2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 7
    size2 = 5
    res = np.full([size, size2], 42, order="F", dtype=np.int32)
    sdfg(res=res, res_d0=size, res_d1=size2)

    assert res[0, 0] == size * size2
    assert res[1, 0] == size
    assert res[2, 0] == size2


def test_fortran_frontend_present(tmp_path):
    """``present(a)`` on an OPTIONAL dummy of an internal subprogram.
    After ``hlfir-inline-all`` flattens ``tf2``'s body into ``main``,
    each call site leaves a ``fir.is_present`` whose operand traces
    through the inlined alias to either ``main``'s mandatory dummy
    ``a`` (host bound storage → constant ``1``) or to ``fir.absent``
    (caller passed nothing → constant ``0``).  The bridge folds these
    statically at AST-extract time."""
    src = """
subroutine main(res, res2, a)
  integer, dimension(4) :: res
  integer, dimension(4) :: res2
  integer :: a
  call tf2(res, a=a)
  call tf2(res2)

contains

  subroutine tf2(res, a)
    integer, dimension(4) :: res
    integer, optional :: a
    res(1) = present(a)
  end subroutine tf2
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    # ``a`` is intent(in) on the SDFG signature (the OPTIONAL dummy in
    # the inlined ``tf2`` body); ``is_present`` is folded statically so
    # the value never reaches a tasklet, but the SDFG signature still
    # demands a scalar binding.
    sdfg(res=res, res2=res2, a=5)

    assert res[0] == 1
    assert res2[0] == 0


def test_fortran_frontend_bitwise_ops(tmp_path):
    src = """
    SUBROUTINE bitwise_ops(inp, res)

    integer, dimension(11) :: inp
    integer, dimension(11) :: res

    res(1) = IBSET(inp(1), 0)
    res(2) = IBSET(inp(2), 30)

    res(3) = IBCLR(inp(3), 0)
    res(4) = IBCLR(inp(4), 30)

    res(5) = IEOR(inp(5), 63)
    res(6) = IEOR(inp(6), 480)

    res(7) = ISHFT(inp(7), 5)
    res(8) = ISHFT(inp(8), 30)

    res(9) = ISHFT(inp(9), -5)
    res(10) = ISHFT(inp(10), -30)

    res(11) = ISHFT(inp(11), 0)

    END SUBROUTINE bitwise_ops
"""
    sdfg = build_sdfg(src, tmp_path, name='bitwise_ops').build()

    size = 11
    inp = np.full([size], 42, order="F", dtype=np.int32)
    inp[:] = [32, 32, 33, 1073741825, 53, 530, 12, 1, 128, 1073741824, 12]

    res = np.full([size], 42, order="F", dtype=np.int32)

    sdfg(inp=inp, res=res)

    assert np.allclose(res, [33, 1073741856, 32, 1, 10, 1010, 384, 1073741824, 4, 1, 12])


def test_fortran_frontend_bitwise_ops2(tmp_path):
    src = """
    SUBROUTINE bitwise_ops(inp, res)

    integer, dimension(6) :: inp
    integer, dimension(6) :: res

    res(1) = IAND(inp(1), 0)
    res(2) = IAND(inp(2), 31)

    res(3) = BTEST(inp(3), 0)
    res(4) = BTEST(inp(4), 5)

    res(5) = IBITS(inp(5), 0, 5)
    res(6) = IBITS(inp(6), 3, 10)

    END SUBROUTINE bitwise_ops
"""
    sdfg = build_sdfg(src, tmp_path, name='bitwise_ops').build()

    size = 6
    inp = np.full([size], 42, order="F", dtype=np.int32)
    inp[:] = [2147483647, 16, 3, 31, 30, 630]

    res = np.full([size], 42, order="F", dtype=np.int32)

    sdfg(inp=inp, res=res)

    assert np.allclose(res, [0, 16, 1, 0, 30, 78])


def test_fortran_frontend_allocated(tmp_path):
    src = """
    SUBROUTINE allocated_test(res)

    integer, allocatable, dimension(:) :: data
    integer, dimension(3) :: res

    res(1) = ALLOCATED(data)

    ALLOCATE(data(6))

    res(2) = ALLOCATED(data)

    DEALLOCATE(data)

    res(3) = ALLOCATED(data)

    END SUBROUTINE allocated_test
"""
    sdfg = build_sdfg(src, tmp_path, name='allocated_test').build()

    size = 3
    res = np.full([size], 42, order="F", dtype=np.int32)

    sdfg(res=res)

    assert np.allclose(res, [0, 1, 0])


def test_fortran_frontend_allocated_nested(tmp_path):
    src = """
    MODULE allocated_test_interface
        INTERFACE
            SUBROUTINE allocated_test_nested(data, res)
                integer, allocatable, dimension(:) :: data
                integer, dimension(3) :: res
            END SUBROUTINE allocated_test_nested
        END INTERFACE
    END MODULE

    SUBROUTINE allocated_test(res)
    USE allocated_test_interface
    implicit none
    integer, allocatable, dimension(:) :: data
    integer, dimension(3) :: res

    res(1) = ALLOCATED(data)

    ALLOCATE(data(6))

    CALL allocated_test_nested(data, res)

    END SUBROUTINE allocated_test

    SUBROUTINE allocated_test_nested(data, res)

    integer, allocatable, dimension(:) :: data
    integer, dimension(3) :: res

    res(2) = ALLOCATED(data)

    DEALLOCATE(data)

    res(3) = ALLOCATED(data)

    END SUBROUTINE allocated_test_nested
"""
    sdfg = build_sdfg(src, tmp_path, name='allocated_test', entry='_QPallocated_test').build()

    size = 3
    res = np.full([size], 42, order="F", dtype=np.int32)

    sdfg(res=res)

    assert np.allclose(res, [0, 1, 0])
