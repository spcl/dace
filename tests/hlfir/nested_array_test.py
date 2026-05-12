"""Verbatim port of f2dace/dev:tests/fortran/nested_array_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_nested_array_access(tmp_path):
    src = """
subroutine main(d)
  double precision d(4)
  integer test(3, 3, 3)
  integer indices(3, 3, 3)
  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_fortran_frontend_nested_array_access_pointer_args_1(tmp_path):
    src = """
subroutine main(d, test, indices)
  implicit none
  double precision, intent(inout) :: d(4)

  ! WARNING: There are some caveats about these arguments.
  ! - SDFG will treat these pointer args as **allocatable array args**, and will generate the extra symbols for them.
  ! - But the function actually does not read from the input pointers, but immediately repoint them to internal arrays.
  integer, pointer, intent(inout) :: test(:, :, :)
  integer, pointer, intent(inout) :: indices(:, :, :)

  integer, target :: internal_test(3, 4, 5)
  integer, target :: internal_indices(3, 4, 5)
  indices => internal_test
  indices => internal_indices
  internal_indices(1, 1, 1) = 2
  internal_indices(1, 1, 2) = 3
  internal_indices(1, 1, 3) = 1
  internal_test(internal_indices(1, 1, 1), internal_indices(1, 1, 2), internal_indices(1, 1, 3)) = 2
  d(internal_test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([4], 42, order="F", dtype=np.float64)
    test = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    indices = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    sdfg(d=d, test=test, indices=indices)
    assert np.allclose(d, [42, 5.5, 42, 42])


def test_fortran_frontend_nested_array_access_pointer_args_2(tmp_path):
    src = """
subroutine main(d, test, indices)
  implicit none
  double precision, intent(inout) :: d(4)

  ! WARNING: There are some caveats about these arguments.
  ! - SDFG will treat these pointer args as **allocatable array args**, and will generate the extra symbols for them.
  ! - This time, the function will actually write and read from those arrays (although igoring their initial values).
  integer, pointer, intent(inout) :: test(:, :, :)
  integer, pointer, intent(inout) :: indices(:, :, :)

  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([4], 42, order="F", dtype=np.float64)
    test = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    indices = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    # ``test`` collides with sympy's ``LazyFunction`` attribute; the bridge
    # renames it to ``program_test`` on the SDFG side (commit 52bf266f7
    # documents the contract).  Pass through the renamed SDFG-side name
    # here; the binding wrapper (when used) restores ``test`` on the
    # Python wrapper via ``builder.dace_name_map``.
    #
    # The dim symbols ``indices_d0`` / ``indices_d1`` / ``test_d0`` /
    # ``test_d1`` (but not the d2 of each, which only appears in shape)
    # are surfaced as free SDFG kwargs because the bridge's column-major
    # stride expressions ``(1, d0, d0*d1)`` reference them outside any
    # array-shape binding context.  Pass the runtime shape values
    # explicitly so DaCe's signature is satisfied.
    sdfg(d=d, program_test=test, indices=indices, indices_d0=3, indices_d1=4, test_d0=3, test_d1=4)
    assert np.allclose(d, [42, 5.5, 42, 42])
