"""Verbatim port of f2dace/dev:tests/fortran/empty_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_empty(tmp_path):
    """Test that empty subroutines and functions are correctly parsed."""
    src = """
module module_mpi
  integer, parameter :: process_mpi_all_size = 0
contains
  logical function fun_with_no_arguments()
    fun_with_no_arguments = (process_mpi_all_size <= 1)
  end function fun_with_no_arguments
end module module_mpi

subroutine main(d)
  use module_mpi, only: fun_with_no_arguments
  double precision d(2, 3)
  logical :: bla = .false.

  bla = fun_with_no_arguments()
  if (bla) then
    d(1, 1) = 0
    d(1, 2) = 5
    d(2, 3) = 0
  else
    d(1, 2) = 1
  end if
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)
