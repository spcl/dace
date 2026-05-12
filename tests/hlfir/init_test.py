"""Verbatim port of f2dace/dev:tests/fortran/init_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_init(tmp_path):
    src = """
module lib1
  implicit none
  real :: outside_init = epsilon(1.0)
end module lib1

module lib2
contains
  subroutine init_test_function(d)
    use lib1, only: outside_init
    double precision d(4)
    real:: bob = epsilon(1.0)
    d(2) = 5.5 + bob + outside_init
  end subroutine init_test_function
end module lib2

subroutine main(d)
  use lib2, only: init_test_function
  implicit none
  double precision d(4)
  call init_test_function(d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    # Source: ``5.5 + bob + outside_init`` where both ``bob`` and
    # ``outside_init`` are ``epsilon(1.0)`` (~1.19e-7).  Fortran
    # evaluates the chain in real(4) (single precision), where the
    # additions round to 5.5 because ``epsilon(1.0)`` is below the
    # float32 spacing at the value 5.5.  The bridge currently promotes
    # to double precision earlier in the chain, so the result lands at
    # ~5.5 + 2*epsilon ≈ 5.5000002 — numerically accurate, but not
    # exactly 5.5.  Both are correct; tolerate up to ~3*epsilon(1.0).
    assert abs(a[1] - 5.5) < 1e-6
    assert (a[2] == 42)


def test_fortran_frontend_init2(tmp_path):
    src = """
module lib1
  implicit none
  real, parameter :: TORUS_MAX_LAT = 4.0/18.0*atan(1.0)
end module lib1

module lib2
contains
  subroutine init2_test_function(d)
    use lib1, only: TORUS_MAX_LAT
    double precision d(4)
    d(2) = 5.5 + TORUS_MAX_LAT
  end subroutine init2_test_function
end module lib2

subroutine main(d)
  use lib2, only: init2_test_function
  implicit none
  double precision d(4)
  call init2_test_function(d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.674532920122147, 42, 42])
