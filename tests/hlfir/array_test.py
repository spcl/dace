"""Verbatim port of f2dace/dev:tests/fortran/array_test.py.

Each test keeps the original Fortran source unchanged.  The harness is
swapped from f2dace's ``SourceCodeBuilder().check_with_gfortran() +
create_singular_sdfg_from_string`` to FaCe's ``build_sdfg`` + an f2py
reference where applicable.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_array_access(tmp_path):
    """Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct."""
    src = """
subroutine main(d)
  double precision d(4)
  d(2) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_fortran_frontend_array_ranges(tmp_path):
    """Tests that the Fortran frontend can parse multidimenstional arrays with vectorized ranges."""
    src = """
subroutine main(d)
  double precision d(3, 4, 5), e(3, 4, 5), f(3, 4, 5)
  e(:, :, :) = 1.0
  f(:, :, :) = 2.0
  f(:, 2:4, :) = 3.0
  f(1, 1, :) = 4.0
  d(:, :, :) = e(:, :, :) + f(:, :, :)
  d(1, 2:4, 1) = e(1, 2:4, 1)*10.0
  d(1, 1, 1) = sum(e(:, 1, :))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 15)
    assert (d[0, 1, 0] == 10)
    assert (d[1, 0, 0] == 3)
    assert (d[2, 3, 3] == 4)
    assert (d[0, 0, 2] == 5)


def test_fortran_frontend_array_multiple_ranges_with_symbols(tmp_path):
    """Tests that the Fortran frontend can parse multidimenstional arrays with vectorized ranges over symbols."""
    src = """
subroutine main(a, lu, iend, m)
  integer, intent(in) :: iend, m
  double precision, intent(inout) :: a(iend, m, m), lu(iend, m, m)
  lu(1:iend,1:m,1:m) = a(1:iend,1:m,1:m)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    iend, m = 3, 4
    lu = np.full([iend, m, m], 0, order="F", dtype=np.float64)
    a = np.full([iend, m, m], 42, order="F", dtype=np.float64)
    sdfg(a=a, lu=lu, iend=iend, m=m)
    assert np.allclose(lu, 42)


def test_fortran_frontend_array_3dmap(tmp_path):
    """Tests that the normalization of multidimensional array indices works correctly."""
    src = """
subroutine main(d)
  double precision d(4, 4, 4)
  d(:, :, :) = 7
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4, 4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0, 0] == 7)
    assert (a[3, 3, 3] == 7)


def test_fortran_frontend_twoconnector(tmp_path):
    """Tests that the multiple connectors to one array are handled correctly."""
    src = """
subroutine main(d)
  double precision d(4)
  d(2) = d(1) + d(3)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 84)
    assert (a[2] == 42)


def test_fortran_frontend_input_output_connector(tmp_path):
    """Tests that the presence of input and output connectors for the same array is handled correctly."""
    src = """
subroutine main(d)
  double precision d(2, 3)
  integer a, b
  a = 1
  b = 2
  d(:, :) = 0.0
  d(a, b) = d(1, 1) + 5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    # `a`, `b` are local scalars promoted to symbols (they index `d`).  Pass
    # placeholder values; the SDFG initialises them via interstate edges.
    sdfg(d=a, a=0, b=0)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


def test_fortran_frontend_memlet_in_map_test(tmp_path):
    """Tests that no assumption is made where the iteration variable is inside a memlet subset."""
    src = """
subroutine main(INP, OUT)
  real INP(100, 10)
  real OUT(100, 10)
  integer I
  do I = 1, 100
    call inner_loops(INP(I, :), OUT(I, :))
  end do
end subroutine main

subroutine inner_loops(INP, OUT)
  real INP(10)
  real OUT(10)
  integer J


  do J = 1, 10
    OUT(J) = INP(J) + 1
  end do
end subroutine inner_loops
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    sdfg.validate()
    rng = np.random.default_rng(0)
    inp = np.asfortranarray(rng.random((100, 10), dtype=np.float64).astype(np.float32))
    out = np.zeros((100, 10), order="F", dtype=np.float32)
    sdfg(inp=inp, out=out, i=0, j=0)
    np.testing.assert_allclose(out, inp + 1.0, rtol=1e-6, atol=1e-6)


def test_pass_an_arrayslice_that_looks_like_a_scalar_from_outside_with_literal_size(tmp_path):
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d)
  use lib
  integer :: i
  integer :: sz
  real, intent(inout) :: d(50)
  do i=1, 50
    d(i) = i * 1.0
  end do
  sz = 0
  do i=2,3
    sz = sz + i
  end do
  d(1) = f(d(11), 5)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.full([50], 42, order="F", dtype=np.float32)
    sdfg(d=d)
    assert d[0] == 65


def test_pass_an_arrayslice_that_looks_like_a_scalar_from_outside_with_symbolic_size(tmp_path):
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d)
  use lib
  integer :: i
  integer :: sz
  real, intent(inout) :: d(50)
  do i=1, 50
    d(i) = i * 1.0
  end do
  sz = 0
  do i=2,3
    sz = sz + i
  end do
  d(1) = f(d(11), sz)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.full([50], 42, order="F", dtype=np.float32)
    sdfg(d=d)
    assert d[0] == 65
