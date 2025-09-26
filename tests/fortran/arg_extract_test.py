# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_min_in_if():
    """
    Tests that the min intrinsic function is correctly handled in an if condition.
    """
    sources, main = (SourceCodeBuilder().add_file("""
subroutine main(d, res)
  real, dimension(2) :: d
  real, dimension(2) :: res
  if (min(d(1), 1.) .eq. 1) then
    res(1) = 3
    res(2) = 7
  else
    res(1) = 5
    res(2) = 10
  end if
end subroutine main
""").check_with_gfortran().get())
    sdfg = create_singular_sdfg_from_string(sources, "main", True)
    sdfg.simplify()

    d = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=d, res=res)
    assert np.allclose(res, [3, 7])


def test_fortran_frontend_nested_merge():
    """
    Tests that nested merge intrinsic functions are correctly handled.
    """
    sources, main = (SourceCodeBuilder().add_file("""
subroutine main(d, res)
  implicit none
  real, dimension(2) :: d
  real, dimension(2) :: res
  integer :: jg
  logical, dimension(2) :: is_cloud
  jg = 1
  is_cloud(1) = .true.
  d(1) = 10
  d(2) = 20
  res(1) = merge(merge(d(1), d(2), d(1) < d(2) .and. is_cloud(jg)), 0.0, is_cloud(jg))
  res(2) = 52
end subroutine main
""").check_with_gfortran().get())
    sdfg = create_singular_sdfg_from_string(sources, "main", True)
    sdfg.simplify()

    d = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=d, res=res)
    assert np.allclose(res, [10, 52])


def test_fortran_frontend_sequential_merge():
    """
    Tests that sequential merge intrinsic functions are correctly handled.
    """
    sources, main = (SourceCodeBuilder().add_file("""
subroutine main(d, res)
  real, dimension(2) :: d
  real, dimension(2) :: res
  real :: merge_val
  real :: merge_val2
  integer :: jg
  logical, dimension(2) :: is_cloud
  jg = 1
  is_cloud(1) = .true.
  d(1) = 10
  d(2) = 20
  merge_val = merge(d(1), d(2), d(1) < d(2) .and. is_cloud(jg))
  merge_val2 = merge(merge_val, 0.0, is_cloud(jg))
  res(1) = merge_val2
  res(2) = 52
end subroutine main
""").check_with_gfortran().get())
    sdfg = create_singular_sdfg_from_string(sources, "main", True)
    sdfg.simplify()

    d = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=d, res=res)
    assert np.allclose(res, [10, 52])


if __name__ == "__main__":
    test_fortran_frontend_min_in_if()
    test_fortran_frontend_nested_merge()
    test_fortran_frontend_sequential_merge()
