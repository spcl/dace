# Copyright 2025 ETH Zurich and the DaCe quthors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder
import pytest


def test_fortran_frontend_case_const():
    """Tests that the cases statement can use parameters."""
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: a = 1

contains
  subroutine foo(v)
    integer, intent(inout) :: v

    ! if (v == a) then
    !   v = 5
    ! end if

    select case(v)
    case(a)
      v = 5
    end select
  end subroutine foo
end module lib

subroutine main(d)
  use lib
  implicit none
  integer :: d(5, 5)
  call foo(d(1, 2))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.array([[i + j for i in range(5)] for j in range(5)], order="F", dtype=np.int32)
    sdfg(d=a)
    assert (a[0, 1] == 5)


@pytest.mark.skip("Fails because range statements in case selectors are not supported.")
def test_fortran_frontend_case_const_range():
    """Tests that the cases statement can use parameters."""
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: a = 1
  integer, parameter :: b = 2
  integer, parameter :: c = 3

contains
  subroutine foo(v)
    integer, intent(inout) :: v

    ! if (v == a) then
    !   v = 5
    ! end if

    select case(v)
    case(b:c)
      v = 6
    end select
  end subroutine foo
end module lib

subroutine main(d)
  use lib
  implicit none
  integer :: d(5, 5)
  call foo(d(1, 3))
  call foo(d(1, 5))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.array([[i + j for i in range(5)] for j in range(5)], order="F", dtype=np.int32)
    sdfg(d=a)
    assert (a[0, 2] == 6)
    assert (a[0, 4] == 4)


if __name__ == "__main__":
    test_fortran_frontend_case_const()
    test_fortran_frontend_case_const_range()
