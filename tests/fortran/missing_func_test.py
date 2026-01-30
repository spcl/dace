# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_missing_func():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  real d(5, 5), z(5)
  call init_zero_contiguous_dp(z, 5, opt_acc_async=.true., lacc=.false.)
  d(2, 1) = 5.5 + z(1)

contains

  subroutine init_contiguous_dp(var, n, v, opt_acc_async, lacc)
    integer, intent(in) :: n
    real, intent(out) :: var(n)
    real, intent(in) :: v
    logical, intent(in), optional :: opt_acc_async
    logical, intent(in), optional :: lacc
    integer :: i
    logical :: lzacc

    call set_acc_host_or_device(lzacc, lacc)
    do i = 1, n
      var(i) = v
    end do
    call acc_wait_if_requested(1, opt_acc_async)
  end subroutine init_contiguous_dp

  subroutine init_zero_contiguous_dp(var, n, opt_acc_async, lacc)
    integer, intent(in) :: n
    real, intent(out) :: var(n)
    logical, intent(IN), optional :: opt_acc_async
    logical, intent(IN), optional :: lacc

    call init_contiguous_dp(var, n, 0.0, opt_acc_async, lacc)
    var(1) = var(1) + 1.0
  end subroutine init_zero_contiguous_dp

  subroutine set_acc_host_or_device(lzacc, lacc)
    logical, intent(out) :: lzacc
    logical, intent(in), optional :: lacc

    lzacc = .false.
  end subroutine set_acc_host_or_device

  subroutine acc_wait_if_requested(acc_async_queue, opt_acc_async)
    integer, intent(IN) :: acc_async_queue
    logical, intent(IN), optional :: opt_acc_async
  end subroutine acc_wait_if_requested
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 6.5)
    assert (a[2, 0] == 42)


def test_fortran_frontend_missing_extraction():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  real d(5, 5)
  real z(5)
  integer :: jk = 5
  integer :: nrdmax_jg = 3
  do jk = max(0, nrdmax_jg - 2), 2
    d(jk, jk) = 17
  end do
  d(2, 1) = 5.5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(
        sources,
        'main',
    )
    sdfg.simplify()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 17)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    test_fortran_frontend_missing_func()
    test_fortran_frontend_missing_extraction()
