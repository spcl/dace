# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

from dace import nodes
from dace.frontend.fortran import fortran_parser
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string

from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_class():
    """
    Tests that whether clasess are translated correctly
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none

  type, abstract :: t_comm_pattern
  end type t_comm_pattern

  type, extends(t_comm_pattern) :: t_comm_pattern_orig
    integer :: n_pnts
    integer, allocatable :: recv_limits(:)
  contains
    procedure :: setup => setup_comm_pattern
    procedure :: exchange_data_r3d => exchange_data_r3d
  end type t_comm_pattern_orig

contains

  subroutine setup_comm_pattern(p_pat, dst_n_points)
    implicit none
    class(t_comm_pattern_orig), target, intent(OUT) :: p_pat
    integer, intent(IN) :: dst_n_points
    p_pat%n_pnts = dst_n_points
  end subroutine setup_comm_pattern

  subroutine exchange_data_r3d(p_pat, recv)
    implicit none
    class(t_comm_pattern_orig), target, intent(INOUT) :: p_pat
    real, intent(INOUT), target :: recv(:, :, :)
    recv(1, 1, 1) = recv(1, 1, 1) + p_pat%n_pnts
  end subroutine exchange_data_r3d
end module lib

subroutine main(d)
  use lib
  implicit none
  integer d(2)
  real recv(2, 2, 2)
  class(t_comm_pattern_orig), allocatable :: p_pat
  call setup_comm_pattern(p_pat, 400)
  call exchange_data_r3d(p_pat, recv)
  d(1) = p_pat%n_pnts
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    sdfg.compile()
    d = np.full([2], 42, order="F", dtype=np.int64)
    sdfg(d=d)
    assert np.all(d == [400, 42])


if __name__ == "__main__":
    test_fortran_frontend_class()
