# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_optional_adv():
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(res, res2, a)
  integer, dimension(4) :: res, res2
  integer :: a
  integer, dimension(2) :: ret

  call fun(res, a)
  call fun(res2)
  call get_indices_c(1, 1, 1, ret(1), ret(2), 1, 2)

contains

  subroutine fun(res, a)
    integer, dimension(2) :: res
    integer, optional :: a
    res(1) = a
  end subroutine fun

  subroutine get_indices_c(i_blk, i_startblk, i_endblk, i_startidx, &
                           i_endidx, irl_start, opt_rl_end)
    integer, intent(IN) :: i_blk      ! Current block (variable jb in do loops)
    integer, intent(IN) :: i_startblk ! Start block of do loop
    integer, intent(IN) :: i_endblk   ! End block of do loop
    integer, intent(IN) :: irl_start  ! refin_ctrl level where do loop starts
    integer, optional, intent(IN) :: opt_rl_end ! refin_ctrl level where do loop ends
    integer, intent(OUT) :: i_startidx, i_endidx ! Start and end indices (jc loop)
    ! Local variables
    integer :: irl_end
    if (present(opt_rl_end)) then
      irl_end = opt_rl_end
    else
      irl_end = 42
    end if
    if (i_blk == i_startblk) then
      i_startidx = 1
      i_endidx = 42
      if (i_blk == i_endblk) i_endidx = irl_end
    else if (i_blk == i_endblk) then
      i_startidx = 1
      i_endidx = irl_end
    else
      i_startidx = 1
      i_endidx = 42
    end if
  end subroutine get_indices_c
end subroutine main
    """, 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5)

    assert res[0] == 5
    assert res2[0] == 0


if __name__ == "__main__":
    test_fortran_frontend_optional_adv()
