# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_optional_adv():
    test_string = """
                    PROGRAM adv_intrinsic_optional_test_function
                    implicit none
                    integer, dimension(4) :: res
                    integer, dimension(4) :: res2
                    integer :: a
                    CALL intrinsic_optional_test_function(res, res2, a)
                    end

                    SUBROUTINE intrinsic_optional_test_function(res, res2, a)
                    integer, dimension(4) :: res
                    integer, dimension(4) :: res2
                    integer :: a
                    integer,dimension(2) :: ret

                    CALL intrinsic_optional_test_function2(res, a)
                    CALL intrinsic_optional_test_function2(res2)
                    CALL get_indices_c(1, 1, 1, ret(1), ret(2), 1, 2)

                    END SUBROUTINE intrinsic_optional_test_function

                    SUBROUTINE intrinsic_optional_test_function2(res, a)
                    integer, dimension(2) :: res
                    integer, optional :: a

                    res(1) = a

                    END SUBROUTINE intrinsic_optional_test_function2

                    SUBROUTINE get_indices_c(i_blk, i_startblk, i_endblk, i_startidx, &
                         i_endidx, irl_start, opt_rl_end)


  INTEGER, INTENT(IN) :: i_blk      ! Current block (variable jb in do loops)
  INTEGER, INTENT(IN) :: i_startblk ! Start block of do loop
  INTEGER, INTENT(IN) :: i_endblk   ! End block of do loop
  INTEGER, INTENT(IN) :: irl_start  ! refin_ctrl level where do loop starts

  INTEGER, OPTIONAL, INTENT(IN) :: opt_rl_end ! refin_ctrl level where do loop ends

  INTEGER, INTENT(OUT) :: i_startidx, i_endidx ! Start and end indices (jc loop)

  ! Local variables

  INTEGER :: irl_end

  IF (PRESENT(opt_rl_end)) THEN
    irl_end = opt_rl_end
  ELSE
    irl_end = 42
  ENDIF

  IF (i_blk == i_startblk) THEN
    i_startidx = 1
    i_endidx   = 42
    IF (i_blk == i_endblk) i_endidx = irl_end 
  ELSE IF (i_blk == i_endblk) THEN
    i_startidx = 1
    i_endidx   = irl_end
  ELSE
    i_startidx = 1
    i_endidx = 42
  ENDIF

END SUBROUTINE get_indices_c

                    """
    sources={}
    sources["adv_intrinsic_optional_test_function"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_optional_test_function", True,sources=sources)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5)

    assert res[0] == 5
    assert res2[0] == 0

if __name__ == "__main__":

    test_fortran_frontend_optional_adv()
