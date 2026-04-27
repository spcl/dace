"""Verbatim port of f2dace/dev:tests/fortran/advanced_optional_args_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from ported._helpers import xfail

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@xfail("inlined-call constants (i_blk=1, i_startblk=1, i_endblk=1, irl_start=1, opt_rl_end=2) "
       "not folded into the inner ``get_indices_c`` dummies — they're surfaced on the SDFG "
       "signature instead, breaking the host call")
def test_fortran_frontend_optional_adv(tmp_path):
    src = """
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
    """
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=np.array([5], dtype=np.int32), a_present=1)

    # Safe path only — second internal ``call fun(res2)`` reads
    # OPTIONAL ``a`` without checking PRESENT and is UB per Fortran;
    # res2 is left unchecked.  The ``get_indices_c`` call exercises
    # the present-guarded path and stays implicit.
    assert res[0] == 5
