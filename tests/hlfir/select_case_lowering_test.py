"""End-to-end coverage for the bridge's ``lower-fir-select-case`` pass.

The pass lowers ``fir.select_case`` to chains of ``arith.cmp`` +
``cf.cond_br`` *before* ``hlfir-inline-all`` runs.  Without it the
upstream MLIR inliner segfaults whenever a callee contains
``SELECT CASE`` — see [LowerFirSelectCase.cpp](../../dace/frontend/hlfir/passes/LowerFirSelectCase.cpp).

These tests pin two small Fortran programs that drive the pass in its
two interesting shapes:

  * **point case** (``case(1)``)   — ``selector == 1``
  * **closed interval** (``case(2:4)``) — ``(selector >= 2) AND (selector <= 4)``

Both place the ``SELECT CASE`` inside a module-contained subroutine
called from ``main`` so the inliner has to actually clone the lowered
CFG into the caller (which is the original segfault path).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_select_case_point_in_inlined_callee(tmp_path: Path):
    """``select case (v) case(1) v=5 end select`` inside a module
    subroutine — exercises the point-case ``selector == 1`` lowering
    + the inline-all path the pass was added to unblock."""
    src = """
module lib
  implicit none
contains
  subroutine foo(v)
    integer, intent(inout) :: v
    select case(v)
    case(1)
      v = 5
    end select
  end subroutine foo
end module lib

subroutine main(d)
  use lib
  implicit none
  integer :: d(5)
  call foo(d(1))
  call foo(d(2))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.array([1, 7, 0, 0, 0], dtype=np.int32, order="F")
    sdfg(d=a)
    # d(1) was 1 → matched, becomes 5; d(2) was 7 → no match, unchanged.
    assert a[0] == 5
    assert a[1] == 7


def test_select_case_interval_in_inlined_callee(tmp_path: Path):
    """``case(2:4)`` lowers to ``v >= 2 AND v <= 4`` — covers the
    closed-interval branch of LowerFirSelectCase."""
    src = """
module lib
  implicit none
contains
  subroutine foo(v)
    integer, intent(inout) :: v
    select case(v)
    case(2:4)
      v = 6
    end select
  end subroutine foo
end module lib

subroutine main(d)
  use lib
  implicit none
  integer :: d(6)
  call foo(d(1))
  call foo(d(2))
  call foo(d(3))
  call foo(d(4))
  call foo(d(5))
  call foo(d(6))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32, order="F")
    sdfg(d=a)
    # Only 2..4 match the case (2:4) range → become 6; 1, 5, 0 stay put.
    assert a[0] == 1
    assert a[1] == 6
    assert a[2] == 6
    assert a[3] == 6
    assert a[4] == 5
    assert a[5] == 0
