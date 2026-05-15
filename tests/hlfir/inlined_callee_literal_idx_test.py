"""Lower-bound inference through an inlined subroutine call.

Pinpointed in ``velocity_full`` bisection: ICON's pattern is

    SUBROUTINE outer(arr, ...)
      INTEGER, ALLOCATABLE :: arr(:)
      CALL inner(arr, ..., -5)        ! literal at the call site
    END SUBROUTINE
    SUBROUTINE inner(arr, ..., irl_end)
      INTEGER, INTENT(IN) :: irl_end
      INTEGER :: local
      local = irl_end                  ! stash into a local
      x = arr(local)                   ! access via fir.load
    END SUBROUTINE

After ``hlfir-inline-all`` + ``hlfir-flatten-structs`` the
``inner`` body is spliced into ``outer``.  The designate index
becomes ``fir.load %local_decl`` rather than ``arith.constant -5``,
so ``inferLowerBoundsFromLiteralAccesses`` misses the literal.
The bridge then defaults ``offset_arr_d0 = 1`` and ``arr(-5)``
lowers to ``arr[-6]`` -> segfault at runtime.

This test pins the inference to follow the inline-callee load/store
chain.  Currently the bridge only matches when the designate index
is a direct ``arith.constant``; the file documents the gap so the
next bridge fix has a regression gate.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_callee
  implicit none
  contains
  subroutine read_end_index(arr, irl_end, out)
    integer, allocatable, intent(in) :: arr(:)
    integer, intent(in) :: irl_end
    integer, intent(out) :: out
    integer :: local
    local = irl_end
    out = arr(local)
  end subroutine read_end_index
end module mo_callee

subroutine outer(arr, out)
  use mo_callee, only: read_end_index
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(out) :: out
  call read_end_index(arr, -5, out)
end subroutine outer
"""


def test_inlined_callee_propagates_negative_literal(tmp_path: Path):
    """Caller passes literal ``-5`` to inlined callee; inner body reads
    ``arr(local)`` where ``local = -5`` was stored.

    After bridge inlining, the designate index is a ``fir.load``
    of the local's storage, not a raw ``arith.constant``.  The
    inference must trace through the load/store chain to recover
    ``-5`` and specialise ``offset_arr_d0`` to ``-5``.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="outer", entry="_QPouter").build()
    sdfg.validate()

    inferred_offset = dict(sdfg.constants).get('offset_arr_d0')
    assert inferred_offset == -5, (f"expected offset_arr_d0 == -5 (literal propagated through "
                                   f"inlined subroutine + load/store chain); got {inferred_offset}.  "
                                   f"This is the bridge gap identified in velocity_full bisection.")

    arr = np.asfortranarray(np.array([100, 200, 300, 400, 500], dtype=np.int32))  # 5 elements
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out, arr_d0=np.int64(5))
    assert out[0] == 100, f"arr(-5) (first element with lb=-5) should be 100; got {out[0]}"
