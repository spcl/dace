"""E8 regression: a flattened nested array member loses its
non-default lower bound.

``outer_t`` holds ``arr(2)`` (array of ``inner_t``); ``inner_t`` holds
``v(0:3)`` (lower bound 0).  ``hlfir-flatten-structs`` rewrites
``o%arr(i)%v(j)`` into a flat companion ``o_arr_v`` whose synthesised
``hlfir.declare`` carries only a ``fir.shape`` (extents, no bounds) --
the ``v(0:3)`` lower bound lives solely on the per-access
``hlfir.designate``'s ``fir.shape_shift`` and is discarded when the
designate is rewritten away.  ``resolveLowerBounds`` then falls back
to the SequenceType extents and assigns lb=1 to every flattened dim,
so ``offset_o_arr_v_d1`` is 1 instead of 0 and ``o%arr(1)%v(0)``
indexes element -1.

This is the velocity_tendencies / ICON ``p_patch%pprog(jg)%vn(:,:,
min_rlcell:)`` shape (nested member, negative block lower bound).

f2py's crackfortran can't wrap the derived-type dummy, so the
non-transformed reference is the exact closed-form result; the per-dim
offset constants are the direct correctness signal.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mn
  implicit none
  type inner_t
    real(8) :: v(0:3)
  end type
  type outer_t
    type(inner_t) :: arr(2)
  end type
contains
  subroutine kn(o, out)
    type(outer_t), intent(inout) :: o
    real(8), intent(out) :: out
    out = o%arr(1)%v(0) + o%arr(2)%v(3)
  end subroutine kn
end module mn
"""


def test_flatten_nested_array_nondefault_lb(tmp_path: Path):
    """The flattened companion of ``inner%v(0:3)`` must carry lb 0 in
    its inner dimension (offset 0), not the default 1."""
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, d, name="kn", entry="_QMmnPkn").build()
    sdfg.validate()

    consts = dict(sdfg.constants)
    offs = {k: int(v) for k, v in consts.items() if k.startswith("offset_") and "arr_v" in k}
    # Companion is (arr dim, v dim): arr lb 1, v lb 0.
    assert offs.get("offset_o_arr_v_d0") == 1, offs
    assert offs.get("offset_o_arr_v_d1") == 0, (f"inner v(0:3) lower bound 0 lost in flattening; got {offs}")
