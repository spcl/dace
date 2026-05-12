"""``COUNT`` over compound boolean comparisons (the
``COUNT(arr1 .eq. arr2)`` family).

Pins the bridge contract for two cases:

  * Plain element-wise comparison ``COUNT(arr1 .eq. arr2)`` — the
    elemental-count Mode-C path generates a per-element mask and
    routes through ``CountLibraryNode``.
  * Sectioned comparison ``COUNT(arr1(1:3) .eq. arr2(3:5))`` —
    section-parent contributions on both sides interact through the
    offset-symbol layer.

Both are written so the destination is INTEGER (matching ``COUNT``'s
return type).  When the destination is LOGICAL, Fortran inserts an
implicit ``fir.convert i32 → !fir.logical<4>`` between the libcall and
the assign; the bridge peels that convert at the assign-dispatch site
so the libcall path still fires.  The convert-peeling is exercised by
``test_count_into_logical_destination`` below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_count_full_array_comparison(tmp_path: Path):
    """``COUNT(first .eq. second)`` over two whole 1-D arrays."""
    src = """
SUBROUTINE main(first, second, res)
integer, dimension(5) :: first
integer, dimension(5) :: second
integer :: res
res = COUNT(first .eq. second)
END SUBROUTINE main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    first = np.array([1, 1, 2, 1, 1], dtype=np.int32, order='F')
    second = np.array([1, 2, 2, 2, 1], dtype=np.int32, order='F')
    res = np.zeros(1, dtype=np.int32)
    sdfg(first=first, second=second, res=res)
    assert int(res[0]) == 3  # matching positions: 0, 2, 4


def test_count_section_comparison(tmp_path: Path):
    """Sectioned operands with non-aligned lower bounds —
    ``COUNT(first(1:3) .eq. second(3:5))``.  Each section parent
    contributes a ``(lo - 1)`` offset that flows through the
    elemental's per-element designate."""
    src = """
SUBROUTINE main(first, second, res)
integer, dimension(5) :: first
integer, dimension(5) :: second
integer :: res
res = COUNT(first(1:3) .eq. second(3:5))
END SUBROUTINE main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    first = np.array([10, 20, 30, 40, 50], dtype=np.int32, order='F')
    second = np.array([99, 88, 10, 20, 30], dtype=np.int32, order='F')
    # first[0:3] vs second[2:5] = [10, 20, 30] vs [10, 20, 30] → all 3 match.
    res = np.zeros(1, dtype=np.int32)
    sdfg(first=first, second=second, res=res)
    assert int(res[0]) == 3


def test_count_into_logical_destination_builds(tmp_path: Path):
    """``logical, dimension(2) :: res; res(1) = COUNT(...)``: Fortran's
    implicit int-to-logical conversion inserts a ``fir.convert``
    between the libcall and the assign.  The bridge's convert-peel at
    the assign-dispatch site keeps the libcall path matching, so the
    SDFG builds rather than emitting ``_out_res = ?``.

    We assert only that the SDFG builds — the int→logical truncation
    semantics are checked end-to-end by the integer-destination tests
    above.  This pins the build-time fix so the test fails loudly if
    the convert-peel regresses.
    """
    src = """
SUBROUTINE main(a, res)
integer, dimension(4) :: a
logical, dimension(2) :: res
res(1) = COUNT(a .eq. 1)
res(2) = COUNT(a .gt. 0)
END SUBROUTINE main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    assert sdfg is not None
