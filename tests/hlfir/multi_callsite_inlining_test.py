"""Two inlined call sites of the same callee against different arrays.

Reduces the cloudsc shape that broke view_test_2 (multi-callsite
section-slice dummies) to a minimal form.  Both variants compile the
SAME Fortran source via f2py for the reference and via the bridge
for the SDFG  --  e2e per ``feedback_e2e_numerical``.

1. ``..._whole_array``: bar receives a whole array.  Routes through
   the bridge's per-SSA alias trace; no view-alias machinery
   involved.
2. ``..._section_slice``: bar receives a section slice.  Activates
   the section_alias path (Pass 0b multi-callsite rename) so each
   call site gets its own VarInfo.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_BAR_DEF = """
SUBROUTINE bar(x)
double precision, intent(inout) :: x(10)
integer i
DO i = 2, 9
    x(i) = x(i) * 2
ENDDO
DO i = 2, 9
    x(i) = sin(x(i))
ENDDO
END SUBROUTINE bar
"""


def test_fortran_frontend_multi_callsite_whole_array(tmp_path):
    """Two ``bar`` calls, one against ``a``, one against ``b``  --  whole arrays."""
    src = """
MODULE kernel_mod
CONTAINS
""" + _BAR_DEF + """
SUBROUTINE driver(a, b)
double precision, intent(inout) :: a(10), b(10)
integer i

DO i = 1, 10
    a(i) = i
    b(i) = i * 100
ENDDO

CALL bar(a)
CALL bar(b)
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'multi_callsite_whole_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='multi_callsite_whole', entry='_QMkernel_modPdriver').build()

    a_ref = np.zeros(10, dtype=np.float64)
    b_ref = np.zeros(10, dtype=np.float64)
    ref.kernel_mod.driver(a_ref, b_ref)

    a = np.zeros(10, dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)
    sdfg(a=a, b=b)
    np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, b_ref, rtol=1e-12, atol=1e-12)


def test_fortran_frontend_multi_callsite_section_slice(tmp_path):
    """Two ``bar`` calls, each on a section slice of a different 2-D array."""
    src = """
MODULE kernel_mod
CONTAINS
""" + _BAR_DEF + """
SUBROUTINE driver(a, b)
double precision, intent(inout) :: a(10, 5), b(10, 5)
integer i, j

DO j = 1, 5
    DO i = 1, 10
        a(i, j) = i + 10 * j
        b(i, j) = (i + 10 * j) * 100
    ENDDO
ENDDO

CALL bar(a(:, 2))
CALL bar(b(:, 2))
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'multi_callsite_slice_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='multi_callsite_slice', entry='_QMkernel_modPdriver').build()

    a_ref = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    b_ref = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    ref.kernel_mod.driver(a_ref, b_ref)

    a = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    b = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    sdfg(a=a, b=b)
    np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, b_ref, rtol=1e-12, atol=1e-12)
