"""Two inlined call sites of the same callee against different arrays.

Minimal repro of the structural shape that used to break `view_test_2`
(multiple call sites of `viewlens` against different sections of the
same parent): the callee body is non-trivial enough that the inliner
keeps two separate ``hlfir.declare`` ops with the same
``uniq_name = "_QFbarEx"`` after ``hlfir-inline-all`` — one per call
site — so the bridge has to disambiguate them per call site instead
of collapsing to one VarInfo.

Two variants:

1. ``..._whole_array``: bar receives a whole array argument.  Routes
   through the bridge's per-SSA alias trace back to the caller's
   array; no view-alias machinery involved.
2. ``..._section_slice``: bar receives a section slice.  Activates
   the view-alias path; covered by Pass 0b in
   ``extract_vars.cpp`` (per-occurrence rename of section-slice
   inlined declares) plus the view-writeback link in
   ``emit_tasklet.py`` (for self-update accesses inside the inlined
   body).

In both variants `bar` operates on a slice of its dummy
(``DO i = 2, 9: x(i) = x(i) * 2`` then the same loop for ``sin``) —
self-update accesses through the inlined dummy, which is what tripped
the view writeback path before the Phase I split fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_BAR_DEF = """
                    SUBROUTINE bar(x)
                    double precision x(10)
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
    """Two `bar` calls, one against `a`, one against `b` — whole arrays.

    Whole-array dummies route through per-SSA alias trace, so the
    multi-callsite VarInfo collapse is harmless.
    """
    test_string = f"""
                    PROGRAM multi_callsite_test_program
                    implicit none
                    double precision a(10), b(10)
                    CALL multi_callsite_test_function(a, b)
                    end

                    SUBROUTINE multi_callsite_test_function(a, b)
                    double precision a(10), b(10)
                    integer i

                    DO i = 1, 10
                        a(i) = i
                        b(i) = i * 100
                    ENDDO

                    CALL bar(a)
                    CALL bar(b)

                    END SUBROUTINE multi_callsite_test_function
{_BAR_DEF}
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='multi_callsite_whole',
                      entry='_QPmulti_callsite_test_function').build()

    a = np.full([10], 0, order="F", dtype=np.float64)
    b = np.full([10], 0, order="F", dtype=np.float64)
    sdfg(a=a, b=b)

    # bar transforms only x(2:9) (Fortran 1-based) = [1:9] in NumPy.
    init_a = np.arange(1, 11, dtype=np.float64)
    init_b = init_a * 100
    expected_a = init_a.copy()
    expected_a[1:9] = np.sin(expected_a[1:9] * 2)
    expected_b = init_b.copy()
    expected_b[1:9] = np.sin(expected_b[1:9] * 2)
    assert np.allclose(a, expected_a)
    assert np.allclose(b, expected_b)


def test_fortran_frontend_multi_callsite_section_slice(tmp_path):
    """Two `bar` calls, each on a section slice of a different 2-D array.

    Column 2 of `a` and column 2 of `b` each get the full `x*2; sin(x)`
    transform.  Pass 0b renames the inlined ``_QFbarEx`` declares to
    ``x_call0`` / ``x_call1`` per call site so each gets its own
    view_subset / view_source pointing at the right caller column.
    """
    test_string = f"""
                    PROGRAM multi_callsite_test_program
                    implicit none
                    double precision a(10, 5), b(10, 5)
                    CALL multi_callsite_test_function(a, b)
                    end

                    SUBROUTINE multi_callsite_test_function(a, b)
                    double precision a(10, 5), b(10, 5)
                    integer i, j

                    DO j = 1, 5
                        DO i = 1, 10
                            a(i, j) = i + 10 * j
                            b(i, j) = (i + 10 * j) * 100
                        ENDDO
                    ENDDO

                    CALL bar(a(:, 2))
                    CALL bar(b(:, 2))

                    END SUBROUTINE multi_callsite_test_function
{_BAR_DEF}
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='multi_callsite_slice',
                      entry='_QPmulti_callsite_test_function').build()

    a = np.full([10, 5], 0, order="F", dtype=np.float64)
    b = np.full([10, 5], 0, order="F", dtype=np.float64)
    sdfg(a=a, b=b)

    # bar transforms only x(2:9) of its (sliced) dummy.
    # Column 2 (Fortran 1-based) = a[:, 1] in NumPy; initial values
    # set by the j=2 fill loop: a[i, 1] = (i+1) + 20.
    init_a_col2 = np.arange(1, 11, dtype=np.float64) + 10 * 2
    init_b_col2 = init_a_col2 * 100
    expected_a_col2 = init_a_col2.copy()
    expected_a_col2[1:9] = np.sin(expected_a_col2[1:9] * 2)
    expected_b_col2 = init_b_col2.copy()
    expected_b_col2[1:9] = np.sin(expected_b_col2[1:9] * 2)
    assert np.allclose(a[:, 1], expected_a_col2)
    assert np.allclose(b[:, 1], expected_b_col2)

    # Column 1 should be untouched (initial values from the fill loop).
    expected_a_col1 = np.arange(1, 11, dtype=np.float64) + 10
    assert np.allclose(a[:, 0], expected_a_col1)
