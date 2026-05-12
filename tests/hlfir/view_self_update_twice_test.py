"""Two self-updates on the same view dummy in the same block.

Minimal repro of cloudsc's lines 1369-1378 PLUDE pattern: section-
slice INOUT dummy gets two back-to-back self-updates in the same
inlined-callee block.  Without the Phase I split + view-writeback
cache reset the second update reads through the write-side view,
giving the view ``in=1 (tasklet) + out=2 (writeback + downstream
read)`` — which DaCe's ``get_view_edge`` can't disambiguate.

Two variants:
* plain back-to-back self-updates
* same wrapped in an IF (lifts the cond expression to an interstate
  edge; section_alias + anchor-state safety net must keep this
  working)

E2e against an f2py-compiled reference of the same Fortran source.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_DRIVER_PROLOGUE_HEAD = """
MODULE kernel_mod
CONTAINS
"""

_DRIVER_PROLOGUE_TAIL = """
SUBROUTINE driver(t, klon, klev, nb)
integer, intent(in) :: klon, klev, nb
double precision, intent(inout) :: t(klon, klev, nb)
integer ibl
DO ibl = 1, nb
    CALL accumulate(t(:, :, ibl), klon, klev)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""


def _run(tmp_path, src, name, klon=4, klev=5, nb=3, seed=7):
    ref = f2py(src, tmp_path / 'ref', f'{name}_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=name, entry='_QMkernel_modPdriver').build()

    rng = np.random.default_rng(seed)
    t_in = np.asfortranarray(rng.random((klon, klev, nb)))

    t_ref = np.asfortranarray(t_in.copy())
    ref.kernel_mod.driver(t_ref, klon, klev, nb)

    t = np.asfortranarray(t_in.copy())
    sdfg(t=t, klon=klon, klev=klev, nb=nb)
    np.testing.assert_allclose(t, t_ref, rtol=1e-12, atol=1e-12)


def test_fortran_frontend_view_self_update_twice(tmp_path):
    """Plain back-to-back self-updates on a view dummy — no IF wrap."""
    src = _DRIVER_PROLOGUE_HEAD + """
SUBROUTINE accumulate(x, klon, klev)
integer :: klon, klev
double precision x(klon, klev)
integer i, j
DO j = 1, klev
    DO i = 1, klon
        x(i, j) = x(i, j) + 10.0
        x(i, j) = x(i, j) + 100.0
    ENDDO
ENDDO
END SUBROUTINE accumulate
""" + _DRIVER_PROLOGUE_TAIL
    _run(tmp_path, src, 'view_self_update_twice')


def test_fortran_frontend_view_self_update_twice_in_if(tmp_path):
    """Same back-to-back self-updates, this time wrapped in an IF
    — exactly the cloudsc lines 1364-1385 shape."""
    src = _DRIVER_PROLOGUE_HEAD + """
SUBROUTINE accumulate(x, klon, klev)
integer :: klon, klev
double precision x(klon, klev)
integer i, j
DO j = 1, klev
    DO i = 1, klon
        IF (x(i, j) > 0.5) THEN
            x(i, j) = x(i, j) + 10.0
            x(i, j) = x(i, j) + 100.0
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE accumulate
""" + _DRIVER_PROLOGUE_TAIL
    _run(tmp_path, src, 'view_self_update_twice_if')
