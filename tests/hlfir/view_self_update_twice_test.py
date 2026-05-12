"""Two self-updates on the same view dummy in the same conditional block.

Minimal repro of the cloudsc state `s_62` view-edge ambiguity (the
first bridge gap in cloudsc_full).

Pattern (from cloudsc lines 1369-1378, inside an IF body):

    tendency_loc_q(JL, JK) = tendency_loc_q(JL, JK) + ZQADJ    ! self-update #1
    ...
    tendency_loc_q(JL, JK) = tendency_loc_q(JL, JK) + ZQADJ2   ! self-update #2

The first self-update triggers the Phase I split (read+write of the
same view in one tasklet must hit different access nodes to keep the
state DAG acyclic).  The Phase I split mints a fresh write-side view
and the bridge's view-writeback linking edge.  Without the
``cache.pop`` in ``_ensure_view_writeback_link`` the SECOND
self-update routes its read through the write-side view from #1 —
giving that view ``in=1 (tasklet write) + out=2 (writeback + downstream
read)``, which DaCe's ``get_view_edge`` can't disambiguate.

Fix: after the writeback link is added, drop ``target`` from the
per-state cache so subsequent reads mint a fresh read-side view with
its own ``src->view`` linking.  Then the write-side has the clean
``in=1 / out=1`` shape ``get_view_edge`` recognises.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_DRIVER_PROLOGUE = """
                    PROGRAM p
                    implicit none
                    integer, parameter :: klon=4, klev=5, nb=3
                    double precision t(klon, klev, nb)
                    integer ibl
                    DO ibl = 1, nb
                        t(:, :, ibl) = 1.0
                    ENDDO
                    CALL driver(t, klon, klev, nb)
                    end

                    SUBROUTINE driver(t, klon, klev, nb)
                    integer :: klon, klev, nb
                    double precision t(klon, klev, nb)
                    integer ibl
                    DO ibl = 1, nb
                        CALL accumulate(t(:, :, ibl), klon, klev)
                    ENDDO
                    END SUBROUTINE driver
"""


def test_fortran_frontend_view_self_update_twice(tmp_path):
    """Plain back-to-back self-updates on a view dummy — no IF wrap."""
    test_string = _DRIVER_PROLOGUE + """
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
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='view_self_update_twice', entry='_QPdriver').build()

    klon, klev, nb = 4, 5, 3
    t = np.full([klon, klev, nb], 1.0, order='F', dtype=np.float64)
    sdfg(t=t, klon=klon, klev=klev, nb=nb)
    assert np.allclose(t, 111.0)


def test_fortran_frontend_view_self_update_twice_in_if(tmp_path):
    """Same back-to-back self-updates, this time wrapped in an IF
    — exactly the cloudsc lines 1364-1385 shape."""
    test_string = _DRIVER_PROLOGUE + """
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
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='view_self_update_twice_if', entry='_QPdriver').build()

    klon, klev, nb = 4, 5, 3
    t = np.full([klon, klev, nb], 1.0, order='F', dtype=np.float64)
    sdfg(t=t, klon=klon, klev=klev, nb=nb)
    assert np.allclose(t, 111.0)
