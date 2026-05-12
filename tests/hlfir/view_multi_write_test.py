"""Multiple writes per loop body across distinct section-slice dummies.

Probe inspired by cloudsc's tendency-init loop body:

    DO JK = 1, KLEV
      DO JL = KIDIA, KFDIA
        tendency_loc_T(JL, JK) = 0.0
        tendency_loc_q(JL, JK) = 0.0
        tendency_loc_a(JL, JK) = 0.0
      ENDDO
    ENDDO

Each of T/q/a is a section slice of a rank-3 caller array
(``tendency_loc_T(:, :, IBL)``).  The inlined callee writes each one
inside the same inner-loop body in three back-to-back tasklets.  The
SDFG produced by the bridge ends up with these view AccessNodes
carrying tasklet-write in_edges plus the writeback linking memlet,
which (per the cloudsc dump) trips DaCe's ``get_view_edge`` in some
shape we want to isolate here.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_view_multi_write(tmp_path):
    test_string = """
                    PROGRAM p
                    implicit none
                    integer, parameter :: klon=4, klev=5, nb=3
                    double precision tendency_t(klon, klev, nb)
                    double precision tendency_q(klon, klev, nb)
                    double precision tendency_a(klon, klev, nb)
                    integer ibl
                    DO ibl = 1, nb
                        tendency_t(:, :, ibl) = 42.0
                        tendency_q(:, :, ibl) = 42.0
                        tendency_a(:, :, ibl) = 42.0
                        CALL init_tendencies(tendency_t(:, :, ibl), &
                                             tendency_q(:, :, ibl), &
                                             tendency_a(:, :, ibl), &
                                             klon, klev)
                    ENDDO
                    CALL driver(tendency_t, tendency_q, tendency_a, klon, klev, nb)
                    end

                    SUBROUTINE driver(t, q, a, klon, klev, nb)
                    integer :: klon, klev, nb
                    double precision t(klon, klev, nb), q(klon, klev, nb), a(klon, klev, nb)
                    integer ibl
                    DO ibl = 1, nb
                        CALL init_tendencies(t(:, :, ibl), q(:, :, ibl), a(:, :, ibl), klon, klev)
                    ENDDO
                    END SUBROUTINE driver

                    SUBROUTINE init_tendencies(t, q, a, klon, klev)
                    integer :: klon, klev
                    double precision t(klon, klev), q(klon, klev), a(klon, klev)
                    integer i, j
                    DO j = 1, klev
                        DO i = 1, klon
                            t(i, j) = 0.0
                            q(i, j) = 0.0
                            a(i, j) = 0.0
                        ENDDO
                    ENDDO
                    END SUBROUTINE init_tendencies
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='view_multi_write', entry='_QPdriver').build()

    klon, klev, nb = 4, 5, 3
    t = np.full([klon, klev, nb], 42, order='F', dtype=np.float64)
    q = np.full([klon, klev, nb], 42, order='F', dtype=np.float64)
    a = np.full([klon, klev, nb], 42, order='F', dtype=np.float64)
    sdfg(t=t, q=q, a=a, klon=klon, klev=klev, nb=nb)

    assert np.allclose(t, 0.0)
    assert np.allclose(q, 0.0)
    assert np.allclose(a, 0.0)
