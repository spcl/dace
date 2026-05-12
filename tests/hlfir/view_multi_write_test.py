"""Multiple writes per loop body across distinct section-slice dummies.

Probe inspired by cloudsc's tendency-init loop body: three view
dummies each get a write in the same inlined-callee block.

E2e against an f2py-compiled reference of the same Fortran source.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_view_multi_write(tmp_path):
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE init_tendencies(t, q, a, klon, klev)
integer, intent(in) :: klon, klev
double precision, intent(out) :: t(klon, klev), q(klon, klev), a(klon, klev)
integer i, j
DO j = 1, klev
    DO i = 1, klon
        t(i, j) = 0.0d0
        q(i, j) = 0.0d0
        a(i, j) = 0.0d0
    ENDDO
ENDDO
END SUBROUTINE init_tendencies

SUBROUTINE driver(t, q, a, klon, klev, nb)
integer, intent(in) :: klon, klev, nb
double precision, intent(inout) :: t(klon, klev, nb), q(klon, klev, nb), a(klon, klev, nb)
integer ibl
DO ibl = 1, nb
    CALL init_tendencies(t(:, :, ibl), q(:, :, ibl), a(:, :, ibl), klon, klev)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'view_multi_write_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='view_multi_write', entry='_QMkernel_modPdriver').build()

    klon, klev, nb = 4, 5, 3

    def fresh():
        return np.full([klon, klev, nb], 42.0, order='F', dtype=np.float64)

    t_ref, q_ref, a_ref = fresh(), fresh(), fresh()
    ref.kernel_mod.driver(t_ref, q_ref, a_ref, klon, klev, nb)

    t, q, a = fresh(), fresh(), fresh()
    sdfg(t=t, q=q, a=a, klon=klon, klev=klev, nb=nb)
    np.testing.assert_allclose(t, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q, q_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-12)
