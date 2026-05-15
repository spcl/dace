"""Arithmetic-bearing array read inside an IF condition.

Minimal repro of the bridge gap from cloudsc_full line 2140:

    IF (ZLCOND2(JL) < RLMIN .OR. (1.0 - ZA(JL, JK)) < ZEPSEC) THEN

The OR-arm reads ``ZA(JL, JK)`` inside an arithmetic sub-expression
``1.0 - ZA(JL, JK)``.  Flang wraps the parenthesised expression in
``hlfir.no_reassoc`` and the cond ultimately lifts to an interstate-
edge assignment.  The bridge's ``buildExprWithSubscripts`` must peel
``hlfir.no_reassoc`` (and recurse through ``arith.subf``) so the
inner load's subscript survives  --  otherwise C++ codegen rejects
``1 - za`` (int - double*).

E2e against an f2py-compiled reference of the same Fortran source.
"""

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_if_arith_array_read(tmp_path):
    src = """
SUBROUTINE filter(za, zlcond2, out, klon, klev)
integer :: klon, klev
double precision za(klon, klev), zlcond2(klon)
integer :: out(klon, klev)
integer i, j
double precision, parameter :: rlmin = 0.5
double precision, parameter :: zepsec = 0.5
DO j = 1, klev
    DO i = 1, klon
        out(i, j) = 0
        IF (zlcond2(i) < rlmin .OR. (1.0 - za(i, j)) < zepsec) THEN
            out(i, j) = 1
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE filter
"""
    ref = f2py(src, tmp_path / 'ref', 'filter_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='filter', entry='_QPfilter').build()

    rng = np.random.default_rng(7)
    klon, klev = 4, 3
    za = np.asfortranarray(rng.random((klon, klev)))
    zlcond2 = np.asfortranarray(rng.random(klon))

    out_ref = np.zeros((klon, klev), order='F', dtype=np.int32)
    ref.filter(za=za, zlcond2=zlcond2, out=out_ref)

    out = np.zeros((klon, klev), order='F', dtype=np.int32)
    sdfg(za=za, zlcond2=zlcond2, out=out, klon=klon, klev=klev)
    np.testing.assert_array_equal(out, out_ref)
