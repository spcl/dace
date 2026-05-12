"""Arithmetic-bearing array read inside an IF condition.

Minimal repro of the next bridge gap surfacing in cloudsc_full (line
2140 of ``cloudscexp2_simplified.F90``):

    IF (ZLCOND2(JL) < RLMIN .OR. (1.0 - ZA(JL, JK)) < ZEPSEC) THEN

The OR-arm reads ``ZA(JL, JK)`` inside an arithmetic sub-expression
(``1.0 - ZA(JL, JK)``).  The bridge's ``buildExprWithSubscripts`` /
``buildBoolExpr`` walk renders the comparison correctly when the
array read is the top-level cmp operand, but drops the subscript on
``ZA`` when it sits inside ``arith.subf``.  The lifted interstate-
edge assignment ends up with ``1 - za`` (bare array name) on the
RHS, which the C++ codegen then renders as ``1 - za`` against the
array pointer — ``invalid operands of types 'int' and 'double*' to
binary 'operator-'``.

The condition needs to render as ``1.0 - za[(_it_jl)-1, (_it_jk)-1]``
(or routed through the same scalar-tasklet lift that array reads in
condition expressions go through).
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_if_arith_array_read(tmp_path):
    test_string = """
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
    sdfg = build_sdfg(test_string, tmp_path, name='filter', entry='_QPfilter').build()

    klon, klev = 4, 3
    za = np.full([klon, klev], 0.9, order='F', dtype=np.float64)
    zlcond2 = np.full([klon], 1.0, order='F', dtype=np.float64)
    out = np.full([klon, klev], 42, order='F', dtype=np.int32)
    # za=0.9 -> (1.0 - 0.9) = 0.1 < zepsec=0.5 -> True -> out=1 everywhere.
    sdfg(za=za, zlcond2=zlcond2, out=out, klon=klon, klev=klev)
    assert np.all(out == 1)
