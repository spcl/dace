"""Same array element used in MAX and MIN within one expression.

Tight repro of the cloudsc_full PCOVPTOT bug.  Fortran source mimics
line 2436-2438 of cloudscexp2_simplified.F90:

    ZCOVPTOT(JL) = 1.0 - (1.0 - ZCOVPTOT(JL))
                       * (1.0 - MAX(ZA(JL, JK), ZA(JL, JK-1)))
                       / (1.0 - MIN(ZA(JL, JK-1), 1.0 - 1.0E-06))

``ZA(JL, JK-1)`` appears TWICE in the expression — as the second
operand of MAX and as the first operand of MIN.  Flang's CSE merges
the two loads into one SSA value; the bridge's expression builder
must then emit a tasklet whose code references that SHARED operand
correctly in BOTH places.

The cloudsc SDFG generated:

    _out_zcovptot = ... max(_in_za_0, _in_za_1)
                  / ... min(_in_za_0, 0.999...)
                       ^^^^^^^^^^^^^^ should be _in_za_1

Where ``_in_za_0`` reads ``za[jl, jk]`` and ``_in_za_1`` reads
``za[jl, jk-1]``.  The MIN's first operand got the wrong connector,
so the expression silently computed against the current JK instead
of the previous one.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_max_min_shared_load(tmp_path):
    test_string = """
                    SUBROUTINE cov_update(zcov, za, klon, klev)
                    integer :: klon, klev
                    double precision, intent(inout) :: zcov(klon)
                    double precision za(klon, klev)
                    integer jl, jk
                    DO jk = 2, klev
                        DO jl = 1, klon
                            zcov(jl) = 1.0d0 - (1.0d0 - zcov(jl)) &
                                * (1.0d0 - MAX(za(jl, jk), za(jl, jk-1))) &
                                / (1.0d0 - MIN(za(jl, jk-1), 1.0d0 - 1.0d-6))
                        ENDDO
                    ENDDO
                    END SUBROUTINE cov_update
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='cov_update', entry='_QPcov_update').build()

    klon, klev = 1, 6
    # Pick values that distinguish jk from jk-1:
    # za[*, 0] = 0.5, za[*, 1] = 0.2, za[*, 2] = 0.3, ...
    za = np.array([[0.5, 0.2, 0.3, 0.7, 0.4, 0.1]], dtype=np.float64, order='F')
    zcov_in = np.array([0.0], dtype=np.float64)

    # Python reference: mirror exactly.
    zcov_ref = zcov_in.copy()
    for jk in range(1, klev):  # 0-based: jk in [1, klev-1]
        for jl in range(klon):
            zcov_ref[jl] = 1.0 - (1.0 - zcov_ref[jl]) \
                * (1.0 - max(za[jl, jk], za[jl, jk-1])) \
                / (1.0 - min(za[jl, jk-1], 1.0 - 1.0e-6))

    zcov = zcov_in.copy()
    sdfg(zcov=zcov, za=np.asfortranarray(za), klon=klon, klev=klev)
    np.testing.assert_allclose(zcov, zcov_ref, rtol=1e-12, atol=1e-12)
