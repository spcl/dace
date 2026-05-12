"""Conditional zero-or-keep pattern that the cloudsc kernel uses for PLUDE.

Minimal repro of the bridge gap that causes cloudsc_full to leave
87/548 elements of PLUDE at 0.0 where the f2py reference has a
non-zero value.  The Fortran shape (lines 1768-1788 of
``cloudscexp2_simplified.F90``):

    IF (JK < KLEV .AND. JK >= NCLDTOP) THEN
      DO JL = ...
        PLUDE(JL, JK) = PLUDE(JL, JK) * ZDTGDP(JL)        ! self-mul
        IF (LDCUM(JL) .AND. PLUDE(JL, JK) > RLMIN          ! 3-conjunct
                       .AND. PLU(JL, JK+1) > ZEPSEC) THEN
          ! ... uses PLUDE(JL, JK), doesn't write it
        ELSE
          PLUDE(JL, JK) = 0.0                              ! ZERO branch
        ENDIF
      ENDDO
    ENDIF

Key shape elements:
* PLUDE is INTENT(INOUT) on a section slice (cloudsc_full's
  ``PLUDE(:, :, IBL)``); after section_alias the writes route
  to the parent.
* The IF reads PLUDE AFTER the self-mul; the condition's PLUDE
  value depends on the just-written ``PLUDE * ZDTGDP``.
* The ELSE branch unconditionally zeroes PLUDE.
* Outer IF gates on JK/loop iter (``JK < KLEV .AND. JK >=
  NCLDTOP``) — cells outside the JK range must keep their input.

The full-cloudsc symptom is that the SDFG always takes the ELSE
branch on certain (JK, IBL) cells where the reference takes the
THEN branch.  This minimal kernel currently PASSES — the cloudsc
gap requires more context than is captured here.  Keeping the
test as a structural regression guard for the self-update + 3-
conjunct conditional pattern; if it ever regresses, that's a
smaller debugging surface than the full cloudsc.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cloudsc_plude_conditional_zero(tmp_path):
    test_string = """
                    SUBROUTINE driver(plude, plu, ldcum, zgdp, ptsphy, rlmin, &
                                      klon, klev, nblocks, ncldtop)
                    integer :: klon, klev, nblocks, ncldtop
                    double precision plude(klon, klev, nblocks)
                    double precision plu(klon, klev, nblocks)
                    logical ldcum(klon, nblocks)
                    double precision zgdp(klon)
                    double precision ptsphy
                    double precision rlmin
                    integer ibl
                    DO ibl = 1, nblocks
                        CALL kernel(plude(:, :, ibl), plu(:, :, ibl), &
                                    ldcum(:, ibl), zgdp, ptsphy, rlmin, klon, klev, ncldtop)
                    ENDDO
                    END SUBROUTINE driver

                    SUBROUTINE kernel(plude, plu, ldcum, zgdp, ptsphy, rlmin, &
                                      klon, klev, ncldtop)
                    integer :: klon, klev, ncldtop
                    double precision, intent(inout) :: plude(klon, klev)
                    double precision plu(klon, klev)
                    logical ldcum(klon)
                    double precision zgdp(klon)
                    double precision ptsphy
                    double precision rlmin
                    integer jk, jl
                    ! Local arrays + scalars body-init — mirrors cloudsc
                    ! line 1255 ``ZEPSEC = 1.E-14`` and line 1589
                    ! ``ZDTGDP(JL) = PTSPHY * ZGDP(JL)``.
                    double precision zepsec, zdtgdp(klon)
                    zepsec = 1.0d-12
                    DO jk = ncldtop, klev
                        ! Mimic cloudsc: ZDTGDP is computed locally each
                        ! JK iteration in a separate loop block.
                        IF (jk < klev) THEN
                            DO jl = 1, klon
                                zdtgdp(jl) = ptsphy * zgdp(jl)
                            ENDDO
                            DO jl = 1, klon
                                plude(jl, jk) = plude(jl, jk) * zdtgdp(jl)
                                IF (ldcum(jl) .AND. plude(jl, jk) > rlmin &
                                              .AND. plu(jl, jk+1) > zepsec) THEN
                                    ! keep plude as-is (no write)
                                ELSE
                                    plude(jl, jk) = 0.0d0
                                ENDIF
                            ENDDO
                        ENDIF
                    ENDDO
                    END SUBROUTINE kernel
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='driver', entry='_QPdriver').build()

    klon, klev, nblocks = 1, 137, 4
    ncldtop = 15
    rng = np.random.default_rng(42)
    plude_in = rng.random((klon, klev, nblocks))
    plu = rng.random((klon, klev, nblocks))
    ldcum = rng.integers(0, 2, (klon, nblocks)).astype(np.bool_)
    zgdp = rng.random((klon, ))
    ptsphy_val = rng.random()
    rlmin_val = rng.random()

    # Expected (Python ref): mirrors the kernel logic exactly.
    expected = plude_in.copy(order='F')
    for ibl in range(nblocks):
        zdtgdp_local = ptsphy_val * zgdp.copy()
        for jk in range(klev):
            jk1 = jk + 1  # Fortran 1-based
            if jk1 < klev and jk1 >= ncldtop:
                for jl in range(klon):
                    expected[jl, jk, ibl] *= zdtgdp_local[jl]
                    cond = (ldcum[jl, ibl] and expected[jl, jk, ibl] > rlmin_val and plu[jl, jk + 1, ibl] > 1e-12)
                    if not cond:
                        expected[jl, jk, ibl] = 0.0

    plude = np.asfortranarray(plude_in.copy())
    from dace.data import Scalar

    def _route(name, val, dtype):
        return val if isinstance(sdfg.arglist().get(name), Scalar) else np.array([val], dtype=dtype)

    sdfg(plude=plude,
         plu=np.asfortranarray(plu),
         ldcum=np.asfortranarray(ldcum),
         zgdp=np.asfortranarray(zgdp),
         ptsphy=_route('ptsphy', ptsphy_val, np.float64),
         rlmin=_route('rlmin', rlmin_val, np.float64),
         klon=klon,
         klev=klev,
         nblocks=nblocks,
         ncldtop=ncldtop)

    np.testing.assert_allclose(plude, expected, rtol=1e-12, atol=1e-12)
