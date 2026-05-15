"""Conditional zero-or-keep pattern from cloudsc's PLUDE block.

Mirrors ``cloudscexp2_simplified.F90`` lines 1768-1788:

    IF (JK < KLEV .AND. JK >= NCLDTOP) THEN
      DO JL = ...
        PLUDE(JL, JK) = PLUDE(JL, JK) * ZDTGDP(JL)
        IF (LDCUM(JL) .AND. PLUDE(JL, JK) > RLMIN
                       .AND. PLU(JL, JK+1) > ZEPSEC) THEN
          ! ... uses PLUDE(JL, JK), doesn't write it
        ELSE
          PLUDE(JL, JK) = 0.0
        ENDIF
      ENDDO
    ENDIF

Section-slice INOUT dummy, self-update, 3-conjunct IF reading the
just-written value, ELSE branch unconditionally zeroes.  Structural
regression guard for the section_alias + view-writeback contracts.

E2e against an f2py-compiled reference of the same Fortran source.
"""

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cloudsc_plude_conditional_zero(tmp_path):
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE kernel(plude, plu, ldcum, zgdp, ptsphy, rlmin, klon, klev, ncldtop)
integer, intent(in) :: klon, klev, ncldtop
double precision, intent(inout) :: plude(klon, klev)
double precision, intent(in) :: plu(klon, klev)
logical, intent(in) :: ldcum(klon)
double precision, intent(in) :: zgdp(klon)
double precision, intent(in) :: ptsphy
double precision, intent(in) :: rlmin
integer jk, jl
double precision zepsec, zdtgdp(klon)
zepsec = 1.0d-12
DO jk = ncldtop, klev
    IF (jk < klev) THEN
        DO jl = 1, klon
            zdtgdp(jl) = ptsphy * zgdp(jl)
        ENDDO
        DO jl = 1, klon
            plude(jl, jk) = plude(jl, jk) * zdtgdp(jl)
            IF (ldcum(jl) .AND. plude(jl, jk) > rlmin &
                          .AND. plu(jl, jk+1) > zepsec) THEN
                ! keep
            ELSE
                plude(jl, jk) = 0.0d0
            ENDIF
        ENDDO
    ENDIF
ENDDO
END SUBROUTINE kernel

SUBROUTINE driver(plude, plu, ldcum, zgdp, ptsphy, rlmin, klon, klev, nblocks, ncldtop)
integer, intent(in) :: klon, klev, nblocks, ncldtop
double precision, intent(inout) :: plude(klon, klev, nblocks)
double precision, intent(in) :: plu(klon, klev, nblocks)
logical, intent(in) :: ldcum(klon, nblocks)
double precision, intent(in) :: zgdp(klon)
double precision, intent(in) :: ptsphy
double precision, intent(in) :: rlmin
integer ibl
DO ibl = 1, nblocks
    CALL kernel(plude(:, :, ibl), plu(:, :, ibl), &
                ldcum(:, ibl), zgdp, ptsphy, rlmin, klon, klev, ncldtop)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'cloudsc_plude_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='cloudsc_plude', entry='_QMkernel_modPdriver').build()

    klon, klev, nblocks = 1, 137, 4
    ncldtop = 15
    rng = np.random.default_rng(42)
    plude_in = np.asfortranarray(rng.random((klon, klev, nblocks)))
    plu = np.asfortranarray(rng.random((klon, klev, nblocks)))
    ldcum = np.asfortranarray(rng.integers(0, 2, (klon, nblocks)).astype(np.bool_))
    zgdp = np.asfortranarray(rng.random((klon, )))
    ptsphy_val = rng.random()
    rlmin_val = rng.random()

    plude_ref = np.asfortranarray(plude_in.copy())
    ref.kernel_mod.driver(plude=plude_ref,
                          plu=plu,
                          ldcum=ldcum,
                          zgdp=zgdp,
                          ptsphy=ptsphy_val,
                          rlmin=rlmin_val,
                          ncldtop=ncldtop)

    plude = np.asfortranarray(plude_in.copy())
    from dace.data import Scalar

    def _route(name, val, dtype):
        return val if isinstance(sdfg.arglist().get(name), Scalar) else np.array([val], dtype=dtype)

    sdfg(plude=plude,
         plu=plu,
         ldcum=ldcum,
         zgdp=zgdp,
         ptsphy=_route('ptsphy', ptsphy_val, np.float64),
         rlmin=_route('rlmin', rlmin_val, np.float64),
         klon=klon,
         klev=klev,
         nblocks=nblocks,
         ncldtop=ncldtop)
    np.testing.assert_allclose(plude, plude_ref, rtol=1e-12, atol=1e-12)
