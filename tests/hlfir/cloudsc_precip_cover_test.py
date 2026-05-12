"""Precip-cover MAX-RAN overlap formula from CLOUDSC.

Lifts the cumulative cover update block (cloudscexp2_simplified.F90
lines 2418-2451) that maintains ``ZCOVPTOT`` across the JK column.
This is the block where the cloudsc_full integration test sees its
PCOVPTOT pattern A (~1e-3 decreasing offset at consecutive JKs):

    DO JL = KIDIA, KFDIA
      IF (ZQPRETOT(JL) > ZEPSEC) THEN
        ZCOVPTOT(JL) = 1.0 - ((1.0 - ZCOVPTOT(JL)) &
         & * (1.0 - MAX(ZA(JL,JK), ZA(JL,JK-1))) &
         & / (1.0 - MIN(ZA(JL,JK-1), 1.0 - 1.E-06)) )
        ZCOVPTOT(JL) = MAX(ZCOVPTOT(JL), RCOVPMIN)
        ZCOVPCLR(JL) = MAX(0.0, ZCOVPTOT(JL) - ZA(JL,JK))
        ZRAINCLD(JL) = ZQXFG(JL,NCLDQR) / ZCOVPTOT(JL)
        ZSNOWCLD(JL) = ZQXFG(JL,NCLDQS) / ZCOVPTOT(JL)
        ZCOVPMAX(JL) = MAX(ZCOVPTOT(JL), ZCOVPMAX(JL))
      ELSE
        ZRAINCLD(JL) = 0.0
        ZSNOWCLD(JL) = 0.0
        ZCOVPTOT(JL) = 0.0
        ZCOVPCLR(JL) = 0.0
        ZCOVPMAX(JL) = 0.0
      ENDIF
    ENDDO

Combines the cumulative IF/ELSE pattern, the kind-mixed
``1.0 - 1.E-06`` constant (REAL(4) lowered to f32 arithmetic then
promoted to f64), and 6 cell writes per branch — all in one
self-contained loop.  The test sweeps multiple JK values inside
a column-by-column driver so ZA(JL, JK-1) reads carry across JKs.

E2e against an f2py-compiled reference of the same Fortran source.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cloudsc_precip_cover(tmp_path):
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE kernel(zcovptot, zcovpclr, zcovpmax, zraincld, zsnowcld, &
                  zqpretot, za, zqxfg, rcovpmin, zepsec, &
                  kidia, kfdia, klon, klev, ncldqr, ncldqs, ncldtop)
integer, intent(in) :: kidia, kfdia, klon, klev, ncldqr, ncldqs, ncldtop
double precision, intent(inout) :: zcovptot(klon), zcovpclr(klon), zcovpmax(klon)
double precision, intent(inout) :: zraincld(klon), zsnowcld(klon)
double precision, intent(in) :: zqpretot(klon)
double precision, intent(in) :: za(klon, klev)
double precision, intent(in) :: zqxfg(klon, 5)
double precision, intent(in) :: rcovpmin, zepsec
integer :: jk, jl
DO jk = ncldtop, klev
    DO jl = kidia, kfdia
        IF (zqpretot(jl) > zepsec) THEN
            zcovptot(jl) = 1.0 - ((1.0 - zcovptot(jl)) * &
             &            (1.0 - MAX(za(jl, jk), za(jl, jk - 1))) / &
             &            (1.0 - MIN(za(jl, jk - 1), 1.0 - 1.E-06)) )
            zcovptot(jl) = MAX(zcovptot(jl), rcovpmin)
            zcovpclr(jl) = MAX(0.0, zcovptot(jl) - za(jl, jk))
            zraincld(jl) = zqxfg(jl, ncldqr) / zcovptot(jl)
            zsnowcld(jl) = zqxfg(jl, ncldqs) / zcovptot(jl)
            zcovpmax(jl) = MAX(zcovptot(jl), zcovpmax(jl))
        ELSE
            zraincld(jl) = 0.0
            zsnowcld(jl) = 0.0
            zcovptot(jl) = 0.0
            zcovpclr(jl) = 0.0
            zcovpmax(jl) = 0.0
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE kernel

SUBROUTINE driver(zcovptot, zcovpclr, zcovpmax, zraincld, zsnowcld, &
                  zqpretot, za, zqxfg, rcovpmin, zepsec, &
                  klon, klev, nblocks, ncldqr, ncldqs, ncldtop)
integer, intent(in) :: klon, klev, nblocks, ncldqr, ncldqs, ncldtop
double precision, intent(inout) :: zcovptot(klon, nblocks), zcovpclr(klon, nblocks)
double precision, intent(inout) :: zcovpmax(klon, nblocks), zraincld(klon, nblocks), zsnowcld(klon, nblocks)
double precision, intent(in) :: zqpretot(klon, nblocks)
double precision, intent(in) :: za(klon, klev, nblocks)
double precision, intent(in) :: zqxfg(klon, 5, nblocks)
double precision, intent(in) :: rcovpmin, zepsec
integer :: ibl
DO ibl = 1, nblocks
    CALL kernel(zcovptot(:, ibl), zcovpclr(:, ibl), zcovpmax(:, ibl), &
                zraincld(:, ibl), zsnowcld(:, ibl), &
                zqpretot(:, ibl), za(:, :, ibl), zqxfg(:, :, ibl), &
                rcovpmin, zepsec, 1, klon, klon, klev, ncldqr, ncldqs, ncldtop)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'cloudsc_pc_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='cloudsc_pc', entry='_QMkernel_modPdriver').build()

    klon, klev, nblocks = 1, 137, 4
    ncldqr, ncldqs, ncldtop = 3, 4, 15
    rng = np.random.default_rng(42)
    zcovptot_in = np.asfortranarray(rng.random((klon, nblocks)))
    zcovpclr_in = np.asfortranarray(rng.random((klon, nblocks)))
    zcovpmax_in = np.asfortranarray(rng.random((klon, nblocks)))
    zraincld_in = np.asfortranarray(rng.random((klon, nblocks)))
    zsnowcld_in = np.asfortranarray(rng.random((klon, nblocks)))
    zqpretot = np.asfortranarray(rng.random((klon, nblocks)))
    za = np.asfortranarray(rng.random((klon, klev, nblocks)))
    zqxfg = np.asfortranarray(rng.random((klon, 5, nblocks)))
    rcovpmin = float(rng.random())
    zepsec = 1.0e-12

    def _run_ref():
        return [np.asfortranarray(x.copy()) for x in (zcovptot_in, zcovpclr_in, zcovpmax_in, zraincld_in, zsnowcld_in)]

    zcovptot_r, zcovpclr_r, zcovpmax_r, zraincld_r, zsnowcld_r = _run_ref()
    ref.kernel_mod.driver(zcovptot=zcovptot_r,
                          zcovpclr=zcovpclr_r,
                          zcovpmax=zcovpmax_r,
                          zraincld=zraincld_r,
                          zsnowcld=zsnowcld_r,
                          zqpretot=zqpretot,
                          za=za,
                          zqxfg=zqxfg,
                          rcovpmin=rcovpmin,
                          zepsec=zepsec,
                          ncldqr=ncldqr,
                          ncldqs=ncldqs,
                          ncldtop=ncldtop)

    zcovptot, zcovpclr, zcovpmax, zraincld, zsnowcld = _run_ref()
    from dace.data import Scalar

    def _route(name, val, dtype):
        return val if isinstance(sdfg.arglist().get(name), Scalar) else np.array([val], dtype=dtype)

    sdfg(zcovptot=zcovptot,
         zcovpclr=zcovpclr,
         zcovpmax=zcovpmax,
         zraincld=zraincld,
         zsnowcld=zsnowcld,
         zqpretot=zqpretot,
         za=za,
         zqxfg=zqxfg,
         rcovpmin=_route('rcovpmin', rcovpmin, np.float64),
         zepsec=_route('zepsec', zepsec, np.float64),
         klon=klon,
         klev=klev,
         nblocks=nblocks,
         ncldqr=ncldqr,
         ncldqs=ncldqs,
         ncldtop=ncldtop)

    np.testing.assert_allclose(zcovptot, zcovptot_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zcovpclr, zcovpclr_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zcovpmax, zcovpmax_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zraincld, zraincld_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zsnowcld, zsnowcld_r, rtol=1e-12, atol=1e-12)
