"""Cumulative flux accumulation loop from CLOUDSC.

Lifts the flux-output block (cloudscexp2_simplified.F90 lines 3467-3526)
that accumulates ``PFSQLF/PFSQIF/PFSQRF/PFSQSF`` etc. as running sums
over the JK axis.

The pattern:

    PFSQLF(JL, 1) = 0.0
    DO JK = 1, KLEV
      PFSQLF(JL, JK+1) = PFSQLF(JL, JK)
      PFSQLF(JL, JK+1) = PFSQLF(JL, JK+1) + per_step_jk

where ``per_step_jk`` is a fused expression over the JK-level inputs
(``ZQXN2D``, ``ZQX0``, ``PVFL``, ``PLUDE``, ``ZFOEALFA``).  The
``PFSQLF(JK+1) = PFSQLF(JK)`` assignment is a RAW carry across JK
iterations; the bridge has to sequence this in the right state order
or else the same iteration's update overwrites a stale carry.

E2e against an f2py-compiled reference of the same Fortran source.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cloudsc_flux_accumulation(tmp_path):
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE kernel(pfsqlf, pfsqif, pfsqrf, pfsqsf, zqxn2d, zqx0, pvfl, pvfi, plude, zfoealfa, paph, &
                  ptsphy, zqtmst, zrg_r, kidia, kfdia, klon, klev, nclv, ncldql, ncldqi, ncldqr, ncldqs)
integer, intent(in) :: kidia, kfdia, klon, klev, nclv, ncldql, ncldqi, ncldqr, ncldqs
double precision, intent(inout) :: pfsqlf(klon, klev + 1)
double precision, intent(inout) :: pfsqif(klon, klev + 1)
double precision, intent(inout) :: pfsqrf(klon, klev + 1)
double precision, intent(inout) :: pfsqsf(klon, klev + 1)
double precision, intent(in) :: zqxn2d(klon, klev, nclv)
double precision, intent(in) :: zqx0(klon, klev, nclv)
double precision, intent(in) :: pvfl(klon, klev), pvfi(klon, klev)
double precision, intent(in) :: plude(klon, klev)
double precision, intent(in) :: zfoealfa(klon, klev)
double precision, intent(in) :: paph(klon, klev + 1)
double precision, intent(in) :: ptsphy, zqtmst, zrg_r
integer :: jk, jl
double precision :: zalfaw, zgdph_r

DO jl = kidia, kfdia
    pfsqlf(jl, 1) = 0.0
    pfsqif(jl, 1) = 0.0
    pfsqrf(jl, 1) = 0.0
    pfsqsf(jl, 1) = 0.0
ENDDO

DO jk = 1, klev
    DO jl = kidia, kfdia
        zgdph_r = - zrg_r * (paph(jl, jk + 1) - paph(jl, jk)) * zqtmst
        pfsqlf(jl, jk + 1) = pfsqlf(jl, jk)
        pfsqif(jl, jk + 1) = pfsqif(jl, jk)
        pfsqrf(jl, jk + 1) = pfsqlf(jl, jk)
        pfsqsf(jl, jk + 1) = pfsqif(jl, jk)

        zalfaw = zfoealfa(jl, jk)

        pfsqlf(jl, jk + 1) = pfsqlf(jl, jk + 1) + &
         & (zqxn2d(jl, jk, ncldql) - zqx0(jl, jk, ncldql) + pvfl(jl, jk) * ptsphy - zalfaw * plude(jl, jk)) * zgdph_r
        pfsqrf(jl, jk + 1) = pfsqrf(jl, jk + 1) + (zqxn2d(jl, jk, ncldqr) - zqx0(jl, jk, ncldqr)) * zgdph_r
        pfsqif(jl, jk + 1) = pfsqif(jl, jk + 1) + &
         & (zqxn2d(jl, jk, ncldqi) - zqx0(jl, jk, ncldqi) + pvfi(jl, jk) * ptsphy - (1.0 - zalfaw) * plude(jl, jk)) * zgdph_r
        pfsqsf(jl, jk + 1) = pfsqsf(jl, jk + 1) + (zqxn2d(jl, jk, ncldqs) - zqx0(jl, jk, ncldqs)) * zgdph_r
    ENDDO
ENDDO
END SUBROUTINE kernel

SUBROUTINE driver(pfsqlf, pfsqif, pfsqrf, pfsqsf, zqxn2d, zqx0, pvfl, pvfi, plude, zfoealfa, paph, &
                  ptsphy, zqtmst, zrg_r, klon, klev, nclv, nblocks, ncldql, ncldqi, ncldqr, ncldqs)
integer, intent(in) :: klon, klev, nclv, nblocks, ncldql, ncldqi, ncldqr, ncldqs
double precision, intent(inout) :: pfsqlf(klon, klev + 1, nblocks)
double precision, intent(inout) :: pfsqif(klon, klev + 1, nblocks)
double precision, intent(inout) :: pfsqrf(klon, klev + 1, nblocks)
double precision, intent(inout) :: pfsqsf(klon, klev + 1, nblocks)
double precision, intent(in) :: zqxn2d(klon, klev, nclv, nblocks)
double precision, intent(in) :: zqx0(klon, klev, nclv, nblocks)
double precision, intent(in) :: pvfl(klon, klev, nblocks), pvfi(klon, klev, nblocks)
double precision, intent(in) :: plude(klon, klev, nblocks)
double precision, intent(in) :: zfoealfa(klon, klev, nblocks)
double precision, intent(in) :: paph(klon, klev + 1, nblocks)
double precision, intent(in) :: ptsphy, zqtmst, zrg_r
integer :: ibl
DO ibl = 1, nblocks
    CALL kernel(pfsqlf(:, :, ibl), pfsqif(:, :, ibl), pfsqrf(:, :, ibl), pfsqsf(:, :, ibl), &
                zqxn2d(:, :, :, ibl), zqx0(:, :, :, ibl), pvfl(:, :, ibl), pvfi(:, :, ibl), &
                plude(:, :, ibl), zfoealfa(:, :, ibl), paph(:, :, ibl), &
                ptsphy, zqtmst, zrg_r, 1, klon, klon, klev, nclv, ncldql, ncldqi, ncldqr, ncldqs)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'cloudsc_flux_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='cloudsc_flux', entry='_QMkernel_modPdriver').build()

    klon, klev, nclv, nblocks = 1, 137, 5, 4
    ncldql, ncldqi, ncldqr, ncldqs = 1, 2, 3, 4
    rng = np.random.default_rng(42)
    zqxn2d = np.asfortranarray(rng.random((klon, klev, nclv, nblocks)))
    zqx0 = np.asfortranarray(rng.random((klon, klev, nclv, nblocks)))
    pvfl = np.asfortranarray(rng.random((klon, klev, nblocks)))
    pvfi = np.asfortranarray(rng.random((klon, klev, nblocks)))
    plude = np.asfortranarray(rng.random((klon, klev, nblocks)))
    zfoealfa = np.asfortranarray(rng.random((klon, klev, nblocks)))
    paph = np.asfortranarray(rng.random((klon, klev + 1, nblocks)))
    ptsphy = rng.random()
    zqtmst = rng.random()
    zrg_r = rng.random()

    pfsqlf_ref = np.asfortranarray(rng.random((klon, klev + 1, nblocks)))
    pfsqif_ref = np.asfortranarray(rng.random((klon, klev + 1, nblocks)))
    pfsqrf_ref = np.asfortranarray(rng.random((klon, klev + 1, nblocks)))
    pfsqsf_ref = np.asfortranarray(rng.random((klon, klev + 1, nblocks)))
    pfsqlf = np.asfortranarray(pfsqlf_ref.copy())
    pfsqif = np.asfortranarray(pfsqif_ref.copy())
    pfsqrf = np.asfortranarray(pfsqrf_ref.copy())
    pfsqsf = np.asfortranarray(pfsqsf_ref.copy())

    ref.kernel_mod.driver(pfsqlf=pfsqlf_ref,
                          pfsqif=pfsqif_ref,
                          pfsqrf=pfsqrf_ref,
                          pfsqsf=pfsqsf_ref,
                          zqxn2d=zqxn2d,
                          zqx0=zqx0,
                          pvfl=pvfl,
                          pvfi=pvfi,
                          plude=plude,
                          zfoealfa=zfoealfa,
                          paph=paph,
                          ptsphy=ptsphy,
                          zqtmst=zqtmst,
                          zrg_r=zrg_r,
                          ncldql=ncldql,
                          ncldqi=ncldqi,
                          ncldqr=ncldqr,
                          ncldqs=ncldqs)

    from dace.data import Scalar

    def _route(name, val, dtype):
        return val if isinstance(sdfg.arglist().get(name), Scalar) else np.array([val], dtype=dtype)

    sdfg(pfsqlf=pfsqlf,
         pfsqif=pfsqif,
         pfsqrf=pfsqrf,
         pfsqsf=pfsqsf,
         zqxn2d=zqxn2d,
         zqx0=zqx0,
         pvfl=pvfl,
         pvfi=pvfi,
         plude=plude,
         zfoealfa=zfoealfa,
         paph=paph,
         ptsphy=_route('ptsphy', ptsphy, np.float64),
         zqtmst=_route('zqtmst', zqtmst, np.float64),
         zrg_r=_route('zrg_r', zrg_r, np.float64),
         klon=klon,
         klev=klev,
         nclv=nclv,
         nblocks=nblocks,
         ncldql=ncldql,
         ncldqi=ncldqi,
         ncldqr=ncldqr,
         ncldqs=ncldqs)

    np.testing.assert_allclose(pfsqlf, pfsqlf_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pfsqif, pfsqif_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pfsqrf, pfsqrf_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pfsqsf, pfsqsf_ref, rtol=1e-12, atol=1e-12)
