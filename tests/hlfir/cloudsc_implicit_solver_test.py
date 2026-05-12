"""Implicit microphysics solver pipeline from CLOUDSC.

Lifts the full implicit-solver block (cloudscexp2_simplified.F90
lines 3257-3380) that combines five back-to-back loopnests:

1. ZQLHS construction (lines 3257-3278) — diagonals as
   ``1 + ZFALLSINK(JM) + sum_O ZSOLQB(JO,JN)`` and off-diagonals as
   ``-ZSOLQB(JN,JM)``.  Two IF branches in the JN/JM nest.
2. ZQXN RHS initialization (lines 3283-3294) — ``ZQXN(JM) = ZQX(JK,JM)
   + sum_N ZSOLQA(JM,N)``.
3. LU forward+back substitution (lines 3308-3336) — already covered
   in ``cloudsc_lu_solver_test.py`` but exercised here in the
   full-pipeline context.
4. Sub-epsilon clamp (lines 3341-3348) — ``IF ZQXN(JN) < ZEPSEC THEN
   ZQXN(NCLDQV) += ZQXN(JN); ZQXN(JN) = 0``.  IF-mutation of a shared
   reduction target.
5. ZQXN -> ZQXN2D + ZQXNM1 carry (lines 3353-3358) — column-wise
   write-back and cross-JK carry.
6. Precip-flux derivation + ZCOVPTOT reset (lines 3366-3380) — feeds
   PCOVPTOT's pattern A.

This is the chunk where the cloudsc_full integration test's per-step
flux errors (Fortran JK=22, 34, 39, 48) originate.  Running all six
loopnests in sequence over a JK loop catches cross-loopnest state
interactions that the per-piece tests can't see.

E2e against an f2py-compiled reference of the same Fortran source.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cloudsc_implicit_solver(tmp_path):
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE kernel(zqxn2d, zqxnm1, zpfplsx, zcovptot, zqpretot, &
                  zqx, zsolqa, zsolqb, zfallsink, zrdtgdp, zepsec, &
                  kidia, kfdia, klon, klev, nclv, ncldqv, ncldqr, ncldqs, ncldtop)
integer, intent(in) :: kidia, kfdia, klon, klev, nclv, ncldqv, ncldqr, ncldqs, ncldtop
double precision, intent(inout) :: zqxn2d(klon, klev, nclv)
double precision, intent(inout) :: zqxnm1(klon, nclv)
double precision, intent(inout) :: zpfplsx(klon, klev + 1, nclv)
double precision, intent(inout) :: zcovptot(klon)
double precision, intent(inout) :: zqpretot(klon)
double precision, intent(in) :: zqx(klon, klev, nclv)
double precision, intent(in) :: zsolqa(klon, klev, nclv, nclv)
double precision, intent(in) :: zsolqb(klon, klev, nclv, nclv)
double precision, intent(in) :: zfallsink(klon, klev, nclv)
double precision, intent(in) :: zrdtgdp(klon, klev)
double precision, intent(in) :: zepsec
integer :: jk, jl, jm, jn, ik, jo
double precision :: zqlhs(klon, nclv, nclv)
double precision :: zqxn(klon, nclv)
double precision :: zexplicit

DO jk = ncldtop, klev
    ! 1. ZQLHS construction
    DO jm = 1, nclv
        DO jn = 1, nclv
            IF (jn == jm) THEN
                DO jl = kidia, kfdia
                    zqlhs(jl, jn, jm) = 1.0 + zfallsink(jl, jk, jm)
                    DO jo = 1, nclv
                        zqlhs(jl, jn, jm) = zqlhs(jl, jn, jm) + zsolqb(jl, jk, jo, jn)
                    ENDDO
                ENDDO
            ELSE
                DO jl = kidia, kfdia
                    zqlhs(jl, jn, jm) = -zsolqb(jl, jk, jn, jm)
                ENDDO
            ENDIF
        ENDDO
    ENDDO

    ! 2. ZQXN RHS initialization
    DO jm = 1, nclv
        DO jl = kidia, kfdia
            zexplicit = 0.0
            DO jn = 1, nclv
                zexplicit = zexplicit + zsolqa(jl, jk, jm, jn)
            ENDDO
            zqxn(jl, jm) = zqx(jl, jk, jm) + zexplicit
        ENDDO
    ENDDO

    ! 3. LU forward sweep (factorization)
    DO jn = 1, nclv - 1
        DO jm = jn + 1, nclv
            zqlhs(kidia:kfdia, jm, jn) = zqlhs(kidia:kfdia, jm, jn) / zqlhs(kidia:kfdia, jn, jn)
            DO ik = jn + 1, nclv
                DO jl = kidia, kfdia
                    zqlhs(jl, jm, ik) = zqlhs(jl, jm, ik) - zqlhs(jl, jm, jn) * zqlhs(jl, jn, ik)
                ENDDO
            ENDDO
        ENDDO
    ENDDO

    ! LU forward sweep on rhs
    DO jn = 2, nclv
        DO jm = 1, jn - 1
            zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) - zqlhs(kidia:kfdia, jn, jm) * zqxn(kidia:kfdia, jm)
        ENDDO
    ENDDO

    ! LU back sweep
    zqxn(kidia:kfdia, nclv) = zqxn(kidia:kfdia, nclv) / zqlhs(kidia:kfdia, nclv, nclv)
    DO jn = nclv - 1, 1, -1
        DO jm = jn + 1, nclv
            zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) - zqlhs(kidia:kfdia, jn, jm) * zqxn(kidia:kfdia, jm)
        ENDDO
        zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) / zqlhs(kidia:kfdia, jn, jn)
    ENDDO

    ! 4. Sub-epsilon clamp (transfer negative residual to vapor)
    DO jn = 1, nclv - 1
        DO jl = kidia, kfdia
            IF (zqxn(jl, jn) < zepsec) THEN
                zqxn(jl, ncldqv) = zqxn(jl, ncldqv) + zqxn(jl, jn)
                zqxn(jl, jn) = 0.0
            ENDIF
        ENDDO
    ENDDO

    ! 5. Write-back to ZQXN2D + ZQXNM1 cross-JK carry
    DO jm = 1, nclv
        DO jl = kidia, kfdia
            zqxnm1(jl, jm) = zqxn(jl, jm)
            zqxn2d(jl, jk, jm) = zqxn(jl, jm)
        ENDDO
    ENDDO

    ! 6. Precip flux + ZCOVPTOT zero-precip reset
    DO jm = 1, nclv
        DO jl = kidia, kfdia
            zpfplsx(jl, jk + 1, jm) = zfallsink(jl, jk, jm) * zqxn(jl, jm) * zrdtgdp(jl, jk)
        ENDDO
    ENDDO
    DO jl = kidia, kfdia
        zqpretot(jl) = zpfplsx(jl, jk + 1, ncldqs) + zpfplsx(jl, jk + 1, ncldqr)
    ENDDO
    DO jl = kidia, kfdia
        IF (zqpretot(jl) < zepsec) THEN
            zcovptot(jl) = 0.0
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE kernel

SUBROUTINE driver(zqxn2d, zqxnm1, zpfplsx, zcovptot, zqpretot, &
                  zqx, zsolqa, zsolqb, zfallsink, zrdtgdp, zepsec, &
                  klon, klev, nclv, nblocks, ncldqv, ncldqr, ncldqs, ncldtop)
integer, intent(in) :: klon, klev, nclv, nblocks, ncldqv, ncldqr, ncldqs, ncldtop
double precision, intent(inout) :: zqxn2d(klon, klev, nclv, nblocks)
double precision, intent(inout) :: zqxnm1(klon, nclv, nblocks)
double precision, intent(inout) :: zpfplsx(klon, klev + 1, nclv, nblocks)
double precision, intent(inout) :: zcovptot(klon, nblocks)
double precision, intent(inout) :: zqpretot(klon, nblocks)
double precision, intent(in) :: zqx(klon, klev, nclv, nblocks)
double precision, intent(in) :: zsolqa(klon, klev, nclv, nclv, nblocks)
double precision, intent(in) :: zsolqb(klon, klev, nclv, nclv, nblocks)
double precision, intent(in) :: zfallsink(klon, klev, nclv, nblocks)
double precision, intent(in) :: zrdtgdp(klon, klev, nblocks)
double precision, intent(in) :: zepsec
integer :: ibl
DO ibl = 1, nblocks
    CALL kernel(zqxn2d(:, :, :, ibl), zqxnm1(:, :, ibl), zpfplsx(:, :, :, ibl), &
                zcovptot(:, ibl), zqpretot(:, ibl), &
                zqx(:, :, :, ibl), zsolqa(:, :, :, :, ibl), zsolqb(:, :, :, :, ibl), &
                zfallsink(:, :, :, ibl), zrdtgdp(:, :, ibl), zepsec, &
                1, klon, klon, klev, nclv, ncldqv, ncldqr, ncldqs, ncldtop)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'cloudsc_solver_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='cloudsc_solver', entry='_QMkernel_modPdriver').build()

    klon, klev, nclv, nblocks = 1, 137, 5, 4
    ncldqv, ncldqr, ncldqs, ncldtop = 5, 3, 4, 15
    rng = np.random.default_rng(42)

    # Diagonally-dominant ZSOLQB and small ZFALLSINK so ZQLHS stays
    # well-conditioned for the LU solve (real cloudsc has this by
    # construction; the random inputs need explicit shaping).
    zsolqb_in = rng.random((klon, klev, nclv, nclv, nblocks)) * 0.1
    zfallsink = rng.random((klon, klev, nclv, nblocks)) * 0.1
    zsolqa_in = rng.random((klon, klev, nclv, nclv, nblocks)) * 0.1
    zqx = rng.random((klon, klev, nclv, nblocks))
    zrdtgdp = rng.random((klon, klev, nblocks))
    zsolqa = np.asfortranarray(zsolqa_in)
    zsolqb = np.asfortranarray(zsolqb_in)
    zfallsink = np.asfortranarray(zfallsink)
    zqx = np.asfortranarray(zqx)
    zrdtgdp = np.asfortranarray(zrdtgdp)
    zepsec = 1.0e-12

    def _alloc_io():
        return (np.asfortranarray(rng.random(
            (klon, klev, nclv, nblocks))), np.asfortranarray(rng.random(
                (klon, nclv, nblocks))), np.asfortranarray(rng.random(
                    (klon, klev + 1, nclv, nblocks))), np.asfortranarray(rng.random(
                        (klon, nblocks))), np.asfortranarray(rng.random((klon, nblocks))))

    rng_io = np.random.default_rng(99)

    def _rng_io_alloc():
        return (np.asfortranarray(rng_io.random(
            (klon, klev, nclv, nblocks))), np.asfortranarray(rng_io.random(
                (klon, nclv, nblocks))), np.asfortranarray(rng_io.random(
                    (klon, klev + 1, nclv, nblocks))), np.asfortranarray(rng_io.random(
                        (klon, nblocks))), np.asfortranarray(rng_io.random((klon, nblocks))))

    zqxn2d_init, zqxnm1_init, zpfplsx_init, zcovptot_init, zqpretot_init = _rng_io_alloc()

    zqxn2d_r = zqxn2d_init.copy(order='F')
    zqxnm1_r = zqxnm1_init.copy(order='F')
    zpfplsx_r = zpfplsx_init.copy(order='F')
    zcovptot_r = zcovptot_init.copy(order='F')
    zqpretot_r = zqpretot_init.copy(order='F')

    ref.kernel_mod.driver(zqxn2d=zqxn2d_r,
                          zqxnm1=zqxnm1_r,
                          zpfplsx=zpfplsx_r,
                          zcovptot=zcovptot_r,
                          zqpretot=zqpretot_r,
                          zqx=zqx,
                          zsolqa=zsolqa,
                          zsolqb=zsolqb,
                          zfallsink=zfallsink,
                          zrdtgdp=zrdtgdp,
                          zepsec=zepsec,
                          ncldqv=ncldqv,
                          ncldqr=ncldqr,
                          ncldqs=ncldqs,
                          ncldtop=ncldtop)

    zqxn2d = zqxn2d_init.copy(order='F')
    zqxnm1 = zqxnm1_init.copy(order='F')
    zpfplsx = zpfplsx_init.copy(order='F')
    zcovptot = zcovptot_init.copy(order='F')
    zqpretot = zqpretot_init.copy(order='F')

    from dace.data import Scalar

    def _route(name, val, dtype):
        return val if isinstance(sdfg.arglist().get(name), Scalar) else np.array([val], dtype=dtype)

    sdfg(zqxn2d=zqxn2d,
         zqxnm1=zqxnm1,
         zpfplsx=zpfplsx,
         zcovptot=zcovptot,
         zqpretot=zqpretot,
         zqx=zqx,
         zsolqa=zsolqa,
         zsolqb=zsolqb,
         zfallsink=zfallsink,
         zrdtgdp=zrdtgdp,
         zepsec=_route('zepsec', zepsec, np.float64),
         klon=klon,
         klev=klev,
         nclv=nclv,
         nblocks=nblocks,
         ncldqv=ncldqv,
         ncldqr=ncldqr,
         ncldqs=ncldqs,
         ncldtop=ncldtop)

    np.testing.assert_allclose(zqxn2d, zqxn2d_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zqxnm1, zqxnm1_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zpfplsx, zpfplsx_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zcovptot, zcovptot_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zqpretot, zqpretot_r, rtol=1e-12, atol=1e-12)
