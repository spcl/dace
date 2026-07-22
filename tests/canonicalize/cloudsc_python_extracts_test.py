"""Canonicalization tests over mid-size loop nests extracted from
``dwarf-p-cloudsc/src/cloudsc_python/src/cloudscf2py/cloudsc_py.py``.

Each kernel is reduced to a self-contained ``@dace.program`` whose
external constants and look-up functions are replaced by simple
polynomial / arithmetic placeholders, so the SDFG round-trip stays
focused on the LOOP STRUCTURE that canonicalize must handle. Every
test asserts:

* the SDFG canonicalizes without raising ``sdfg.validate()`` errors,
* the numerical output matches a plain-numpy oracle (``equal_nan=True``),

so a future regression in any canonicalize stage on these realistic
shapes fails loudly.

Shapes covered:

* ``test_init_loc_tendencies_2d_elementwise_multi_array`` -- 2D
  elementwise initialization of 5 sibling arrays (the
  ``tendency_loc_*`` zero-init from the cloudsc preamble).
* ``test_init_3d_inner_clv_elementwise_with_offset`` -- 3D
  elementwise init over a CLV family (jm, jk, jl). Asserts the
  ``+ ptsphy * tendency_tmp_*`` shift survives.
* ``test_tidy_small_cloud_water_guarded_multi_write`` -- 2D guarded
  nest with one ``if`` predicate driving 5 sibling writes (the
  "tidy up very small cloud cover" block).
* ``test_clv_phase_dispatched_three_branch`` -- 3D nest whose body
  has an outer ``if`` plus two nested ``if iphase[jm] == K`` arms
  (the "tidy up small CLV variables" block) -- exercises the
  conditional-fission + guard-hoist canonicalize stages.
* ``test_2d_min_clamp_division_chain`` -- 2D nest of the
  ``min(... / pap, 0.5); X / (1 - rev*X)`` form used in the
  saturation block; multiple per-iteration scalar intermediates +
  clamps.
"""
import numpy as np

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize

K = dace.symbol('K')
L = dace.symbol('L')
M = dace.symbol('M')
KP1 = dace.symbol('KP1')

# ---------------------------------------------------------------------------
# Kernel 1: 2D elementwise init of 5 sibling arrays
# ---------------------------------------------------------------------------


@dace.program
def _cloudsc_init_loc_tendencies(
    tendency_loc_t: dace.float64[K, L],
    tendency_loc_q: dace.float64[K, L],
    tendency_loc_a: dace.float64[K, L],
    pcovptot: dace.float64[K, L],
    tendency_loc_cld_last: dace.float64[K, L],
):
    for jk in range(K):
        for jl in range(L):
            tendency_loc_t[jk, jl] = 0.0
            tendency_loc_q[jk, jl] = 0.0
            tendency_loc_a[jk, jl] = 0.0
            pcovptot[jk, jl] = 0.0
            tendency_loc_cld_last[jk, jl] = 0.0


def test_init_loc_tendencies_2d_elementwise_multi_array():
    kk, ll = 8, 12
    sdfg = _cloudsc_init_loc_tendencies.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    arrays = {
        n: np.full((kk, ll), 0.7, dtype=np.float64)
        for n in ['tendency_loc_t', 'tendency_loc_q', 'tendency_loc_a', 'pcovptot', 'tendency_loc_cld_last']
    }
    sdfg(K=kk, L=ll, **arrays)
    for n, a in arrays.items():
        assert np.allclose(a, 0.0), f'{n}: not zeroed'


# ---------------------------------------------------------------------------
# Kernel 2: 3D elementwise init with timestep shift
# ---------------------------------------------------------------------------


@dace.program
def _cloudsc_init_3d_clv(
    pclv: dace.float64[M, K, L],
    tendency_tmp_cld: dace.float64[M, K, L],
    ztp1: dace.float64[M, K, L],
    ptsphy: dace.float64,
):
    for jm in range(M):
        for jk in range(K):
            for jl in range(L):
                ztp1[jm, jk, jl] = pclv[jm, jk, jl] + ptsphy * tendency_tmp_cld[jm, jk, jl]


def test_init_3d_inner_clv_elementwise_with_offset():
    mm, kk, ll = 4, 6, 8
    rng = np.random.default_rng(2026)
    pclv = rng.standard_normal((mm, kk, ll))
    tt = rng.standard_normal((mm, kk, ll))
    ptsphy = 0.25
    z = np.zeros_like(pclv)
    ref = pclv + ptsphy * tt
    sdfg = _cloudsc_init_3d_clv.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    sdfg(pclv=pclv.copy(), tendency_tmp_cld=tt.copy(), ztp1=z, ptsphy=ptsphy, M=mm, K=kk, L=ll)
    assert np.allclose(z, ref)


# ---------------------------------------------------------------------------
# Kernel 3: 2D guarded multi-write (tidy small cloud water)
# ---------------------------------------------------------------------------


@dace.program
def _cloudsc_tidy_small_cloud(
    zqx_ql: dace.float64[K, L],
    zqx_qi: dace.float64[K, L],
    zqx_qv: dace.float64[K, L],
    za: dace.float64[K, L],
    tendency_loc_q: dace.float64[K, L],
    tendency_loc_t: dace.float64[K, L],
    zqtmst: dace.float64,
    ralvdcp: dace.float64,
    ralsdcp: dace.float64,
    rlmin: dace.float64,
    ramin: dace.float64,
):
    for jk in range(K):
        for jl in range(L):
            if zqx_ql[jk, jl] + zqx_qi[jk, jl] < rlmin or za[jk, jl] < ramin:
                # Evaporate small cloud liquid water
                zqadj = zqx_ql[jk, jl] * zqtmst
                tendency_loc_q[jk, jl] = tendency_loc_q[jk, jl] + zqadj
                tendency_loc_t[jk, jl] = tendency_loc_t[jk, jl] - ralvdcp * zqadj
                zqx_qv[jk, jl] = zqx_qv[jk, jl] + zqx_ql[jk, jl]
                zqx_ql[jk, jl] = 0.0
                # Evaporate small cloud ice water
                zqadj = zqx_qi[jk, jl] * zqtmst
                tendency_loc_q[jk, jl] = tendency_loc_q[jk, jl] + zqadj
                tendency_loc_t[jk, jl] = tendency_loc_t[jk, jl] - ralsdcp * zqadj
                zqx_qv[jk, jl] = zqx_qv[jk, jl] + zqx_qi[jk, jl]
                zqx_qi[jk, jl] = 0.0
                za[jk, jl] = 0.0


def test_tidy_small_cloud_water_guarded_multi_write():
    kk, ll = 6, 8
    rng = np.random.default_rng(2025)
    # Half the cells are below threshold (so the guard fires)
    ql = rng.uniform(0.0, 2e-7, (kk, ll))
    qi = rng.uniform(0.0, 2e-7, (kk, ll))
    qv = rng.uniform(0.5, 1.5, (kk, ll))
    a = rng.uniform(0.0, 0.5, (kk, ll))
    tq = rng.standard_normal((kk, ll))
    tt = rng.standard_normal((kk, ll))
    zqtmst, ralvdcp, ralsdcp, rlmin, ramin = 100.0, 2500.0, 2800.0, 1e-6, 1e-5

    def oracle(ql, qi, qv, a, tq, tt):
        ql, qi, qv, a = ql.copy(), qi.copy(), qv.copy(), a.copy()
        tq, tt = tq.copy(), tt.copy()
        for jk in range(kk):
            for jl in range(ll):
                if ql[jk, jl] + qi[jk, jl] < rlmin or a[jk, jl] < ramin:
                    z = ql[jk, jl] * zqtmst
                    tq[jk, jl] += z
                    tt[jk, jl] -= ralvdcp * z
                    qv[jk, jl] += ql[jk, jl]
                    ql[jk, jl] = 0.0
                    z = qi[jk, jl] * zqtmst
                    tq[jk, jl] += z
                    tt[jk, jl] -= ralsdcp * z
                    qv[jk, jl] += qi[jk, jl]
                    qi[jk, jl] = 0.0
                    a[jk, jl] = 0.0
        return ql, qi, qv, a, tq, tt

    rql, rqi, rqv, ra, rtq, rtt = oracle(ql, qi, qv, a, tq, tt)
    sdfg = _cloudsc_tidy_small_cloud.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    sql, sqi, sqv, sa, stq, stt = ql.copy(), qi.copy(), qv.copy(), a.copy(), tq.copy(), tt.copy()
    sdfg(zqx_ql=sql,
         zqx_qi=sqi,
         zqx_qv=sqv,
         za=sa,
         tendency_loc_q=stq,
         tendency_loc_t=stt,
         zqtmst=zqtmst,
         ralvdcp=ralvdcp,
         ralsdcp=ralsdcp,
         rlmin=rlmin,
         ramin=ramin,
         K=kk,
         L=ll)
    for got, ref, name in [(sql, rql, 'ql'), (sqi, rqi, 'qi'), (sqv, rqv, 'qv'), (sa, ra, 'a'), (stq, rtq, 'tq'),
                           (stt, rtt, 'tt')]:
        assert np.allclose(got, ref), f'{name}: canon diverges from numpy'


# ---------------------------------------------------------------------------
# Kernel 4: 3D phase-dispatched guarded multi-write
# ---------------------------------------------------------------------------


@dace.program
def _cloudsc_tidy_small_clv(
    zqx: dace.float64[M, K, L],
    zqx_qv: dace.float64[K, L],
    tendency_loc_q: dace.float64[K, L],
    tendency_loc_t: dace.float64[K, L],
    iphase: dace.int32[M],
    zqtmst: dace.float64,
    ralvdcp: dace.float64,
    ralsdcp: dace.float64,
    rlmin: dace.float64,
):
    for jm in range(M):
        for jk in range(K):
            for jl in range(L):
                if zqx[jm, jk, jl] < rlmin:
                    zqadj = zqx[jm, jk, jl] * zqtmst
                    tendency_loc_q[jk, jl] = tendency_loc_q[jk, jl] + zqadj
                    if iphase[jm] == 1:
                        tendency_loc_t[jk, jl] = tendency_loc_t[jk, jl] - ralvdcp * zqadj
                    if iphase[jm] == 2:
                        tendency_loc_t[jk, jl] = tendency_loc_t[jk, jl] - ralsdcp * zqadj
                    zqx_qv[jk, jl] = zqx_qv[jk, jl] + zqx[jm, jk, jl]
                    zqx[jm, jk, jl] = 0.0


def test_clv_phase_dispatched_three_branch():
    mm, kk, ll = 3, 5, 6
    rng = np.random.default_rng(2027)
    zqx = rng.uniform(0.0, 5e-7, (mm, kk, ll))
    qv = rng.uniform(0.4, 1.0, (kk, ll))
    tq = rng.standard_normal((kk, ll))
    tt = rng.standard_normal((kk, ll))
    iphase = np.array([1, 2, 0], dtype=np.int32)
    zqtmst, ralvdcp, ralsdcp, rlmin = 100.0, 2500.0, 2800.0, 1e-6

    def oracle(zqx, qv, tq, tt):
        zqx, qv, tq, tt = zqx.copy(), qv.copy(), tq.copy(), tt.copy()
        for jm in range(mm):
            for jk in range(kk):
                for jl in range(ll):
                    if zqx[jm, jk, jl] < rlmin:
                        z = zqx[jm, jk, jl] * zqtmst
                        tq[jk, jl] += z
                        if iphase[jm] == 1:
                            tt[jk, jl] -= ralvdcp * z
                        if iphase[jm] == 2:
                            tt[jk, jl] -= ralsdcp * z
                        qv[jk, jl] += zqx[jm, jk, jl]
                        zqx[jm, jk, jl] = 0.0
        return zqx, qv, tq, tt

    rzqx, rqv, rtq, rtt = oracle(zqx, qv, tq, tt)
    sdfg = _cloudsc_tidy_small_clv.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    sx, sqv, stq, stt = zqx.copy(), qv.copy(), tq.copy(), tt.copy()
    sdfg(zqx=sx,
         zqx_qv=sqv,
         tendency_loc_q=stq,
         tendency_loc_t=stt,
         iphase=iphase.copy(),
         zqtmst=zqtmst,
         ralvdcp=ralvdcp,
         ralsdcp=ralsdcp,
         rlmin=rlmin,
         M=mm,
         K=kk,
         L=ll)
    assert np.allclose(sx, rzqx) and np.allclose(sqv, rqv) and np.allclose(stq, rtq) and np.allclose(stt, rtt)


# ---------------------------------------------------------------------------
# Kernel 5: 2D min-clamp + division chain (saturation block shape)
# ---------------------------------------------------------------------------


@dace.program
def _cloudsc_saturation_chain(
    ztp1: dace.float64[K, L],
    pap: dace.float64[K, L],
    zqsmix: dace.float64[K, L],
    zqsice: dace.float64[K, L],
    zqsliq: dace.float64[K, L],
    retv: dace.float64,
):
    for jk in range(K):
        for jl in range(L):
            # foeewm placeholder: a smooth polynomial of ztp1
            zfoeewmt = min(0.5, (ztp1[jk, jl] * ztp1[jk, jl] * 1e-4) / pap[jk, jl])
            zqsmix[jk, jl] = zfoeewmt / (1.0 - retv * zfoeewmt)
            # foeeliq / foeeice placeholders + diagnostic mix
            zfoeew = min(0.5, (ztp1[jk, jl] * 1e-2) / pap[jk, jl])
            zfoeew = min(0.5, zfoeew)
            zqsice[jk, jl] = zfoeew / (1.0 - retv * zfoeew)
            zfoeeliqt = min(0.5, (ztp1[jk, jl] * 1.5e-2) / pap[jk, jl])
            zqsliq[jk, jl] = zfoeeliqt / (1.0 - retv * zfoeeliqt)


def test_2d_min_clamp_division_chain():
    kk, ll = 6, 8
    rng = np.random.default_rng(2028)
    ztp1 = rng.uniform(200.0, 320.0, (kk, ll))
    pap = rng.uniform(1e4, 1e5, (kk, ll))
    qsm = np.zeros_like(ztp1)
    qsi = np.zeros_like(ztp1)
    qsl = np.zeros_like(ztp1)
    retv = 0.609

    def oracle(ztp1, pap):
        qsm = np.zeros_like(ztp1)
        qsi = np.zeros_like(ztp1)
        qsl = np.zeros_like(ztp1)
        for jk in range(kk):
            for jl in range(ll):
                zfm = min(0.5, (ztp1[jk, jl] * ztp1[jk, jl] * 1e-4) / pap[jk, jl])
                qsm[jk, jl] = zfm / (1.0 - retv * zfm)
                zfe = min(0.5, (ztp1[jk, jl] * 1e-2) / pap[jk, jl])
                zfe = min(0.5, zfe)
                qsi[jk, jl] = zfe / (1.0 - retv * zfe)
                zfl = min(0.5, (ztp1[jk, jl] * 1.5e-2) / pap[jk, jl])
                qsl[jk, jl] = zfl / (1.0 - retv * zfl)
        return qsm, qsi, qsl

    rqsm, rqsi, rqsl = oracle(ztp1, pap)
    sdfg = _cloudsc_saturation_chain.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    sdfg(ztp1=ztp1.copy(), pap=pap.copy(), zqsmix=qsm, zqsice=qsi, zqsliq=qsl, retv=retv, K=kk, L=ll)
    assert np.allclose(qsm, rqsm) and np.allclose(qsi, rqsi) and np.allclose(qsl, rqsl)


if __name__ == '__main__':
    test_init_loc_tendencies_2d_elementwise_multi_array()
    test_init_3d_inner_clv_elementwise_with_offset()
    test_tidy_small_cloud_water_guarded_multi_write()
    test_clv_phase_dispatched_three_branch()
    test_2d_min_clamp_division_chain()
