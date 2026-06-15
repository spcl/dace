# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""T2: CLOUDSC loopnest shapes ported from the dace-fortran corpus.

Faithful Python ``@dace.program`` ports of the characteristic loopnest
shapes in ``dace-fortran/tests/cloudsc_full/cloudsc_bottom_lower.F90``
(the CLOUDSC microphysics kernel) — distinct from the
``cloudsc_snippet_{one..four}`` fixtures in ``test_cloudsc.py``:

- the 2-D affine multi-output initialisation
  (``ZTP1 = PT + PTSPHY*PTEND_T`` …),
- the 3-D species x level x column initialisation
  (``ZQX(JL,JK,JM) = PCLV + PTSPHY*PTEND_CLD``),
- the branchy read-modify-write "tidy up very small cloud water" nest
  (a guarded multi-array accumulation — the CLOUDSC-characteristic
  conditional pattern).

The physics constants are baked in as literals (the contract is e2e
numerical equivalence of the vectorised SDFG vs the non-vectorised
reference via ``run_vectorization_test`` — exact constant values are
irrelevant, only consistency). The innermost loop is over the
contiguous column dimension (the vectorisation axis), mirroring the
Fortran ``JL=KIDIA,KFDIA`` inner loop.
"""

import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import numpy
import pytest
import dace

from tests.passes.vectorization.helpers.harness import run_vectorization_test

KLEV = dace.symbol("KLEV")
KLON = dace.symbol("KLON")
NCLV = dace.symbol("NCLV")

_PTSPHY = 50.0
_RLMIN = 1.0e-8
_RAMIN = 1.0e-8
_RALVDCP = 2.5008e6 / 1004.7
_RALSDCP = 2.8345e6 / 1004.7
_ZQTMST = 1.0 / _PTSPHY


@dace.program
def cloudsc_init_affine(pt: dace.float64[KLEV, KLON], pa: dace.float64[KLEV, KLON], ptend_t: dace.float64[KLEV, KLON],
                        ptend_a: dace.float64[KLEV, KLON], ztp1: dace.float64[KLEV, KLON], za: dace.float64[KLEV,
                                                                                                            KLON]):
    # cloudsc_bottom_lower.F90: "non CLV initialization" nest.
    for jk in range(KLEV):
        for jl in range(KLON):
            ztp1[jk, jl] = pt[jk, jl] + _PTSPHY * ptend_t[jk, jl]
            za[jk, jl] = pa[jk, jl] + _PTSPHY * ptend_a[jk, jl]


def test_cloudsc_init_affine(remainder_strategy, branch_mode):
    klev, klon = 16, 64
    pt = numpy.random.rand(klev, klon)
    pa = numpy.random.rand(klev, klon)
    ptend_t = numpy.random.rand(klev, klon)
    ptend_a = numpy.random.rand(klev, klon)
    ztp1 = numpy.zeros((klev, klon))
    za = numpy.zeros((klev, klon))
    run_vectorization_test(
        dace_func=cloudsc_init_affine,
        arrays={
            "pt": pt,
            "pa": pa,
            "ptend_t": ptend_t,
            "ptend_a": ptend_a,
            "ztp1": ztp1,
            "za": za
        },
        params={
            "KLEV": klev,
            "KLON": klon
        },
        sdfg_name="cloudsc_init_affine",
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
    )


@dace.program
def cloudsc_species_init(pclv: dace.float64[NCLV, KLEV, KLON], ptend_cld: dace.float64[NCLV, KLEV, KLON],
                         zqx: dace.float64[NCLV, KLEV, KLON]):
    # cloudsc_bottom_lower.F90: "initialization for CLV family" 3-D nest.
    for jm in range(NCLV):
        for jk in range(KLEV):
            for jl in range(KLON):
                zqx[jm, jk, jl] = pclv[jm, jk, jl] + _PTSPHY * ptend_cld[jm, jk, jl]


def test_cloudsc_species_init(remainder_strategy, branch_mode):
    nclv, klev, klon = 5, 16, 64
    pclv = numpy.random.rand(nclv, klev, klon)
    ptend_cld = numpy.random.rand(nclv, klev, klon)
    zqx = numpy.zeros((nclv, klev, klon))
    run_vectorization_test(
        dace_func=cloudsc_species_init,
        arrays={
            "pclv": pclv,
            "ptend_cld": ptend_cld,
            "zqx": zqx
        },
        params={
            "NCLV": nclv,
            "KLEV": klev,
            "KLON": klon
        },
        sdfg_name="cloudsc_species_init",
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
    )


@dace.program
def cloudsc_tidy_branch(zqx_l: dace.float64[KLEV, KLON], zqx_i: dace.float64[KLEV, KLON],
                        zqx_v: dace.float64[KLEV, KLON], za: dace.float64[KLEV, KLON],
                        ptend_q: dace.float64[KLEV, KLON], ptend_t: dace.float64[KLEV, KLON]):
    # cloudsc_bottom_lower.F90: "Tidy up very small cloud cover or total
    # cloud water" — guarded read-modify-write over several arrays (the
    # CLOUDSC-characteristic conditional accumulation pattern).
    for jk in range(KLEV):
        for jl in range(KLON):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < _RLMIN or za[jk, jl] < _RAMIN:
                zqadj_l = zqx_l[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALVDCP * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0
                zqadj_i = zqx_i[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_i
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALSDCP * zqadj_i
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_i[jk, jl]
                zqx_i[jk, jl] = 0.0
                za[jk, jl] = 0.0


def test_cloudsc_tidy_branch(remainder_strategy, branch_mode, request):
    # BUG #1 FIXED: BranchNormalization._normalize_single_arm now lifts
    # a linear chain of substantive states (the frontend serialises this
    # ~9-statement multi-array RMW arm into 3 states), applying the
    # per-state merge rewrite to each — value-preserving for chained
    # RMW. The former strict-xfail tripwire on branch_mode=="merge" is
    # removed; both lowering paths must now pass.
    klev, klon = 16, 64
    # Mix of tiny (trigger the guard) and normal magnitudes.
    zqx_l = numpy.where(
        numpy.random.rand(klev, klon) < 0.5,
        numpy.random.rand(klev, klon) * 1e-12, numpy.random.rand(klev, klon))
    zqx_i = numpy.random.rand(klev, klon) * 1e-12
    zqx_v = numpy.random.rand(klev, klon)
    za = numpy.where(
        numpy.random.rand(klev, klon) < 0.5,
        numpy.random.rand(klev, klon) * 1e-12, numpy.random.rand(klev, klon))
    ptend_q = numpy.random.rand(klev, klon)
    ptend_t = numpy.random.rand(klev, klon)
    run_vectorization_test(
        dace_func=cloudsc_tidy_branch,
        arrays={
            "zqx_l": zqx_l,
            "zqx_i": zqx_i,
            "zqx_v": zqx_v,
            "za": za,
            "ptend_q": ptend_q,
            "ptend_t": ptend_t
        },
        params={
            "KLEV": klev,
            "KLON": klon
        },
        sdfg_name="cloudsc_tidy_branch",
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
