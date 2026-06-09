# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""T3: ICON velocity-tendency loopnest shapes ported from dace-fortran.

Faithful Python ``@dace.program`` ports of the two computational
loopnests in ``dace-fortran/tests/velocity_one_loop.f90`` and
``velocity_zekinh_block.f90`` (ICON non-hydrostatic dynamical core):

- ``icon_one_loop``      — the ``one_loop_nest`` 3-D edge x level x
  block stencil: a multi-output update with a vertical (``jk-1``)
  backward reference that lives on the *outer* loop (the inner,
  vectorised, ``je`` axis is plain elementwise).
- ``icon_zekinh_gather`` — the ``zekinh_block`` cell-from-edges
  bilinear interpolation: ``z_ekinh = Σ_m e_bln(m) *
  z_kin_hor_e[edge_idx(m), jk, edge_blk(m)]`` — an indirect
  (index-table) gather over the inner ``jc`` axis (the ICON-
  characteristic indirect-addressing pattern, analogous to spmv).

Layout is C-order ``[nblks, nlev, nproma]`` so the innermost loop runs
over the contiguous ``nproma`` dimension (the vectorisation axis),
mirroring the Fortran ``je``/``jc`` inner loop. Contract: e2e numeric
equivalence of the vectorised SDFG vs the non-vectorised reference via
``run_vectorization_test``.
"""

import pytest
pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import numpy
import pytest
import dace

from tests.passes.vectorization.helpers.harness import run_vectorization_test

NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")


@dace.program
def icon_one_loop(vn: dace.float64[NB, NLEV, NPROMA], wgtfac_e: dace.float64[NB, NLEV, NPROMA],
                  vt: dace.float64[NB, NLEV, NPROMA], vn_ie: dace.float64[NB, NLEV, NPROMA], zkh: dace.float64[NB, NLEV,
                                                                                                               NPROMA]):
    for jb in range(NB):
        for jk in range(1, NLEV):
            for je in range(NPROMA):
                vn_ie[jb, jk, je] = vn[jb, jk, je] - vn[jb, jk - 1, je]
                zkh[jb, jk, je] = vt[jb, jk, je] - wgtfac_e[jb, jk, je]


def test_icon_one_loop(remainder_strategy, branch_mode, emission_style):
    nb, nlev, nproma = 2, 16, 64
    vn = numpy.random.rand(nb, nlev, nproma)
    wgtfac_e = numpy.random.rand(nb, nlev, nproma)
    vt = numpy.random.rand(nb, nlev, nproma)
    vn_ie = numpy.zeros((nb, nlev, nproma))
    zkh = numpy.zeros((nb, nlev, nproma))
    run_vectorization_test(
        dace_func=icon_one_loop,
        arrays={
            "vn": vn,
            "wgtfac_e": wgtfac_e,
            "vt": vt,
            "vn_ie": vn_ie,
            "zkh": zkh
        },
        params={
            "NB": nb,
            "NLEV": nlev,
            "NPROMA": nproma
        },
        sdfg_name="icon_one_loop",
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
        emission_style=emission_style,
    )


@dace.program
def icon_zekinh_gather(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA,
                                                                                3], edge_blk: dace.int32[NB, NPROMA, 3],
                       z_kin_hor_e: dace.float64[NB, NLEV, NPROMA], z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = (e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                                       e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                                       e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])


def test_icon_zekinh_gather(remainder_strategy, branch_mode, emission_style):
    # BUG #2 FIXED (InvalidSDFGEdgeError): _generate_loads_to_packed_storage
    # groups consuming edges by gather subset (dict), one packed buffer
    # + per-lane loads per DISTINCT gather (dedup), rewrites EVERY
    # consuming edge, drops orphan sources; assign_{i} label +
    # <arr>_<n>_packed naming preserved.
    # BUG #2b FIXED (masked-merge): the ICON gather has TWO lane-variant
    # gathered index components (edge_blk dim-0 + edge_idx dim-2, both
    # indexed by jc) — a multi-variable gather with no hardware
    # intrinsic. GenerateIterationMask._lane_variant_indirect_dim_count
    # detects >=2 such components and auto-degrades the masked remainder
    # to a scalar tail (universally safe). A single-index gather
    # (count<=1, e.g. spmv x[idx[i]]) keeps the R2 masked-gather
    # intrinsic path — non-regressing.
    nb, nlev, nproma = 2, 16, 64
    e_bln = numpy.random.rand(nb, 3, nproma)
    # Valid 0-based index tables into [NB] (blk) and [NPROMA] (idx).
    edge_idx = numpy.random.randint(0, nproma, size=(nb, nproma, 3)).astype(numpy.int32)
    edge_blk = numpy.random.randint(0, nb, size=(nb, nproma, 3)).astype(numpy.int32)
    z_kin_hor_e = numpy.random.rand(nb, nlev, nproma)
    z_ekinh = numpy.zeros((nb, nlev, nproma))
    run_vectorization_test(
        dace_func=icon_zekinh_gather,
        arrays={
            "e_bln": e_bln,
            "edge_idx": edge_idx,
            "edge_blk": edge_blk,
            "z_kin_hor_e": z_kin_hor_e,
            "z_ekinh": z_ekinh
        },
        params={
            "NB": nb,
            "NLEV": nlev,
            "NPROMA": nproma
        },
        sdfg_name="icon_zekinh_gather",
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
        emission_style=emission_style,
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
