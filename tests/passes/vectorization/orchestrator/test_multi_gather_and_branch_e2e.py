# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical tests for multi-gather, direct-gather-store, and
branchy (predicated RMW) kernels on the ``VectorizeCPUMultiDim`` tile path.

These pin the specific behaviours that the ICON ``zekinh`` cell-from-edges
interpolation and the CLOUDSC ``tidy`` branch exercise, reduced to the
smallest kernel that reproduces each bug (all regressions fixed 2026-06-15):

* **Direct gather -> output store** -- ``out[..] = A[.., idx[..]]`` with NO
  intervening compute. The bridge -> output ``TileStore`` must carry the
  per-iteration OUTER-dim index, not the full extent (else every outer
  iteration overwrites row ``[0, .., :]``).
* **Multiple distinct gathers of one array** -- ``A[idx0] + A[idx1] + ...``.
  Each distinct indirect index needs its OWN index tile + ``TileLoad``;
  staging only the first and rewiring all consumers aliases them.
* **Multiple distinct structured reads of one array** -- ``e[.,0,.] + e[.,1,.]
  + e[.,2,.]`` (constant non-tile index varies). Same per-distinct-subset
  staging contract as the gather case.
* **Gather x structured product sum** -- the full ICON shape
  ``sum_m e[.,m,.] * A[blk[.,m], .., idx[.,m]]``.
* **Predicated store / RMW / chain** -- a guarded ``if .. : out = ..`` body.
  The K=1 comparison must NOT cast operands to the (bool) output type, and a
  const-assign arm must not be spuriously broadcast as a scalar.

Contract: bit-equivalence (modulo FMA reorder) of the ``VectorizeCPUMultiDim``
output vs the unvectorised reference.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)

NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")
S1 = dace.symbol("S1")
S2 = dace.symbol("S2")

_NB, _NLEV, _NPROMA = 2, 16, 64


def _run_compare(kern, make_inputs, params, widths=(8, ), branch_mode="merge", seeds=(0, 1, 2)):
    """Vectorise ``kern`` with ``VectorizeCPUMultiDim`` and assert bit-equivalence
    with the unvectorised reference across several random seeds."""
    import copy
    ref_sdfg = kern.to_sdfg(simplify=False)
    ref_sdfg.simplify()
    vec_sdfg = copy.deepcopy(ref_sdfg)
    vec_sdfg.name = ref_sdfg.name + "_vec"
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=widths,
                        target_isa=ISA.SCALAR,
                        remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE,
                        branch_mode=branch_mode)).apply_pass(vec_sdfg, {})
    vec_sdfg.validate()
    c_ref = ref_sdfg.compile()
    c_vec = vec_sdfg.compile()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        base = make_inputs(rng)
        ref = {k: v.copy() for k, v in base.items()}
        tst = {k: v.copy() for k, v in base.items()}
        c_ref(**ref, **params)
        c_vec(**tst, **params)
        for name in base:
            np.testing.assert_allclose(tst[name],
                                       ref[name],
                                       rtol=1e-12,
                                       atol=1e-12,
                                       err_msg=f"{kern.name} seed={seed} array={name}")


def _icon_inputs(rng):
    return dict(
        e_bln=rng.random((_NB, 3, _NPROMA)),
        edge_idx=rng.integers(0, _NPROMA, (_NB, _NPROMA, 3)).astype(np.int32),
        edge_blk=rng.integers(0, _NB, (_NB, _NPROMA, 3)).astype(np.int32),
        z_kin_hor_e=rng.random((_NB, _NLEV, _NPROMA)),
        z_ekinh=np.zeros((_NB, _NLEV, _NPROMA)),
    )


_ICON_PARAMS = {"NB": _NB, "NLEV": _NLEV, "NPROMA": _NPROMA}

# ---------------------------------------------------------------- direct gather store


@dace.program
def _gather_idx_direct(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA,
                                                                                3], edge_blk: dace.int32[NB, NPROMA, 3],
                       z_kin_hor_e: dace.float64[NB, NLEV, NPROMA], z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = z_kin_hor_e[jb, jk, edge_idx[jb, jc, 0]]


def test_direct_gather_to_output_store():
    """``out[jb,jk,jc] = A[jb,jk,idx[jb,jc,0]]`` -- direct gather assigned to the
    output (no compute). Regression: the bridge->output TileStore dropped the
    per-iter jb/jk index (wrote the full extent), so every (jb,jk) overwrote
    row [0,0,:]."""
    _run_compare(_gather_idx_direct, _icon_inputs, _ICON_PARAMS)


@dace.program
def _gather_both_direct(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA, 3],
                        edge_blk: dace.int32[NB, NPROMA, 3], z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
                        z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]]


def test_direct_double_gather_to_output_store():
    """Direct two-dim gather (blk dim-0 + idx dim-2) assigned to output."""
    _run_compare(_gather_both_direct, _icon_inputs, _ICON_PARAMS)


# ---------------------------------------------------------------- multi-gather of one array


@dace.program
def _gather_sum2(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA,
                                                                          3], edge_blk: dace.int32[NB, NPROMA, 3],
                 z_kin_hor_e: dace.float64[NB, NLEV, NPROMA], z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk,
                        jc] = (z_kin_hor_e[jb, jk, edge_idx[jb, jc, 0]] + z_kin_hor_e[jb, jk, edge_idx[jb, jc, 1]])


@dace.program
def _gather_sum3(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA,
                                                                          3], edge_blk: dace.int32[NB, NPROMA, 3],
                 z_kin_hor_e: dace.float64[NB, NLEV, NPROMA], z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk,
                        jc] = (z_kin_hor_e[jb, jk, edge_idx[jb, jc, 0]] + z_kin_hor_e[jb, jk, edge_idx[jb, jc, 1]] +
                               z_kin_hor_e[jb, jk, edge_idx[jb, jc, 2]])


@pytest.mark.parametrize("kern", [_gather_sum2, _gather_sum3])
def test_multiple_distinct_gathers_of_one_array(kern):
    """``A[idx0] + A[idx1] (+ A[idx2])`` -- each distinct gather index must
    materialise its OWN index tile + TileLoad (regression: all aliased idx0)."""
    _run_compare(kern, _icon_inputs, _ICON_PARAMS)


# ---------------------------------------------------------------- multi structured of one array


@dace.program
def _struct_sum3(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA,
                                                                          3], edge_blk: dace.int32[NB, NPROMA, 3],
                 z_kin_hor_e: dace.float64[NB, NLEV, NPROMA], z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = e_bln[jb, 0, jc] + e_bln[jb, 1, jc] + e_bln[jb, 2, jc]


def test_multiple_distinct_structured_reads_of_one_array():
    """``e[.,0,.] + e[.,1,.] + e[.,2,.]`` -- distinct constant non-tile index per
    read; each needs its own structured TileLoad (regression: aliased to m=0)."""
    _run_compare(_struct_sum3, _icon_inputs, _ICON_PARAMS)


# ---------------------------------------------------------------- gather x structured (full ICON)


@dace.program
def _icon_zekinh_full(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA,
                                                                               3], edge_blk: dace.int32[NB, NPROMA, 3],
                      z_kin_hor_e: dace.float64[NB, NLEV, NPROMA], z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = (e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                                       e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                                       e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])


def test_icon_zekinh_full_gather_times_structured():
    """The full ICON ``zekinh`` shape: 3 terms of ``e[.,m,.] * A[blk[.,m], ..,
    idx[.,m]]`` summed -- combines per-distinct-gather + per-distinct-structured
    staging. Bit-equal to the unvectorised reference."""
    _run_compare(_icon_zekinh_full, _icon_inputs, _ICON_PARAMS)


# ---------------------------------------------------------------- predicated branch (cloudsc tidy)


def _branch_inputs(rng):
    return dict(a=rng.random((_NLEV, _NPROMA)), b=rng.random((_NLEV, _NPROMA)), c=rng.random((_NLEV, _NPROMA)))


_BRANCH_PARAMS = {"S1": _NLEV, "S2": _NPROMA}


@dace.program
def _pred_store(a: dace.float64[S1, S2], b: dace.float64[S1, S2], c: dace.float64[S1, S2]):
    for i in range(S1):
        for j in range(S2):
            if a[i, j] < 0.5:
                b[i, j] = 0.0


@dace.program
def _pred_rmw(a: dace.float64[S1, S2], b: dace.float64[S1, S2], c: dace.float64[S1, S2]):
    for i in range(S1):
        for j in range(S2):
            if a[i, j] < 0.5:
                b[i, j] = b[i, j] + a[i, j]


@dace.program
def _pred_chain(a: dace.float64[S1, S2], b: dace.float64[S1, S2], c: dace.float64[S1, S2]):
    for i in range(S1):
        for j in range(S2):
            if a[i, j] < 0.5:
                c[i, j] = c[i, j] + b[i, j]
                b[i, j] = 0.0


@pytest.mark.parametrize("kern", [_pred_store, _pred_rmw, _pred_chain])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_predicated_branch_body(kern, branch_mode):
    """Guarded ``if a < 0.5: ...`` body. Regression: the K=1 comparison cast its
    operands to the (bool) output type -- ``(bool)1e-12 -> 1`` -- so the mask was
    always false; and a const-assign arm was spuriously scalar-broadcast."""
    _run_compare(kern, _branch_inputs, _BRANCH_PARAMS, branch_mode=branch_mode)
