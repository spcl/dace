# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on patterns mined from real ICON / CLOUDSC kernels.

Distilled mock kernels for the load-bearing shapes found in
``mo_solve_nonhydro.f90``, ``mo_velocity_advection.f90`` and
``cloudsc.F90`` (see the corresponding pattern reports in the
``CASCADE_UP_DESIGN.md`` companion notes). Each test is a small
``@dace.program`` that exercises one shape end-to-end through the
canonicalize pipeline and asserts numerical equivalence against a
pure-numpy oracle.

Shapes covered:

* **ICON ``solve_nonhydro`` block** -- per-``jb`` bounds (from
  ``get_indices_c``) plus an ``IF istep == 1`` config-flag guard
  wrapping a sequence of three sibling inner ``jk, jc`` nests
  (file ``mo_solve_nonhydro.f90:540-616``).
* **ICON ``velocity_advection`` block** -- same skeleton, with the
  guard pushed inside the ``jb`` loop body around two inner sibling
  loops over identical ``jc`` ranges (``mo_velocity_advection.f90:485-567``).
* **CLOUDSC IPHASE(JM) per-JM guards** -- ``DO JM: ...; IF
  IPHASE(JM) == 1: ...; IF IPHASE(JM) == 2: ...`` separating four
  sibling ``JL`` loops (``cloudsc.F90:2729-2761``).
* **CLOUDSC KLEV+1 promoted upper bound** -- ``DO JK = 1, KLEV + 1``
  where ``KLEV + 1`` becomes a frontend-promoted symbol; cascade-up
  must lift its iedge assignment past every enclosing loop
  (``cloudsc.F90:2795``).
* **ICON invariant scalar (ZQTMST = 1/PTSPHY)** -- a true loop-
  invariant scalar computed at the top of the outer body and read
  by every inner iteration. Cascade-up target.

Refusal-contract shapes (cascade-up MUST be a no-op):

* **CLOUDSC ``IF (ZQPRETOT < ZEPSEC)`` data guard** -- the condition
  reads a per-JL value, so the assignment behind it (a per-JL zero
  reset) cannot be hoisted past the JL loop.
* **Per-(jb, jc) scalar temporaries** -- arithmetic on per-iteration
  scalars stays where it is.

Each test pins the value-preservation contract (against a numpy
oracle) that canonicalize must deliver today; structural assertions
are added where they reflect a contract the pipeline is known to
honour. Real-world patterns this pipeline does not yet handle
cleanly are documented as ``strict=True`` xfails with precise
reasons.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def _ncond_blocks(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock))


def _nreduce(sdfg):
    from dace.libraries.standard.nodes.reduce import Reduce
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))


# ----------------------------------------------------------------------
# ICON solve_nonhydro: per-jb bound + invariant istep guard + sibling
# inner loops. Distilled from mo_solve_nonhydro.f90:540-616.
# ----------------------------------------------------------------------


@dace.program
def icon_solve_nonhydro_shape(z_exner_ex_pr: dace.float64[N, M], z_exner_ic: dace.float64[N, M],
                              z_dexner_dz_c: dace.float64[N, M], exner: dace.float64[N, M],
                              exner_ref_mc: dace.float64[N, M], exner_exfac: dace.float64[N, M],
                              wgtfac_c: dace.float64[N, M], inv_ddqz_z_full: dace.float64[N, M], istep: dace.int32):
    """ICON solve_nonhydro shape: per-row (= per-jb) range over ``N``
    columns, with an invariant ``IF istep == 1`` guard wrapping a
    sequence of three inner ``jk, jc`` style nests over the row's
    column range. ``istep`` is a config flag invariant w.r.t. the
    outer loop. After canonicalize: guard hoisted out, sibling
    nests fused / co-located."""
    for jb in range(0, N):
        # per-row bound (loop-invariant w.r.t. jk, jc but NOT jb)
        start = jb // 4
        end = M
        if istep == 1:
            # sibling 1: 2D over jk, jc -- the exner_ex_pr update
            for jk in range(0, M):
                for jc in range(start, end):
                    z_exner_ex_pr[jc, jk] = (1.0 + exner_exfac[jc, jk]) * (
                        exner[jc, jk] - exner_ref_mc[jc, jk]) - exner_exfac[jc, jk] * z_exner_ex_pr[jc, jk]
            # sibling 2: 2D, shared jc range, reads z_exner_ex_pr
            for jk in range(1, M):
                for jc in range(start, end):
                    z_exner_ic[jc, jk] = (wgtfac_c[jc, jk] * z_exner_ex_pr[jc, jk] +
                                          (1.0 - wgtfac_c[jc, jk]) * z_exner_ex_pr[jc, jk - 1])
            # sibling 3: 2D, shared jc range, reads z_exner_ic -- fusable with sibling 2
            for jk in range(1, M - 1):
                for jc in range(start, end):
                    z_dexner_dz_c[jc, jk] = (z_exner_ic[jc, jk] - z_exner_ic[jc, jk + 1]) * inv_ddqz_z_full[jc, jk]


def _icon_solve_nonhydro_oracle(arr_in, istep):
    n = arr_in['z_exner_ex_pr'].shape[0]
    m = arr_in['z_exner_ex_pr'].shape[1]
    z_exner_ex_pr = arr_in['z_exner_ex_pr'].copy()
    z_exner_ic = arr_in['z_exner_ic'].copy()
    z_dexner_dz_c = arr_in['z_dexner_dz_c'].copy()
    exner = arr_in['exner']
    exner_ref_mc = arr_in['exner_ref_mc']
    exner_exfac = arr_in['exner_exfac']
    wgtfac_c = arr_in['wgtfac_c']
    inv_ddqz_z_full = arr_in['inv_ddqz_z_full']
    for jb in range(0, n):
        start = jb // 4
        end = m
        if istep == 1:
            for jk in range(0, m):
                for jc in range(start, end):
                    z_exner_ex_pr[jc, jk] = (1.0 + exner_exfac[jc, jk]) * (
                        exner[jc, jk] - exner_ref_mc[jc, jk]) - exner_exfac[jc, jk] * z_exner_ex_pr[jc, jk]
            for jk in range(1, m):
                for jc in range(start, end):
                    z_exner_ic[jc, jk] = (wgtfac_c[jc, jk] * z_exner_ex_pr[jc, jk] +
                                          (1.0 - wgtfac_c[jc, jk]) * z_exner_ex_pr[jc, jk - 1])
            for jk in range(1, m - 1):
                for jc in range(start, end):
                    z_dexner_dz_c[jc, jk] = (z_exner_ic[jc, jk] - z_exner_ic[jc, jk + 1]) * inv_ddqz_z_full[jc, jk]
    return z_exner_ex_pr, z_exner_ic, z_dexner_dz_c


def _icon_solve_nonhydro_args(n, m, rng):
    return dict(z_exner_ex_pr=rng.standard_normal((n, m)),
                z_exner_ic=rng.standard_normal((n, m)),
                z_dexner_dz_c=rng.standard_normal((n, m)),
                exner=rng.standard_normal((n, m)),
                exner_ref_mc=rng.standard_normal((n, m)),
                exner_exfac=rng.standard_normal((n, m)),
                wgtfac_c=rng.uniform(0.1, 0.9, (n, m)),
                inv_ddqz_z_full=rng.uniform(0.5, 2.0, (n, m)))


def test_icon_solve_nonhydro_shape_structure():
    """Locked structural contract for the solve_nonhydro shape:

    * The inner ``jk, jc`` body fully parallelises and each nest
      collapses into one ``map[jk, jc]``: 3 collapsed inner Maps.
    * The outer ``jb`` stays a ``LoopRegion`` (1 surviving loop) -- it
      carries a per-iteration bound (``start = jb // 4``) that pins it
      sequential; no pass should turn it into a Map.
    * The ``IF istep == 1`` guard IS hoisted to the SDFG top level
      (``MoveLoopInvariantIfUp`` with the dead-outside-branch
      relaxation lifts past the per-``jb`` iedge assignment of
      ``start = jb // 4`` because ``start`` is only read inside the
      branch). After the hoist the canonical shape is
      ``if istep == 1: { for jb: { start = jb // 4; 3 sibling maps } }``.
    * No iedge assignment leaks to the SDFG top level whose RHS reads
      ``jb`` (refusal: ``jb`` is undeclared at the SDFG root).
    """
    sdfg = icon_solve_nonhydro_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 3, f'expected 3 collapsed inner maps after canonicalize, got {_nmaps(sdfg)}'
    assert _nloops(sdfg) == 1, (f'expected exactly 1 surviving LoopRegion (outer jb with per-iteration bound), '
                                f'got {_nloops(sdfg)}')
    # MoveLoopInvariantIfUp (terminal, require_full_hoist) lifts the
    # invariant ``IF istep == 1`` guard to the SDFG top level, past the
    # per-jb loop, even though the loop body carries the per-jb iedge
    # assignment ``start = jb // 4`` (dead outside the guard branch).
    top_conds = [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]
    assert len(top_conds) == 1, (f'IF istep == 1 must be hoisted to SDFG top level; got {len(top_conds)} '
                                 f'top-level conditional block(s)')
    assert any('istep' in b[0].as_string for b in top_conds[0].branches), \
        'top-level conditional does not test istep'
    # Refusal contract: no top-level iedge assigns ``jb`` or anything
    # whose RHS reads ``jb`` (jb is undeclared at SDFG root).
    for e in sdfg.edges():
        for lhs, rhs in e.data.assignments.items():
            rhs_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}
            assert 'jb' not in rhs_syms, f'per-jb expression {lhs} = {rhs} leaked to SDFG root'


def test_icon_solve_nonhydro_shape_e2e():
    n, m = 6, 6
    rng = np.random.default_rng(102)
    for istep in (1, 0):
        args = _icon_solve_nonhydro_args(n, m, rng)
        exp_ex, exp_ic, exp_dz = _icon_solve_nonhydro_oracle(args, istep)
        sdfg = icon_solve_nonhydro_shape.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        got = {k: v.copy() for k, v in args.items()}
        sdfg(**got, istep=np.int32(istep), N=n, M=m)
        assert np.allclose(got['z_exner_ex_pr'], exp_ex), f'z_exner_ex_pr istep={istep}'
        assert np.allclose(got['z_exner_ic'], exp_ic), f'z_exner_ic istep={istep}'
        assert np.allclose(got['z_dexner_dz_c'], exp_dz), f'z_dexner_dz_c istep={istep}'


# ----------------------------------------------------------------------
# ICON velocity_advection: per-jb bound + IF istep == 1 guard around two
# inner sibling jk loops sharing the same jc range. Distilled from
# mo_velocity_advection.f90:485-567.
# ----------------------------------------------------------------------


@dace.program
def icon_velocity_advection_istep_shape(z_ekinh: dace.float64[N, M], z_w_concorr_mc: dace.float64[N, M],
                                        w_concorr_c: dace.float64[N, M], kin_hor_e: dace.float64[N, M],
                                        w_concorr_me: dace.float64[N, M], wgtfac_c: dace.float64[N, M],
                                        istep: dace.int32, nflatlev: dace.int32):
    """velocity_advection shape: per-row bound + always-on first inner
    nest + ``IF istep == 1`` guard around two subsequent sibling
    inner ``jk, jc`` nests over the same jc range. The guard is
    loop-invariant w.r.t. the outer (per-row) loop -- a hoist
    candidate; the two guarded siblings are fusion candidates."""
    for jb in range(0, N):
        start = jb // 4
        end = M
        # Always-on: 2D over jk, jc.
        for jk in range(0, M):
            for jc in range(start, end):
                z_ekinh[jc, jk] = 0.5 * kin_hor_e[jc, jk]
        # Guard wrapping two siblings.
        if istep == 1:
            for jk in range(0, M):
                for jc in range(start, end):
                    z_w_concorr_mc[jc, jk] = 0.5 * w_concorr_me[jc, jk]
            for jk in range(nflatlev, M):
                for jc in range(start, end):
                    w_concorr_c[jc, jk] = (wgtfac_c[jc, jk] * z_w_concorr_mc[jc, jk] +
                                           (1.0 - wgtfac_c[jc, jk]) * z_w_concorr_mc[jc, jk - 1])


def _icon_velocity_advection_oracle(args, istep, nflatlev):
    n, m = args['z_ekinh'].shape
    z_ekinh = args['z_ekinh'].copy()
    z_w_concorr_mc = args['z_w_concorr_mc'].copy()
    w_concorr_c = args['w_concorr_c'].copy()
    kin_hor_e = args['kin_hor_e']
    w_concorr_me = args['w_concorr_me']
    wgtfac_c = args['wgtfac_c']
    for jb in range(n):
        start = jb // 4
        end = m
        for jk in range(0, m):
            for jc in range(start, end):
                z_ekinh[jc, jk] = 0.5 * kin_hor_e[jc, jk]
        if istep == 1:
            for jk in range(0, m):
                for jc in range(start, end):
                    z_w_concorr_mc[jc, jk] = 0.5 * w_concorr_me[jc, jk]
            for jk in range(nflatlev, m):
                for jc in range(start, end):
                    w_concorr_c[jc, jk] = (wgtfac_c[jc, jk] * z_w_concorr_mc[jc, jk] +
                                           (1.0 - wgtfac_c[jc, jk]) * z_w_concorr_mc[jc, jk - 1])
    return z_ekinh, z_w_concorr_mc, w_concorr_c


def test_icon_velocity_advection_istep_shape_structure():
    """velocity_advection skeleton: same outer-loop + per-jb bound shape
    as solve_nonhydro, with the guard around two of three inner
    siblings.

    * Inner ``jk, jc`` bodies parallelise and each nest collapses into
      one ``map[jk, jc]``: 3 collapsed Maps.
    * 2 surviving LoopRegions: the outer per-jb loop and (after the
      MLIU hoist) the per-jb loop carried inside the guarded branch
      that handles only the guarded siblings -- the always-on first
      nest stays in its own per-jb loop. Both validate + value-preserve.
    * The ``IF istep == 1`` guard IS hoisted to the SDFG top level
      (MLIU's dead-outside-branch relaxation lifts past the per-jb
      iedge assignments inside the guarded scope).
    """
    sdfg = icon_velocity_advection_istep_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 3, f'expected 3 collapsed inner maps, got {_nmaps(sdfg)}'
    assert _nloops(sdfg) == 2, f'expected 2 surviving LoopRegions, got {_nloops(sdfg)}'
    # The invariant ``IF istep == 1`` guard is hoisted to the SDFG top level.
    top_conds = [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]
    assert len(top_conds) == 1, (f'IF istep == 1 must be hoisted to SDFG top level; got {len(top_conds)} '
                                 f'top-level conditional block(s)')
    assert any('istep' in b[0].as_string for b in top_conds[0].branches), \
        'top-level conditional does not test istep'
    for e in sdfg.edges():
        for lhs, rhs in e.data.assignments.items():
            rhs_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}
            assert 'jb' not in rhs_syms, f'per-jb expression {lhs} = {rhs} leaked to SDFG root'


def test_icon_velocity_advection_istep_shape_e2e():
    n, m = 6, 6
    rng = np.random.default_rng(103)
    nflatlev = 2
    for istep in (1, 0):
        args = dict(z_ekinh=rng.standard_normal((n, m)),
                    z_w_concorr_mc=rng.standard_normal((n, m)),
                    w_concorr_c=rng.standard_normal((n, m)),
                    kin_hor_e=rng.standard_normal((n, m)),
                    w_concorr_me=rng.standard_normal((n, m)),
                    wgtfac_c=rng.uniform(0.1, 0.9, (n, m)))
        exp_ek, exp_wmc, exp_wcc = _icon_velocity_advection_oracle(args, istep, nflatlev)
        sdfg = icon_velocity_advection_istep_shape.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        got = {k: v.copy() for k, v in args.items()}
        sdfg(**got, istep=np.int32(istep), nflatlev=np.int32(nflatlev), N=n, M=m)
        assert np.allclose(got['z_ekinh'], exp_ek), f'z_ekinh istep={istep}'
        assert np.allclose(got['z_w_concorr_mc'], exp_wmc), f'z_w_concorr_mc istep={istep}'
        assert np.allclose(got['w_concorr_c'], exp_wcc), f'w_concorr_c istep={istep}'


# ----------------------------------------------------------------------
# CLOUDSC IPHASE(JM) per-JM phase guards. Distilled from
# cloudsc.F90:2729-2761. The guards depend on JM (the loop variable
# of the outer JM=1..NCLV loop), so they CANNOT be hoisted past JM
# -- only push-into-JL fusion is legal.
# ----------------------------------------------------------------------


@dace.program
def cloudsc_iphase_shape(tendency_t: dace.float64[N], tendency_cld: dace.float64[N, M], fluxq: dace.float64[N, M],
                         psupsatsrc: dace.float64[N, M], iphase: dace.int32[M]):
    """Per-JM phase guards: each JM iteration has 4 sibling JL loops,
    two of them guarded by ``IF IPHASE(JM) == 1`` and
    ``IF IPHASE(JM) == 2``. ``IPHASE(JM)`` is a JM-dependent INTEGER
    array read; cascade-up CANNOT hoist anything past JM (RHS would
    read JM). The four JL siblings remain a single-JL fusion target
    after MoveIfIntoLoop pushes the guards into each JL loop."""
    for jm in range(0, M):
        for jl in range(0, N):
            fluxq[jl, jm] = psupsatsrc[jl, jm] + 1.0
        if iphase[jm] == 1:
            for jl in range(0, N):
                tendency_t[jl] += 0.5 * fluxq[jl, jm]
        if iphase[jm] == 2:
            for jl in range(0, N):
                tendency_t[jl] += 0.25 * fluxq[jl, jm]
        for jl in range(0, N):
            tendency_cld[jl, jm] += fluxq[jl, jm]


def _cloudsc_iphase_oracle(args):
    n = args['tendency_t'].shape[0]
    m = args['fluxq'].shape[1]
    tendency_t = args['tendency_t'].copy()
    tendency_cld = args['tendency_cld'].copy()
    fluxq = args['fluxq'].copy()
    psupsatsrc = args['psupsatsrc']
    iphase = args['iphase']
    for jm in range(m):
        for jl in range(n):
            fluxq[jl, jm] = psupsatsrc[jl, jm] + 1.0
        if iphase[jm] == 1:
            for jl in range(n):
                tendency_t[jl] += 0.5 * fluxq[jl, jm]
        if iphase[jm] == 2:
            for jl in range(n):
                tendency_t[jl] += 0.25 * fluxq[jl, jm]
        for jl in range(n):
            tendency_cld[jl, jm] += fluxq[jl, jm]
    return tendency_t, tendency_cld, fluxq


def test_cloudsc_iphase_shape_structure():
    """The outer ``JM`` loop stays sequential. ``tendency_t[jl] += c(jm) *
    fluxq[jl, jm]`` accumulates over ``JM`` into a per-``JL`` location, but it is
    only ONE statement of the multi-statement ``JM`` body (alongside the
    ``fluxq`` write, the two phase guards, and the ``tendency_cld`` write).
    ``loop_to_reduce`` recognises only single-statement accumulator loops, so it
    cannot lift this reduction without loop fission first isolating it -- and the
    front-loaded recipe runs ``loop_to_reduce`` before fission. So ``JM`` remains
    a ``LoopRegion`` and the four inner ``JL`` statements become Maps. The two
    per-``JM`` phase guards (``iphase[jm] == 1`` / ``== 2``) survive as
    ``ConditionalBlock`` s (their condition reads a per-iteration array).
    Correctness is pinned numerically by ``test_cloudsc_iphase_shape_e2e``.
    """
    sdfg = cloudsc_iphase_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nloops(sdfg) == 1, (f'outer JM stays sequential (multi-statement reduction, not isolated '
                                f'pre-fission); got {_nloops(sdfg)} loops')
    assert _ncond_blocks(sdfg) == 2, (f'both IPHASE phase guards must survive (correct refusal); '
                                      f'got {_ncond_blocks(sdfg)} conditionals')
    assert _nmaps(sdfg) <= 4, f'too many residual maps for the 4 JL statements: {_nmaps(sdfg)}'


def test_cloudsc_iphase_shape_e2e():
    n, m = 8, 5
    rng = np.random.default_rng(104)
    args = dict(tendency_t=rng.standard_normal(n),
                tendency_cld=rng.standard_normal((n, m)),
                fluxq=rng.standard_normal((n, m)),
                psupsatsrc=rng.standard_normal((n, m)),
                iphase=np.array([0, 1, 2, 1, 0], dtype=np.int32))
    exp_t, exp_cld, exp_fq = _cloudsc_iphase_oracle(args)
    sdfg = cloudsc_iphase_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = {k: v.copy() for k, v in args.items()}
    sdfg(**got, N=n, M=m)
    assert np.allclose(got['tendency_t'], exp_t)
    assert np.allclose(got['tendency_cld'], exp_cld)
    assert np.allclose(got['fluxq'], exp_fq)


# ----------------------------------------------------------------------
# CLOUDSC KLEV+1 promoted upper-bound. Distilled from cloudsc.F90:2795.
# The outer loop's upper bound is a promoted symbol expression
# (KLEV + 1); cascade-up must keep its iedge assignment at the outer
# scope, not inside the loop body.
# ----------------------------------------------------------------------


@dace.program
def cloudsc_klev_plus_1_shape(pfplsl: dace.float64[N, M], zpfplsx_qr: dace.float64[N, M],
                              zpfplsx_ql: dace.float64[N, M], klev: dace.int32):
    """``DO JK = 1, KLEV + 1`` -- the upper bound is the frontend-
    promoted ``klev + 1`` expression. Through the canonicalize
    pipeline, the promoted symbol's iedge assignment must end up
    OUTSIDE any inner JL loop and must NOT degrade per-JL writes to
    constant-index writes."""
    for jk in range(0, klev + 1):
        for jl in range(0, N):
            pfplsl[jl, jk] = zpfplsx_qr[jl, jk] + zpfplsx_ql[jl, jk]


def test_cloudsc_klev_plus_1_shape_structure():
    """The headline cascade-up success case. After canonicalize:

    * Both loops become Maps (clean parallelism), and -- being a
      fully-parallel ``jk``/``jl`` nest -- collapse into one ``map[jk,
      jl]``.
    * ``SymbolPropagation`` (read-only-Scalar relaxation) folds the
      frontend-promoted ``klev_plus_1 = klev + 1`` alias directly into
      its uses and the now-dead iedge is dropped. The cleaner post-state
      thus has NO ``klev_plus_1`` symbol anywhere; ``klev`` appears
      directly in the collapsed Map's ``jk`` range as ``(0, klev, 1)``
      (i.e. ``klev_plus_1 - 1`` simplified once the alias is substituted).
    * No surviving alias iedge clutters the root.

    The cascade-up contract is preserved in spirit: invariant promoted
    bounds reach the inner Map cleanly. The previous test wording
    asserted the *symbol-indirection* artifact of the old SymProp
    behavior (alias kept alive, lifted to root by cascade-up); SymProp's
    read-only-Scalar relaxation eliminated that indirection entirely
    (cf. ``tests/passes/symbol_propagation_test.py`` ::
    ``test_cloudsc_kidia_kfdia_promote_then_propagate``).
    """
    sdfg = cloudsc_klev_plus_1_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 1, f'expected one collapsed 2D map, got {_nmaps(sdfg)}'
    assert _nloops(sdfg) == 0, f'no LoopRegion should remain, got {_nloops(sdfg)}'
    # The alias is fully substituted: ``klev_plus_1`` MUST NOT survive as an
    # iedge LHS, but ``klev`` MUST appear in the collapsed Map's range.
    root_keys = {lhs for e in sdfg.edges() for lhs in e.data.assignments}
    assert not any('klev_plus_1' in k for k in root_keys), \
        f'klev_plus_1 alias should be folded and its iedge dropped; got {root_keys}'
    map_entry = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    range_strs = {str(r) for r in map_entry.map.range}
    assert any('klev' in s for s in range_strs), \
        f'collapsed Map jk range should reference klev directly; got {range_strs}'


def test_cloudsc_klev_plus_1_shape_e2e():
    n, m = 10, 8
    klev = m - 1
    rng = np.random.default_rng(105)
    pfplsl = rng.standard_normal((n, m))
    zpfplsx_qr = rng.standard_normal((n, m))
    zpfplsx_ql = rng.standard_normal((n, m))
    sdfg = cloudsc_klev_plus_1_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = pfplsl.copy()
    sdfg(pfplsl=got, zpfplsx_qr=zpfplsx_qr, zpfplsx_ql=zpfplsx_ql, klev=np.int32(klev), N=n, M=m)
    exp = pfplsl.copy()
    for jk in range(0, klev + 1):
        for jl in range(0, n):
            exp[jl, jk] = zpfplsx_qr[jl, jk] + zpfplsx_ql[jl, jk]
    assert np.allclose(got, exp)


# ----------------------------------------------------------------------
# ICON / CLOUDSC ZQTMST = 1 / PTSPHY invariant scalar. Computed once
# at outer-scope, read by every inner iteration. Cascade-up target.
# ----------------------------------------------------------------------


@dace.program
def zqtmst_invariant_scalar_shape(arr: dace.float64[N, M], out: dace.float64[N, M], ptsphy: dace.float64):
    """``ZQTMST = 1 / PTSPHY`` is computed once at the start of the
    cloudsc inner-physics loop and read in every (JK, JL) iteration.
    A correct cascade-up keeps the reciprocal at outer scope; an
    incorrect pipeline would recompute it per iteration."""
    zqtmst = 1.0 / ptsphy
    for jk in range(0, M):
        for jl in range(0, N):
            out[jl, jk] = arr[jl, jk] * zqtmst


def test_zqtmst_invariant_scalar_shape_structure():
    """The reciprocal ``zqtmst = 1 / ptsphy`` should be computed ONCE
    at outer scope and reused. Canonical shape: one collapsed 2D Map
    (the fully-parallel ``jk``/``jl`` nest folds into a single
    ``map[jk, jl]``), no surviving LoopRegions, no per-iteration
    division. The reciprocal's tasklet must NOT live inside the map
    scope (the map scope's tasklet code must reference ``zqtmst`` as an
    input symbol or AccessNode read, not recompute it)."""
    sdfg = zqtmst_invariant_scalar_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 1, f'expected one collapsed 2D map, got {_nmaps(sdfg)}'
    assert _nloops(sdfg) == 0, f'no LoopRegion should remain, got {_nloops(sdfg)}'
    # No tasklet inside a Map scope should compute ``1.0 / ptsphy`` --
    # the reciprocal should live in the outer state. Conservative test:
    # search every tasklet's code for the division literal.
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.Tasklet):
            code = getattr(n.code, 'as_string', '')
            # Tasklets inside Map scope are reachable from a MapEntry.
            # Heuristic: any tasklet whose code computes ``1.0 / ptsphy``
            # OR ``1 / ptsphy`` is the buggy per-iteration reciprocal.
            assert '1.0 / ptsphy' not in code and '1 / ptsphy' not in code, \
                f'per-iteration reciprocal leaked into tasklet: {code}'


def test_zqtmst_invariant_scalar_shape_e2e():
    n, m = 8, 7
    rng = np.random.default_rng(106)
    arr = rng.standard_normal((n, m))
    out = np.zeros_like(arr)
    sdfg = zqtmst_invariant_scalar_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    sdfg(arr=arr, out=out, ptsphy=np.float64(2.5), N=n, M=m)
    exp = arr * (1.0 / 2.5)
    assert np.allclose(out, exp)


# ----------------------------------------------------------------------
# CLOUDSC ``IF (ZQPRETOT(JL) < ZEPSEC)`` data-dependent guard.
# Distilled from cloudsc.F90:2705-2719. The guard reads a per-JL
# value; cascade-up MUST refuse to lift anything past JL.
# ----------------------------------------------------------------------


@dace.program
def cloudsc_zqpretot_data_guard_shape(zcovptot: dace.float64[N], zqpretot: dace.float64[N], zpfplsx: dace.float64[N, M],
                                      zepsec: dace.float64):
    """Per-JL accumulator + per-JL data-guarded reset. The data
    guard reads ``zqpretot[jl]``; the conditional zero-write to
    ``zcovptot[jl]`` is per-JL. Cascade-up must be a no-op."""
    for jl in range(0, N):
        zqpretot[jl] = 0.0
        for jm in range(0, M):
            zqpretot[jl] += zpfplsx[jl, jm]
        if zqpretot[jl] < zepsec:
            zcovptot[jl] = 0.0


def _zqpretot_oracle(args, zepsec):
    n = args['zcovptot'].shape[0]
    m = args['zpfplsx'].shape[1]
    zcovptot = args['zcovptot'].copy()
    zqpretot = args['zqpretot'].copy()
    zpfplsx = args['zpfplsx']
    for jl in range(n):
        zqpretot[jl] = 0.0
        for jm in range(m):
            zqpretot[jl] += zpfplsx[jl, jm]
        if zqpretot[jl] < zepsec:
            zcovptot[jl] = 0.0
    return zcovptot, zqpretot


def test_cloudsc_zqpretot_data_guard_shape_structure():
    """The accumulator + data-guarded reset both parallelize. After
    canonicalize:

    * The outer JL loop becomes a Map (per-JL reads/writes).
    * The inner JM reduction ``zqpretot[jl] += zpfplsx[jl, jm]`` is a clean
      single-statement accumulator, so ``loop_to_reduce`` lifts it to a
      ``Reduce`` library node (reducing ``zpfplsx[jl, 0:M]`` over the JM axis
      into ``zqpretot[jl]``) -- one Map plus one Reduce, no residual loop.
    * The data-dependent guard ``IF (ZQPRETOT(JL) < ZEPSEC)`` still SURVIVES
      as a ``ConditionalBlock`` (correct refusal -- its condition reads
      ``zqpretot[jl]`` per JL).

    Correctness (including the guard branching both ways) is pinned
    numerically by ``test_cloudsc_zqpretot_data_guard_shape_e2e``.
    """
    sdfg = cloudsc_zqpretot_data_guard_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 1, f'expected the JL nest as one Map, got {_nmaps(sdfg)} maps'
    assert _nreduce(sdfg) == 1, f'inner JM accumulator must lift to a Reduce node, got {_nreduce(sdfg)}'
    assert _nloops(sdfg) == 0, (f'both the JL nest and the JM reduction should parallelize, '
                                f'got {_nloops(sdfg)} residual loops')
    assert _ncond_blocks(sdfg) == 1, (f'data-dependent guard must survive (correct refusal); '
                                      f'got {_ncond_blocks(sdfg)} conditionals')


def test_cloudsc_zqpretot_data_guard_shape_e2e():
    n, m = 10, 4
    zepsec = 1e-6
    rng = np.random.default_rng(107)
    args = dict(zcovptot=rng.standard_normal(n),
                zqpretot=rng.standard_normal(n),
                zpfplsx=rng.standard_normal((n, m)) * 1e-7)
    exp_cv, exp_qp = _zqpretot_oracle(args, zepsec)
    sdfg = cloudsc_zqpretot_data_guard_shape.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = {k: v.copy() for k, v in args.items()}
    sdfg(**got, zepsec=np.float64(zepsec), N=n, M=m)
    assert np.allclose(got['zcovptot'], exp_cv)
    assert np.allclose(got['zqpretot'], exp_qp)


# ----------------------------------------------------------------------
# CLOUDSC config-flag chain ``IF (NSSOPT == 0) THEN ... ELSEIF
# (NSSOPT == 1)`` inside the JL/JK nest. Distilled from
# cloudsc.F90:1431-1444. The config flag is invariant on every loop;
# MoveLoopInvariantIfUp should hoist it out, after which the
# branches become independent computations to fuse with their
# siblings.
# ----------------------------------------------------------------------


@dace.program
def cloudsc_nssopt_config_chain_shape(arr: dace.float64[N, M], out: dace.float64[N, M], nssopt: dace.int32):
    """The ``NSSOPT`` config-flag chain: an invariant if/elif ladder
    inside the JK/JL nest. Hoistable to the outermost scope; the
    resulting three top-level branches each carry a clean parallel
    nest."""
    for jk in range(0, M):
        for jl in range(0, N):
            if nssopt == 0:
                out[jl, jk] = arr[jl, jk]
            elif nssopt == 1:
                out[jl, jk] = arr[jl, jk] * 2.0
            else:
                out[jl, jk] = arr[jl, jk] + 1.0


def test_cloudsc_nssopt_config_chain_shape_e2e():
    n, m = 8, 6
    rng = np.random.default_rng(108)
    arr = rng.standard_normal((n, m))
    for nssopt in (0, 1, 2):
        sdfg = cloudsc_nssopt_config_chain_shape.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        out = np.zeros_like(arr)
        sdfg(arr=arr, out=out, nssopt=np.int32(nssopt), N=n, M=m)
        if nssopt == 0:
            exp = arr.copy()
        elif nssopt == 1:
            exp = arr * 2.0
        else:
            exp = arr + 1.0
        assert np.allclose(out, exp), f'nssopt={nssopt}'


# ----------------------------------------------------------------------
# Standalone MoveLoopInvariantIfUp on the ICON shapes -- locks in the
# istep top-level hoist contract independently of the pipeline.
# Pipeline integration is deferred (MoveIfIntoLoop ping-pong; see
# pipeline.py NOTE near move_loop_invariant_if_up).
# ----------------------------------------------------------------------


def _run_canonicalize_pre_parallelize(kernel):
    """Drive the canonicalize pipeline up to (but not through) the
    post-fission ``parallelize`` stage, so the loop body has the shape MLIU
    expects to see (post fission / normalize / ssa / cascade-up).

    The front-loaded recipe has TWO ``parallelize`` stages -- an early
    ``LoopToMap`` (right after the reduction recipe, before fission) and the
    post-fission one. Stop before the LAST: the early ``LoopToMap`` has already
    turned the fully-parallel inner nests into Maps, leaving the per-``jb`` loop
    with its still-invariant ``istep`` guard for MLIU to hoist."""
    from dace.transformation.passes.canonicalize.pipeline import _build_stages
    sdfg = kernel.to_sdfg(simplify=True)
    stages = _build_stages()
    last_parallelize = max(i for i, (label, _) in enumerate(stages) if label == 'parallelize')
    for i, (label, unit) in enumerate(stages):
        if i == last_parallelize:
            break
        unit.apply_pass(sdfg, {})
    return sdfg


def test_icon_solve_nonhydro_istep_hoist_standalone_mliu():
    """Standalone: after the pre-parallelize canonicalize stages, the
    ``IF istep == 1`` guard hoists past the per-jb loop via MLIU's
    dead-outside-branch extension. Locks the extended-match contract
    irrespective of where MLIU sits in the pipeline."""
    from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
    sdfg = _run_canonicalize_pre_parallelize(icon_solve_nonhydro_shape)
    result = MoveLoopInvariantIfUp().apply_pass(sdfg, {})
    assert result and result >= 1, 'MLIU must hoist the istep guard'
    sdfg.validate()
    top_conds = [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]
    assert len(top_conds) == 1, f'istep guard must be at SDFG top level; got {len(top_conds)} top conds'
    top_branch_conds = [b[0].as_string for b in top_conds[0].branches]
    assert any('istep' in c for c in top_branch_conds), \
        f'top-level conditional does not test istep; branches={top_branch_conds}'


def test_icon_velocity_advection_istep_hoist_standalone_mliu():
    """Same standalone contract for velocity_advection's istep guard."""
    from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
    sdfg = _run_canonicalize_pre_parallelize(icon_velocity_advection_istep_shape)
    result = MoveLoopInvariantIfUp().apply_pass(sdfg, {})
    assert result and result >= 1, 'MLIU must hoist the istep guard'
    sdfg.validate()
    top_conds = [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]
    assert len(top_conds) == 1, f'istep guard must be at SDFG top level; got {len(top_conds)} top conds'
    top_branch_conds = [b[0].as_string for b in top_conds[0].branches]
    assert any('istep' in c for c in top_branch_conds), \
        f'top-level conditional does not test istep; branches={top_branch_conds}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
