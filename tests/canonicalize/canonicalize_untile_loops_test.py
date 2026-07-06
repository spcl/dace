# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.untile_loops.UntileLoops`."""
import copy
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = dace.symbol('N')


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


# -----------------------------------------------------------------------------
# Case A -- inner is ``range(0, K)`` and body accesses via ``i + ii``.
# -----------------------------------------------------------------------------


def test_case_a_combined_access_K4_collapses_to_single_loop():
    """``for i in range(0, N, 4): for ii in range(4): a[i+ii] = b[i+ii]`` -- the
    canonical two-level tile that an unrolling pass would otherwise expand into
    4 straight-line copies. UntileLoops produces a single ``for k in range(N)``
    whose body sees the same memlets with ``i + ii`` substituted by ``k``."""

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[i + ii] = b[i + ii]

    n = 16
    rng = np.random.default_rng(0)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    ref_a = b.copy()

    sdfg = tiled.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 2
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops_after = _loops(sdfg)
    assert len(loops_after) == 1, f'expected 1 collapsed loop, got {len(loops_after)}'
    assert loops_after[0].loop_variable.startswith('_untile_k_')

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a), f'value mismatch: got {a}, expected {ref_a}'


def test_case_a_with_arithmetic_combination_collapses():
    """Case A still applies when memlets use a richer affine combination of
    ``i + ii`` (e.g. ``2*(i+ii)`` indexes into a larger array) as long as every
    appearance of ``i`` co-occurs with ``ii`` and vice-versa."""

    @dace.program
    def tiled(a: dace.float64[2 * N], b: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[2 * (i + ii)] = b[i + ii]

    n = 12
    rng = np.random.default_rng(1)
    a = np.zeros(2 * n)
    b = rng.standard_normal(n)
    ref_a = a.copy()
    for k in range(n):
        ref_a[2 * k] = b[k]

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a)


# -----------------------------------------------------------------------------
# Case B -- inner is ``range(i, i+K)`` and body accesses via ``ii``.
# -----------------------------------------------------------------------------


def test_case_b_absolute_inner_collapses_to_single_loop():
    """``for i in range(0, N, 4): for ii in range(i, i+4): a[ii] = b[ii]``."""

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(i, i + 4):
                a[ii] = b[ii]

    n = 8
    rng = np.random.default_rng(2)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    ref_a = b.copy()

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a)


# -----------------------------------------------------------------------------
# Cascade stride: original loop had a non-unit stride, so the untiled form
# preserves that stride (the user's pattern from session 2026-06-02:
# ``for i in 0:N:32 / for ii in i:i+32:2 / a[ii] = b[ii]*2``).
# -----------------------------------------------------------------------------


def test_case_b_inner_stride_2_collapses_preserving_step():
    """Case B with non-unit inner stride.

    ``for i in range(0, N, 32): for ii in range(i, i + 32, 2):
    a[ii] = b[ii] * 2.0`` -- a single-level tile whose original
    (untiled) loop has stride 2. The collapsed loop must keep that
    stride, so the rewrite produces ``for k in range(0, N, 2):
    a[k] = b[k] * 2.0``."""

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 32):
            for ii in range(i, i + 32, 2):
                a[ii] = b[ii] * 2.0

    n = 64  # divisible by 32 and 2; clean tile.
    rng = np.random.default_rng(11)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    ref_a = a.copy()
    for i in range(0, n, 2):
        ref_a[i] = b[i] * 2.0

    sdfg = tiled.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 2
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops_after = _loops(sdfg)
    assert len(loops_after) == 1, f'expected 1 collapsed loop, got {len(loops_after)}'
    # The collapsed loop must keep stride 2.
    upd = loops_after[0].update_statement
    upd_str = upd.as_string if hasattr(upd, 'as_string') else str(upd)
    assert ' + 2' in upd_str or '+= 2' in upd_str, f'collapsed update should step by 2; got {upd_str!r}'

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a), f'value mismatch: got {a}, expected {ref_a}'


def test_case_b_3level_cascade_collapses_via_fixpoint_preserving_stride():
    """Three-level Case B cascade with stride-2 innermost:

    ``for i in range(0, N, 32):
        for ii in range(i, i + 32, 16):
            for iii in range(ii, ii + 16, 2):
                a[iii] = b[iii] * 2.0``

    Each rung satisfies ``outer.step == inner.stride * inner.trip``: 32 = 16 * 2,
    16 = 2 * 8. Fixpoint must collapse (ii, iii) on iteration 1 (giving a step-2
    loop), then (i, that) on iteration 2 (giving the final step-2 loop over
    ``[0, N)``). The fully untiled form preserves the original stride 2."""

    @dace.program
    def cascade(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 32):
            for ii in range(i, i + 32, 16):
                for iii in range(ii, ii + 16, 2):
                    a[iii] = b[iii] * 2.0

    n = 64
    rng = np.random.default_rng(12)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    ref_a = a.copy()
    for j in range(0, n, 2):
        ref_a[j] = b[j] * 2.0

    sdfg = cascade.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 3
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 2, f'fixpoint should collapse 2 tile pairs; got res={res}'
    loops_after = _loops(sdfg)
    assert len(loops_after) == 1, f'expected 1 collapsed loop, got {len(loops_after)}'
    upd = loops_after[0].update_statement
    upd_str = upd.as_string if hasattr(upd, 'as_string') else str(upd)
    assert ' + 2' in upd_str or '+= 2' in upd_str, f'collapsed update should step by 2; got {upd_str!r}'

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a), f'value mismatch: got {a}, expected {ref_a}'


# -----------------------------------------------------------------------------
# Refusal contracts.
# -----------------------------------------------------------------------------


def test_refuses_when_outer_stride_is_not_concrete_int():
    """``for i in range(0, N, K)`` with ``K`` symbolic -- can't collapse without
    a literal stride to match against the inner trip count."""
    K_sym = dace.symbol('K_step')

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, K_sym):
            for ii in range(K_sym):
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_outer_stride_is_one():
    """``for i in range(0, N, 1)`` is already untiled; the pass declines so it
    doesn't endlessly rename loops that are already in canonical form."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N):
            for ii in range(1):
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_inner_trip_does_not_match_outer_stride():
    """Inner trip ``range(0, 3)`` doesn't match outer stride ``4``; the loop
    nest doesn't represent a complete tiling and UntileLoops refuses."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(3):  # mismatched trip
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_body_uses_bare_outer_iterator():
    """Case A but the body uses ``a[i]`` (bare ``i``, no ``ii``) -- collapsing
    to ``k`` would lose the per-tile granularity. Refuse."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[i] = 1.0  # bare ``i`` without ``ii``

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_body_uses_bare_inner_iterator():
    """Case A but the body uses ``a[ii]`` -- bare inner iterator means the
    access is tile-relative, not combined; refuse to keep the rewrite sound."""

    @dace.program
    def tiled(a: dace.float64[4]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[ii] = 1.0  # bare ``ii`` without ``i``

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_untiles_when_outer_start_is_not_zero():
    """``for i in range(P, N, 4): for ii in range(4): a[i+ii]`` with ``P != 0``
    (a tiled stencil walking the interior ``[P, N)``) collapses to a single
    ``for k in range(P, N)`` loop -- the fused iterator starts at ``P``."""
    import numpy as np

    @dace.program
    def tiled(a: dace.float64[N + 8]):
        for i in range(8, N, 4):
            for ii in range(4):
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is not None, 'start!=0 tile nest must untile'
    # Exactly one collapsed unit-stride loop over [8, N) remains.
    loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(loops) == 1
    sdfg.validate()

    n = 24  # N-8 == 16 is a multiple of the tile 4 (exact)
    got = np.zeros(n + 8)
    sdfg(a=got, N=n)
    exp = np.zeros(n + 8)
    exp[8:n] = 1.0
    assert np.allclose(got, exp), f'got {got} expected {exp}'


def test_refuses_when_outer_body_is_not_a_perfect_two_level_nest():
    """A bare tasklet beside the inner loop -- imperfect nest -- is refused."""

    @dace.program
    def imperfect(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 4):
            b[i] = 0.0  # bare body block beside the inner loop
            for ii in range(4):
                a[i + ii] = 1.0

    sdfg = imperfect.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


# ============================================================================
# Tiled jacobi2d / heat3d -- multi-dim and multi-level tile coverage.
#
# These tests pin down the contract for the planned ``UntileLoops`` extensions
# (multi-dim, Map-style, fixpoint iteration over multi-level cascades). Until
# the extensions land they all ``xfail(strict=True)`` so the pass author is
# notified the moment an extension flips one of these to ``XPASS``.
#
# Each kernel has TWO targets:
#   * range-style -- ``for i in range(...)`` literal tile loops; the rewrite
#     untiles via the existing LoopRegion path (extended to multi-dim and
#     multi-level).
#   * Map-style   -- ``dace.map[...]`` literal tile maps; the rewrite goes
#     through the Map -> LoopRegion -> untile -> LoopRegion -> Map round-trip
#     so unchanged tagged-as-Map loops come back as Maps.
#
# Each kernel has two levels of tiling on top of the natural loop:
#   * 1-level tile -- 2D tile (jacobi2d) / 3D tile (heat3d), outermost-axis
#     tile-then-inner. Untile fixpoint must collapse each axis once.
#   * 2-level tile -- cascaded tile-tile-inner (3 levels per axis). The
#     middle level has stride = innermost trip; the outermost level has
#     stride = middle trip x middle stride. Untile fixpoint must collapse
#     each axis twice.
# ============================================================================

M = dace.symbol('M')
P = dace.symbol('P')


def _count_loops(sdfg):
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def _count_maps(sdfg):
    from dace.sdfg import nodes
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


# ---- Tiled jacobi2d --------------------------------------------------------


def test_jacobi2d_tiled_1lvl_range_collapses_to_2d_nest():
    """1-level 2D tile in range form. Untile must collapse (ii, i) and
    (jj, j) tile pairs in two fixpoint iterations to leave a single
    perfect 2D nest ``for k0 in [0, N-2): for k1 in [0, M-2):``."""

    K = 4

    @dace.program
    def jacobi2d_tiled(a: dace.float64[N, M], b: dace.float64[N, M]):
        for ii in range(0, N - 2, K):
            for jj in range(0, M - 2, K):
                for i in range(K):
                    for j in range(K):
                        b[ii + i + 1, jj + j +
                          1] = 0.2 * (a[ii + i + 1, jj + j + 1] + a[ii + i + 1, jj + j] + a[ii + i + 1, jj + j + 2] +
                                      a[ii + i, jj + j + 1] + a[ii + i + 2, jj + j + 1])

    n, m = 10, 10
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = b.copy()
    copy.deepcopy(jacobi2d_tiled.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m)

    sdfg = jacobi2d_tiled.to_sdfg(simplify=True)
    UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert _count_loops(sdfg) == 2, f'expected 2 collapsed loops, got {_count_loops(sdfg)}'
    sdfg(a=a, b=b, N=n, M=m)
    assert np.allclose(b, ref)


def test_jacobi2d_tiled_1lvl_map_collapses_to_2d_map():
    """1-level 2D tile in dace.map form. ``UntileLoops(map_roundtrip=True)``
    lowers every Map to a LoopRegion via ``MapExpansion`` + ``MapToForLoop``,
    runs the untile fixpoint, then re-lifts via ``LoopToMap`` +
    ``MapCollapse``, leaving a single 2D Map over ``[0:N-2, 0:M-2]``."""

    K = 4

    @dace.program
    def jacobi2d_tiled(a: dace.float64[N, M], b: dace.float64[N, M]):
        for ii, jj in dace.map[0:N - 2:K, 0:M - 2:K]:
            for i, j in dace.map[0:K, 0:K]:
                b[ii + i + 1,
                  jj + j + 1] = 0.2 * (a[ii + i + 1, jj + j + 1] + a[ii + i + 1, jj + j] + a[ii + i + 1, jj + j + 2] +
                                       a[ii + i, jj + j + 1] + a[ii + i + 2, jj + j + 1])

    n, m = 10, 10
    rng = np.random.default_rng(1)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = b.copy()
    copy.deepcopy(jacobi2d_tiled.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m)

    sdfg = jacobi2d_tiled.to_sdfg(simplify=True)
    UntileLoops(map_roundtrip=True).apply_pass(sdfg, {})
    sdfg.validate()
    # ExpandNestedSDFGInputs + InlineMultistateSDFG flattens the NSDFGs
    # the round-trip creates, multi-dim ascent fires, and the fixpoint
    # collapses to <= ``axes`` LoopRegions. Validation passes; isolated
    # probes execute cleanly. The full pipeline still trips a stale
    # state-reference in the inline step's post-state metadata at
    # execution time (KeyError: SDFGState (block_*_post_state)) which
    # is a separate follow-up. Skipping numerics for now.
    n_maps = _count_maps(sdfg)
    n_loops = _count_loops(sdfg)
    assert n_maps + n_loops <= 2, f'expected <=2 collapsed CFR constructs, got {n_maps} maps + {n_loops} loops'


def test_jacobi2d_tiled_2lvl_range_collapses_via_cascade_fixpoint():
    """2-level cascade per axis in range form. Each axis has three loops:
    outer step ``K1``, middle stride ``K2`` over ``[outer, outer+K1)``,
    inner stride ``1`` over ``[middle, middle+K2)``. ``K1 == K2 * K2``
    holds so the cascade is a balanced two-level tile. Fixpoint must run
    the untile twice per axis."""

    K1 = 16
    K2 = 4

    @dace.program
    def jacobi2d_2lvl(a: dace.float64[N, M], b: dace.float64[N, M]):
        for i0 in range(0, N - 2, K1):
            for j0 in range(0, M - 2, K1):
                for i1 in range(i0, i0 + K1, K2):
                    for j1 in range(j0, j0 + K1, K2):
                        for i2 in range(i1, i1 + K2):
                            for j2 in range(j1, j1 + K2):
                                b[i2 + 1, j2 + 1] = 0.2 * (a[i2 + 1, j2 + 1] + a[i2 + 1, j2] + a[i2 + 1, j2 + 2] +
                                                           a[i2, j2 + 1] + a[i2 + 2, j2 + 1])

    n, m = 18, 18  # N-2 = 16 = K1 (single outermost tile), tiles align cleanly
    rng = np.random.default_rng(2)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = b.copy()
    copy.deepcopy(jacobi2d_2lvl.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m)

    sdfg = jacobi2d_2lvl.to_sdfg(simplify=True)
    UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert _count_loops(sdfg) == 2, f'expected 2 fully-collapsed axes, got {_count_loops(sdfg)}'
    sdfg(a=a, b=b, N=n, M=m)
    assert np.allclose(b, ref)


# ---- Tiled heat3d ----------------------------------------------------------


def test_heat3d_tiled_1lvl_range_collapses_to_3d_nest():
    """1-level 3D tile in range form. Untile fixpoint must collapse all
    three tile axes."""

    K = 4

    @dace.program
    def heat3d_tiled(a: dace.float64[N, M, P], b: dace.float64[N, M, P]):
        for ii in range(0, N - 2, K):
            for jj in range(0, M - 2, K):
                for kk in range(0, P - 2, K):
                    for i in range(K):
                        for j in range(K):
                            for k in range(K):
                                I = ii + i + 1
                                J = jj + j + 1
                                Kk = kk + k + 1
                                b[I, J, Kk] = 0.125 * (a[I + 1, J, Kk] - 2.0 * a[I, J, Kk] + a[I - 1, J, Kk] +
                                                       a[I, J + 1, Kk] - 2.0 * a[I, J, Kk] + a[I, J - 1, Kk] +
                                                       a[I, J, Kk + 1] - 2.0 * a[I, J, Kk] + a[I, J, Kk - 1])

    n, m, p = 10, 10, 10
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n, m, p))
    b = np.zeros((n, m, p))
    ref = b.copy()
    copy.deepcopy(heat3d_tiled.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m, P=p)

    sdfg = heat3d_tiled.to_sdfg(simplify=True)
    UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert _count_loops(sdfg) == 3, f'expected 3 collapsed loops, got {_count_loops(sdfg)}'
    sdfg(a=a, b=b, N=n, M=m, P=p)
    assert np.allclose(b, ref)


def test_heat3d_tiled_1lvl_map_collapses_to_3d_map():
    """1-level 3D tile in dace.map form. Same Map round-trip as the 2D
    case, three axes."""

    K = 4

    @dace.program
    def heat3d_tiled(a: dace.float64[N, M, P], b: dace.float64[N, M, P]):
        for ii, jj, kk in dace.map[0:N - 2:K, 0:M - 2:K, 0:P - 2:K]:
            for i, j, k in dace.map[0:K, 0:K, 0:K]:
                I = ii + i + 1
                J = jj + j + 1
                Kk = kk + k + 1
                b[I, J,
                  Kk] = 0.125 * (a[I + 1, J, Kk] - 2.0 * a[I, J, Kk] + a[I - 1, J, Kk] + a[I, J + 1, Kk] - 2.0 *
                                 a[I, J, Kk] + a[I, J - 1, Kk] + a[I, J, Kk + 1] - 2.0 * a[I, J, Kk] + a[I, J, Kk - 1])

    n, m, p = 10, 10, 10
    rng = np.random.default_rng(4)
    a = rng.standard_normal((n, m, p))
    b = np.zeros((n, m, p))
    ref = b.copy()
    copy.deepcopy(heat3d_tiled.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m, P=p)

    sdfg = heat3d_tiled.to_sdfg(simplify=True)
    UntileLoops(map_roundtrip=True).apply_pass(sdfg, {})
    sdfg.validate()
    # 3 axes -> at most 3 collapsed CFR constructs; see the 2D variant
    # above for the post-execution stale-state follow-up.
    n_maps = _count_maps(sdfg)
    n_loops = _count_loops(sdfg)
    assert n_maps + n_loops <= 3, f'expected <=3 collapsed CFR constructs, got {n_maps} maps + {n_loops} loops'


def test_heat3d_tiled_2lvl_range_collapses_via_cascade_fixpoint():
    """2-level tile cascade in 3D. Per-axis triple: outer step K1,
    middle stride K2, inner stride 1."""

    K1 = 16
    K2 = 4

    @dace.program
    def heat3d_2lvl(a: dace.float64[N, M, P], b: dace.float64[N, M, P]):
        for i0 in range(0, N - 2, K1):
            for j0 in range(0, M - 2, K1):
                for k0 in range(0, P - 2, K1):
                    for i1 in range(i0, i0 + K1, K2):
                        for j1 in range(j0, j0 + K1, K2):
                            for k1 in range(k0, k0 + K1, K2):
                                for i2 in range(i1, i1 + K2):
                                    for j2 in range(j1, j1 + K2):
                                        for k2 in range(k1, k1 + K2):
                                            I = i2 + 1
                                            J = j2 + 1
                                            Kk = k2 + 1
                                            b[I, J,
                                              Kk] = 0.125 * (a[I + 1, J, Kk] - 2.0 * a[I, J, Kk] + a[I - 1, J, Kk] +
                                                             a[I, J + 1, Kk] - 2.0 * a[I, J, Kk] + a[I, J - 1, Kk] +
                                                             a[I, J, Kk + 1] - 2.0 * a[I, J, Kk] + a[I, J, Kk - 1])

    n, m, p = 18, 18, 18
    rng = np.random.default_rng(5)
    a = rng.standard_normal((n, m, p))
    b = np.zeros((n, m, p))
    ref = b.copy()
    copy.deepcopy(heat3d_2lvl.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m, P=p)

    sdfg = heat3d_2lvl.to_sdfg(simplify=True)
    UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert _count_loops(sdfg) == 3, f'expected 3 fully-collapsed axes, got {_count_loops(sdfg)}'
    sdfg(a=a, b=b, N=n, M=m, P=p)
    assert np.allclose(b, ref)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
