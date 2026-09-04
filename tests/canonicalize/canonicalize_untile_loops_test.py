# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.untile_loops.UntileLoops`."""
import copy
import inspect

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy

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


def test_untiles_when_outer_stride_is_bare_symbol():
    """``for i in range(0, N, BS): for ii in range(BS): a[i+ii] = ...`` with a
    bare-**symbol** tile ``BS``. A symbol is assumed non-negative by DaCe
    convention (we do not rely on SymPy sign assumptions), so the single-level
    tile collapses to ``for k in range(0, N)`` -- ``end == BS - 1`` folds against
    ``K_expr - 1`` symbolically. Only a unit inner stride is admitted for
    symbolic tiles (a concrete stride cannot be proven to divide a symbol)."""
    BS = dace.symbol('BS')

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, BS):
            for ii in range(BS):
                a[i + ii] = b[i + ii] * 2.0

    sdfg = tiled.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 2
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops_after = _loops(sdfg)
    assert len(loops_after) == 1, f'expected 1 collapsed loop, got {len(loops_after)}'
    assert loops_after[0].loop_variable.startswith('_untile_k_')

    n = 16  # multiple of BS=4 (clean tile)
    rng = np.random.default_rng(7)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    sdfg(a=a, b=b, N=n, BS=4)
    assert np.allclose(a, b * 2.0), f'value mismatch: got {a}'


def test_untiles_when_outer_stride_is_bare_symbol_case_b():
    """Case B with a bare-symbol tile: ``for i in range(0, N, BS):
    for ii in range(i, i + BS): a[ii] = ...`` collapses to a single loop."""
    BS = dace.symbol('BS')

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, BS):
            for ii in range(i, i + BS):
                a[ii] = b[ii] + 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert len(_loops(sdfg)) == 1

    n = 16
    rng = np.random.default_rng(8)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    sdfg(a=a, b=b, N=n, BS=4)
    assert np.allclose(a, b + 1.0)


def test_refuses_when_outer_stride_is_compound_symbolic_expr():
    """A **compound** symbolic expression tile (``M - 2``) is refused: DaCe
    assumes bare symbols are non-negative but does NOT prove the sign of an
    expression, and such an expression is not a plausible tile size."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, M - 2):
            for ii in range(M - 2):
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


@dace.program
def scaled_outer(a: dace.float64[N * N], b: dace.float64[N * N]):
    for i in range(0, N, 4):
        for ii in range(4):
            a[2 * i + ii] = b[2 * i + ii]


@dace.program
def squared_both(a: dace.float64[N * N], b: dace.float64[N * N]):
    for i in range(0, N, 4):
        for ii in range(4):
            a[i * i + ii * ii] = b[i * i + ii * ii]


@pytest.mark.parametrize('program,index', [(scaled_outer, lambda i, ii: 2 * i + ii),
                                           (squared_both, lambda i, ii: i * i + ii * ii)])
def test_refuses_when_iterators_do_not_enter_as_their_sum(program, index):
    """``i`` and ``ii`` co-occurring is not enough -- they must enter as ``i + ii``.

    The rewrite substitutes ``ii -> k - i`` then ``i -> 0``, so ``2*i + ii`` collapses to ``k`` and
    ``i*i + ii*ii`` to ``k*k``. Both mention the two iterators together, so a co-occurrence audit
    admits them; with ``N = 8`` the first writes ``a[0,1,2,3,8,9,10,11]`` but the collapsed loop
    writes ``a[0:8]``.
    """
    sdfg = program.to_sdfg(simplify=True)
    assert UntileLoops().apply_pass(sdfg, {}) is None

    n = 8
    rng = np.random.default_rng(7)
    a = np.zeros(n * n)
    b = rng.standard_normal(n * n)
    ref = a.copy()
    for i in range(0, n, 4):
        for ii in range(4):
            ref[index(i, ii)] = b[index(i, ii)]

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref)


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
    # ExpandNestedSDFGInputs + InlineMultistateSDFG flatten the round-trip
    # NSDFGs, the multi-dim ascent fires, and the fixpoint collapses to
    # <= ``axes`` CFR constructs. The general splice reconnects the parent's
    # pred/succ chain through the spliced body, so no orphan states remain
    # and the result executes bit-exactly.
    n_maps = _count_maps(sdfg)
    n_loops = _count_loops(sdfg)
    assert n_maps + n_loops <= 2, f'expected <=2 collapsed CFR constructs, got {n_maps} maps + {n_loops} loops'
    sdfg(a=a, b=b, N=n, M=m)
    assert np.allclose(b, ref)


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
    # 3 axes -> at most 3 collapsed CFR constructs; executes bit-exactly
    # (the general splice leaves no orphan connective states).
    n_maps = _count_maps(sdfg)
    n_loops = _count_loops(sdfg)
    assert n_maps + n_loops <= 3, f'expected <=3 collapsed CFR constructs, got {n_maps} maps + {n_loops} loops'
    sdfg(a=a, b=b, N=n, M=m, P=p)
    assert np.allclose(b, ref)


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


# ---- Symbolic-tile multi-dim coverage --------------------------------------


def test_jacobi2d_tiled_1lvl_sym_range_collapses_to_2d_nest():
    """1-level 2D tile with a bare-**symbol** tile ``BS`` per axis. The
    fixpoint collapses (ii, i) and (jj, j) in two iterations, leaving a
    single 2D nest -- same as the concrete-``K`` variant but the tile is a
    runtime block-size parameter."""
    BS = dace.symbol('BS')

    @dace.program
    def jacobi2d_tiled(a: dace.float64[N, M], b: dace.float64[N, M]):
        for ii in range(0, N - 2, BS):
            for jj in range(0, M - 2, BS):
                for i in range(BS):
                    for j in range(BS):
                        b[ii + i + 1, jj + j +
                          1] = 0.2 * (a[ii + i + 1, jj + j + 1] + a[ii + i + 1, jj + j] + a[ii + i + 1, jj + j + 2] +
                                      a[ii + i, jj + j + 1] + a[ii + i + 2, jj + j + 1])

    n, m, bs = 10, 10, 4  # N-2 == M-2 == 8 divisible by BS=4 (clean tile)
    rng = np.random.default_rng(20)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = b.copy()
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            ref[i, j] = 0.2 * (a[i, j] + a[i, j - 1] + a[i, j + 1] + a[i - 1, j] + a[i + 1, j])

    sdfg = jacobi2d_tiled.to_sdfg(simplify=True)
    UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert _count_loops(sdfg) == 2, f'expected 2 collapsed loops, got {_count_loops(sdfg)}'
    sdfg(a=a, b=b, N=n, M=m, BS=bs)
    assert np.allclose(b, ref)


def test_heat3d_tiled_1lvl_sym_range_collapses_to_3d_nest():
    """1-level 3D tile with a bare-symbol tile ``BS`` per axis; fixpoint
    collapses all three tile axes to a single 3D nest."""
    BS = dace.symbol('BS')

    @dace.program
    def heat3d_tiled(a: dace.float64[N, M, P], b: dace.float64[N, M, P]):
        for ii in range(0, N - 2, BS):
            for jj in range(0, M - 2, BS):
                for kk in range(0, P - 2, BS):
                    for i in range(BS):
                        for j in range(BS):
                            for k in range(BS):
                                I = ii + i + 1
                                J = jj + j + 1
                                Kk = kk + k + 1
                                b[I, J, Kk] = 0.125 * (a[I + 1, J, Kk] - 2.0 * a[I, J, Kk] + a[I - 1, J, Kk] +
                                                       a[I, J + 1, Kk] - 2.0 * a[I, J, Kk] + a[I, J - 1, Kk] +
                                                       a[I, J, Kk + 1] - 2.0 * a[I, J, Kk] + a[I, J, Kk - 1])

    n, m, p, bs = 10, 10, 10, 4
    rng = np.random.default_rng(21)
    a = rng.standard_normal((n, m, p))
    b = np.zeros((n, m, p))
    ref = b.copy()
    copy.deepcopy(heat3d_tiled.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m, P=p, BS=bs)

    sdfg = heat3d_tiled.to_sdfg(simplify=True)
    UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert _count_loops(sdfg) == 3, f'expected 3 collapsed loops, got {_count_loops(sdfg)}'
    sdfg(a=a, b=b, N=n, M=m, P=p, BS=bs)
    assert np.allclose(b, ref)


def test_symbolic_tile_nonunit_inner_stride_collapses_under_assumption():
    """A symbolic tile with a non-unit inner stride collapses as a cascade rung
    under a recorded ``tile % stride == 0`` divisibility assumption. The stride's
    divisibility into a symbol is unprovable, so rather than refuse, the pass
    admits the rung and records the relation for the terminal
    AssumeSymbolConstraints trap. The source nest already requires it (else its
    own inner tile overshoots). Value-preserving when the tile divides evenly."""
    import sympy
    from dace.transformation.passes.canonicalize.tracked_assumptions import tracked_assumptions
    BS = dace.symbol('BS')

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, BS):
            for ii in range(i, i + BS, 2):
                a[ii] = b[ii]

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res == 1, 'symbolic tile + non-unit inner stride should collapse (under assumption)'
    recorded = tracked_assumptions(sdfg)
    assert any(a == sympy.Eq(sympy.Mod(BS, 2), 0) for a in recorded), \
        f'expected a recorded BS % 2 == 0 divisibility assumption; got {recorded}'
    # No residual tile loop remains for the collapsed axis.
    tile_loops = [
        r for r in sdfg.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable and 'untile' not in r.loop_variable
    ]
    assert not tile_loops, f'tile loops should be collapsed, found {[l.label for l in tile_loops]}'

    # Value-preserving with an even tile (BS | N): copies the even indices.
    sdfg.validate()
    n, bs = 16, 4
    rng = np.random.default_rng(0)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    exp = a.copy()
    for i in range(0, n, bs):
        for ii in range(i, i + bs, 2):
            exp[ii] = b[ii]
    sdfg(a=a, b=b, N=n, BS=bs)
    assert np.allclose(a, exp), f'symbolic strided untile diverged: {a} vs {exp}'


# ============================================================================
# The tiled-stencil corpus family, exactly as the perf corpus measures it
# (:mod:`tests.corpus.tsvc_2_5`): 1-, 2- and 3-level tiles, each in a constant
# and a symbolic-tile variant, on jacobi2d (2 axes) and heat3d (3 axes).
#
# Every case asserts STRUCTURE as well as values. A value-only check passes on
# a kernel whose hand-written tiling survived untouched -- which is the failure
# these tests exist to catch: the tiled form computes the same numbers, it just
# keeps the tile loops that block re-parallelization and re-tiling.
# ============================================================================

#: kernel -> (axes, length symbol, OUTERMOST tile at ``tsvc_2_5.SIZES``). The canonical extent
#: depends on the outermost tile alone: the inner rungs subdivide that same window.
_TILED_FAMILY = {
    'jacobi2d_tiled_const': (2, 'LEN_2D', 8),
    'jacobi2d_tiled_sym': (2, 'LEN_2D', tsvc_2_5.SIZES['T']),
    'jacobi2d_double_tiled_const': (2, 'LEN_2D', 16),
    'jacobi2d_double_tiled_sym': (2, 'LEN_2D', tsvc_2_5.SIZES['T1']),
    'jacobi2d_triple_tiled_const': (2, 'LEN_2D', 16),
    'jacobi2d_triple_tiled_sym': (2, 'LEN_2D', tsvc_2_5.SIZES['T1']),
    'heat3d_tiled_const': (3, 'LEN_3D', 8),
    'heat3d_tiled_sym': (3, 'LEN_3D', tsvc_2_5.SIZES['T']),
    'heat3d_double_tiled_const': (3, 'LEN_3D', 8),
    'heat3d_double_tiled_sym': (3, 'LEN_3D', tsvc_2_5.SIZES['T1']),
}


def _nest_depth(loop) -> int:
    depth = 0
    graph = loop.parent_graph
    while graph is not None:
        if isinstance(graph, LoopRegion):
            depth += 1
        graph = graph.parent_graph
    return depth


def _tile_union_extent(length: int, tile: int):
    """``[start, end)`` the source tile walk covers: ``range(1, L-1-t, t)`` origins each covering
    ``t`` elements, i.e. the interior rounded UP to the next tile boundary above 1."""
    span = length - 2 - tile
    return (1, 1 + tile * ((span + tile - 1) // tile))


def _assert_canonical_nest(sdfg, axes, extent, sizes):
    """The recovered nest is ``axes`` unit-stride loops over the full canonical extent, one per
    level, with no tile-index loop and no remainder guard left anywhere."""
    loops = _loops(sdfg)
    assert len(loops) == axes, f'expected {axes} recovered loops, got {[l.loop_variable for l in loops]}'
    assert sorted(_nest_depth(l) for l in loops) == list(range(axes)), \
        f'recovered loops are not one perfect chain: depths {[_nest_depth(l) for l in loops]}'
    for loop in loops:
        assert loop.loop_variable.startswith('_untile_k_'), f'leftover tile-index loop {loop.loop_variable}'
        stride = int(symbolic.evaluate(loop_analysis.get_loop_stride(loop), sizes))
        assert stride == 1, f'{loop.loop_variable} kept tile stride {stride}'
        start = int(symbolic.evaluate(loop_analysis.get_init_assignment(loop), sizes))
        end = int(symbolic.evaluate(loop_analysis.get_loop_end(loop) + 1, sizes))
        assert (start, end) == extent, f'{loop.loop_variable} runs [{start}, {end}), expected {extent}'
    guards = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    assert not guards, f'remainder guard left behind: {[b.label for b in guards]}'


def _assert_emitted_nest(sdfg, axes):
    """The generated C++ carries exactly ``axes`` loops, every one of them unit stride. This is
    the only place a surviving tile stride is visible once the SDFG claims to be untiled."""
    code = '\n'.join(c.clean_code for c in sdfg.generate_code())
    emitted = [line.strip() for line in code.splitlines() if line.strip().startswith('for (')]
    assert len(emitted) == axes, f'expected {axes} emitted loops, got {len(emitted)}:\n' + '\n'.join(emitted)
    for line in emitted:
        # Last token of the declarator IS the name: the type spelling varies (``auto``,
        # ``int64_t``, ``long long``) and stripping one fixed prefix left the type glued to
        # the name, so the substring below could never match and every loop read as strided.
        var = line.split('(', 1)[1].split('=', 1)[0].split()[-1]
        assert f'{var} += 1)' in line or f'{var} = ({var} + 1))' in line, f'emitted loop kept a tile stride: {line}'


@pytest.mark.parametrize('name', sorted(_TILED_FAMILY))
def test_tiled_stencil_corpus_untiles_to_the_canonical_nest(name):
    """Every member of the tiled-stencil family collapses to its canonical nest, values intact."""
    axes, length_symbol, tile = _TILED_FAMILY[name]
    sizes = tsvc_2_5.SIZES
    program = getattr(tsvc_2_5, name)
    arrays, scalars = tsvc_2_5.make_inputs(program)
    oracle = getattr(tsvc_2_5_numpy, 'ref_' + name)
    pool = {**{n: a.copy() for n, a in arrays.items()}, **scalars, **{s.lower(): v for s, v in sizes.items()}}
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})

    sdfg = program.to_sdfg(simplify=True)
    before = _loops(sdfg)
    assert len(before) > axes, f'precondition: {name} must arrive hand-tiled, got {len(before)} loops'
    # One collapse removes exactly one loop, so the count is the whole cascade, not a partial one.
    assert UntileLoops().apply_pass(sdfg, {}) == len(before) - axes, \
        f'{name}: untile stopped short, {len(_loops(sdfg))} loops left'
    sdfg.validate()
    _assert_canonical_nest(sdfg, axes, _tile_union_extent(sizes[length_symbol], tile), sizes)
    _assert_emitted_nest(sdfg, axes)

    got = {n: a.copy() for n, a in arrays.items()}
    symbols = {s: v for s, v in sizes.items() if s in {str(x) for x in sdfg.free_symbols}}
    sdfg.compile()(**got, **scalars, **symbols)
    for arr in arrays:
        assert np.allclose(got[arr], pool[arr]), f'{name}/{arr} diverges from the numpy oracle'


def _map_dims(sdfg):
    """``(begin, end, step)`` of every Map dimension in the SDFG."""
    return [rng for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) for rng in n.map.range]


def test_map_tiled_3level_untiles_on_the_pass_defaults():
    """A ``dace.map``-tiled 3-level nest is recovered WITHOUT the caller forcing the round trip.

    The matcher only reads LoopRegions, so before the auto-trigger this kernel came out of the
    pass byte-identical -- the hand-written tiling reached codegen and the canonical form was
    never seen. Structure is what proves the recovery: one 2-D iteration space over the full
    extent, unit stride, no tile-strided Map left."""
    M = dace.symbol('M')

    @dace.program
    def map_tiled(a: dace.float64[N, M], b: dace.float64[N, M]):
        for i0, j0 in dace.map[0:N - 2:16, 0:M - 2:16]:
            for i1, j1 in dace.map[i0:i0 + 16:8, j0:j0 + 16:8]:
                for i2, j2 in dace.map[i1:i1 + 8, j1:j1 + 8]:
                    b[i2 + 1, j2 + 1] = 0.2 * (a[i2 + 1, j2 + 1] + a[i2 + 1, j2] + a[i2 + 1, j2 + 2] + a[i2, j2 + 1] +
                                               a[i2 + 2, j2 + 1])

    n, m = 18, 18
    sdfg = map_tiled.to_sdfg(simplify=True)
    assert _count_maps(sdfg) == 3 and _count_loops(sdfg) == 0, 'precondition: three tiled Maps'

    assert UntileLoops().apply_pass(sdfg, {}) == 4, 'both axes must collapse twice'
    sdfg.validate()
    assert _count_loops(sdfg) == 0, 'a Map must not come back as a loop'
    dims = _map_dims(sdfg)
    assert len(dims) == 2, f'expected one 2-D iteration space, got {len(dims)} Map dimensions'
    sizes = {'N': n, 'M': m}
    for begin, end, step in dims:
        assert int(symbolic.evaluate(step, sizes)) == 1, 'recovered Map kept a tile stride'
        assert (int(symbolic.evaluate(begin, sizes)), int(symbolic.evaluate(end + 1, sizes))) == (0, 16), \
            'recovered Map does not cover the canonical extent'

    rng = np.random.default_rng(11)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = np.zeros((n, m))
    for i in range(1, 17):
        for j in range(1, 17):
            ref[i, j] = 0.2 * (a[i, j] + a[i, j - 1] + a[i, j + 1] + a[i - 1, j] + a[i + 1, j])
    sdfg(a=a, b=b, N=n, M=m)
    assert np.allclose(b, ref)


def test_map_roundtrip_declined_when_a_map_would_not_come_back():
    """The auto round trip lowers EVERY Map, so it is taken only when every one of them re-lifts.

    Here a scatter Map sits beside the tiled nest: ``LoopToMap`` cannot re-derive it (the index
    array is not provably a permutation), so taking the trip would trade the tiling for lost
    parallelism. The pass must leave the SDFG alone instead."""

    @dace.program
    def tiled_plus_scatter(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], idx: dace.int64[N]):
        for i0 in dace.map[0:N:8]:
            for i1 in dace.map[i0:i0 + 8]:
                b[i1] = a[i1] * 2.0
        for i in dace.map[0:N]:
            c[idx[i]] = a[i] * 3.0

    sdfg = tiled_plus_scatter.to_sdfg(simplify=True)
    maps_before = _count_maps(sdfg)
    assert UntileLoops().apply_pass(sdfg, {}) is None, 'the declined round trip must not touch the SDFG'
    assert _count_maps(sdfg) == maps_before and _count_loops(sdfg) == 0, 'no Map may be left as a loop'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# -----------------------------------------------------------------------------
# The remainder clamp -- ``min(i + K, <parent limit>)``.
#
# Every hand-tiled stencil writes its inner bound this way, because the last tile would otherwise
# overrun the array. The clamp is not decoration: it changes the collapsed bound. An UNCLAMPED tile
# overshoots its span, so the collapsed loop must round up to the tile boundary; a clamped one
# covers the parent range exactly, and rounding up there walks off the end of the array. That is a
# heap corruption (``free(): invalid size``), not a slow kernel, which is why every case below
# runs the result and compares it.
# -----------------------------------------------------------------------------

TSTEPS = dace.symbol('TSTEPS')


def jacobi_reference(A, B, n, tsteps):
    """The flat, untiled stencil. Vectorised, so the oracle is never the slow part."""
    for _ in range(tsteps):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[:-2, 1:-1] + A[2:, 1:-1])
        A[1:n - 1, 1:n - 1] = B[1:n - 1, 1:n - 1]
    return A, B


def run_against_flat_stencil(prog, n=96, tsteps=3):
    """Canonicalize, run, and compare against the flat nest. Returns the surviving loops."""
    from dace.transformation.passes.canonicalize import finalize, pipeline as canon
    rng = np.random.default_rng(5)
    a0, b0 = rng.random((n, n)), rng.random((n, n))
    want_a, want_b = jacobi_reference(a0.copy(), b0.copy(), n, tsteps)

    sdfg = prog.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='cpu')
    finalize.finalize_for_target(sdfg, 'cpu')
    a, b = a0.copy(), b0.copy()
    sdfg(A=a, B=b, N=n, TSTEPS=tsteps)
    assert np.allclose(a, want_a) and np.allclose(b, want_b), f'{prog.name}: untiled result diverged'
    return [
        r.loop_variable for g in sdfg.all_sdfgs_recursive() for r in g.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion)
    ]


def test_a_clamped_two_level_tile_collapses_to_the_flat_stencil():
    """``jacobi_2d_tile_2lvl_too_big``: W = 1024 is larger than the whole interior at test size."""

    @dace.program
    def two_lvl(A: dace.float64[N, N], B: dace.float64[N, N]):
        W = 1024
        for t in range(TSTEPS):
            for ii in range(1, N - 1, W):
                for jj in range(1, N - 1, W):
                    for i in range(ii, min(ii + W, N - 1)):
                        for j in range(jj, min(jj + W, N - 1)):
                            B[i, j] = 0.2 * (A[i, j] + A[i, j - 1] + A[i, j + 1] + A[i - 1, j] + A[i + 1, j])
            A[1:N - 1, 1:N - 1] = B[1:N - 1, 1:N - 1]

    survivors = run_against_flat_stencil(two_lvl)
    assert len(survivors) == 1, f'only the time loop may survive; got {survivors}'


def test_a_clamped_tile_with_swapped_inner_axes_collapses():
    """``jacobi_2d_tile_swapped_dims``: the point loops run ``j`` then ``i``, against the tiles."""

    @dace.program
    def swapped(A: dace.float64[N, N], B: dace.float64[N, N]):
        W = 64
        for t in range(TSTEPS):
            for ii in range(1, N - 1, W):
                for jj in range(1, N - 1, W):
                    for j in range(jj, min(jj + W, N - 1)):
                        for i in range(ii, min(ii + W, N - 1)):
                            B[i, j] = 0.2 * (A[i, j] + A[i, j - 1] + A[i, j + 1] + A[i - 1, j] + A[i + 1, j])
            A[1:N - 1, 1:N - 1] = B[1:N - 1, 1:N - 1]

    survivors = run_against_flat_stencil(swapped)
    assert len(survivors) == 1, f'only the time loop may survive; got {survivors}'


def test_a_four_level_cascade_of_nested_clamps_collapses_completely():
    """``jacobi_2d_tile_4lvl_silly``: widths 13/7/19/3, where none divides its parent.

    A correctly strip-mined level clamps to EVERY window enclosing it, so its bound is a Min of
    several terms and no single term is the parent's limit -- their Min is. Comparing the terms one
    at a time refuses exactly this, the well-formed nest; comparing them together collapses all ten
    spatial loops and leaves only the time loop.

    Clamped to the array bound alone, these widths let each level run past the tile containing it.
    The nest still covers the flat range, but visits most points three or four times -- a redundant
    cover, not a tiling, on which no untiling is sound and only the body's idempotence would hide
    the difference. That shape is what this kernel used to have.
    """

    @dace.program
    def four_lvl(A: dace.float64[N, N], B: dace.float64[N, N]):
        W1, W2, W3, W4 = 13, 7, 19, 3
        for t in range(TSTEPS):
            for i1 in range(1, N - 1, W1):
                for j1 in range(1, N - 1, W1):
                    for i2 in range(i1, min(i1 + W1, N - 1), W2):
                        for j2 in range(j1, min(j1 + W1, N - 1), W2):
                            for i3 in range(i2, min(i2 + W2, i1 + W1, N - 1), W3):
                                for j3 in range(j2, min(j2 + W2, j1 + W1, N - 1), W3):
                                    for i4 in range(i3, min(i3 + W3, i2 + W2, i1 + W1, N - 1), W4):
                                        for j4 in range(j3, min(j3 + W3, j2 + W2, j1 + W1, N - 1), W4):
                                            for i in range(i4, min(i4 + W4, i3 + W3, i2 + W2, i1 + W1, N - 1)):
                                                for j in range(j4, min(j4 + W4, j3 + W3, j2 + W2, j1 + W1, N - 1)):
                                                    B[i, j] = 0.2 * (A[i, j] + A[i, j - 1] + A[i, j + 1] + A[i - 1, j] +
                                                                     A[i + 1, j])
            A[1:N - 1, 1:N - 1] = B[1:N - 1, 1:N - 1]

    survivors = run_against_flat_stencil(four_lvl)
    assert len(survivors) == 1, f'all ten spatial loops must collapse; got {survivors}'


@pytest.mark.parametrize('n', [17, 30, 64, 97, 128])
def test_the_nested_clamp_cascade_is_a_partition_at_every_size(n):
    """The reason the clamp is written this way rather than by sizing N to divide the widths.

    A divisibility-based fix holds only while N cooperates, and the size is exactly what a fuzzing
    harness varies. The clamp is correct at EVERY N -- checked here on the iteration set itself,
    so the property is pinned independently of what the compiler then does with it.
    """
    from collections import Counter
    w1, w2, w3, w4 = 13, 7, 19, 3
    seen: Counter = Counter()
    for i1 in range(1, n - 1, w1):
        for i2 in range(i1, min(i1 + w1, n - 1), w2):
            for i3 in range(i2, min(i2 + w2, i1 + w1, n - 1), w3):
                for i4 in range(i3, min(i3 + w3, i2 + w2, i1 + w1, n - 1), w4):
                    for i in range(i4, min(i4 + w4, i3 + w3, i2 + w2, i1 + w1, n - 1)):
                        seen[i] += 1
    assert set(seen) == set(range(1, n - 1)), f'N={n}: the tiles do not cover the flat range'
    assert not seen or max(seen.values()) == 1, f'N={n}: some point is visited twice'
