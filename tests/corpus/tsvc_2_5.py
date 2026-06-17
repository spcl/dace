# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC-2.5 extension corpus: ``@dace.program`` kernels.

Each kernel exercises a vectorization failure mode that the standard
TSVC-2 corpus does not cover. Recurring theme: symbolic dependence
vectors (stride / offset / index function) that survive the front end
and require runtime checks, snapshot renames, or guarded lifts to
parallelize. Pluto-class polyhedral tools refuse these because their
dependence is not affine in the integers.

Source contracts:

* Kernel names mirror their `@dace.program` symbol and become the
  per-kernel split file name (matching the ``tsvc_2`` convention).
* Symbolic-stride kernels take an extra ``S`` (or ``ssym``) symbol; the
  caller binds it at run-time, so the SDFG is built once and reused
  across stride values.
* Each kernel pairs with a numpy oracle in ``reference_python``.
"""
from math import sqrt, exp

import dace
import numpy as np

LEN_1D = dace.symbol("LEN_1D")
LEN_2D = dace.symbol("LEN_2D")
LEN_3D = dace.symbol("LEN_3D")
ITERATIONS = dace.symbol("ITERATIONS")
S = dace.symbol("S")  # symbolic stride (TSVC-2 carries one too; we reuse the name)
SSYM = dace.symbol("SSYM")  # symbolic stride for the strided gather/scatter family
K = dace.symbol("K")  # symbolic offset
M = dace.symbol("M")  # quasi-affine `N // M` denominator
T = dace.symbol("T")  # symbolic tile size (single-level tiling)
T1 = dace.symbol("T1")  # symbolic outer tile size (two-level tiling)
T2 = dace.symbol("T2")  # symbolic inner tile size (two-level tiling)
LEN_R7 = dace.symbol("LEN_R7")  # reroll length; bound to a multiple of the 7x reroll factor

# ==========================================================================
#  %A  Symbolic-stride load (gather)
# ==========================================================================


@dace.program
def ext_strided_load_ssym(src: dace.float64[SSYM * LEN_1D], dst: dace.float64[LEN_1D], scale: dace.float64):
    """``dst[i] = src[i * SSYM] * scale`` with ``SSYM`` a runtime symbol.

    The compiler cannot prove the access pattern is contiguous because
    ``SSYM`` is unknown; native auto-vectorizers fall back to scalar
    code unless they emit a runtime stride check + gather intrinsic.
    """
    for i, in dace.map[0:LEN_1D:1]:
        dst[i] = src[i * SSYM] * scale


@dace.program
def ext_strided_load_2(src: dace.float64[2 * LEN_1D], dst: dace.float64[LEN_1D], scale: dace.float64):
    """``dst[i] = src[i * 2] * scale`` -- the constant-stride sibling
    of ``ext_strided_load_ssym``. Most compilers vectorize this via
    ``vpcompressd``-style gathers."""
    for i, in dace.map[0:LEN_1D:1]:
        dst[i] = src[i * 2] * scale


# ==========================================================================
#  %B  Symbolic-stride store (scatter)
# ==========================================================================


@dace.program
def ext_strided_store_ssym(src: dace.float64[LEN_1D], dst: dace.float64[SSYM * LEN_1D], scale: dace.float64):
    """``dst[i * SSYM] = src[i] * scale``. The scatter is potentially
    non-permutation (depends on ``SSYM``); a safe lift requires a
    runtime guard ensuring distinct write indices."""
    for i, in dace.map[0:LEN_1D:1]:
        dst[i * SSYM] = src[i] * scale


@dace.program
def ext_strided_store_2(src: dace.float64[LEN_1D], dst: dace.float64[2 * LEN_1D], scale: dace.float64):
    """``dst[i * 2] = src[i] * scale`` -- constant-stride sibling."""
    for i, in dace.map[0:LEN_1D:1]:
        dst[i * 2] = src[i] * scale


# ==========================================================================
#  %C  Indirect gather + indirect scatter
# ==========================================================================


@dace.program
def ext_gather_load(src: dace.float64[LEN_1D], idx: dace.int64[LEN_1D], dst: dace.float64[LEN_1D], scale: dace.float64):
    """``dst[i] = src[idx[i]] * scale``. The read pattern is fully
    data-dependent; vectorization requires a gather intrinsic."""
    for i, in dace.map[0:LEN_1D:1]:
        dst[i] = src[idx[i]] * scale


@dace.program
def ext_scatter_store(src: dace.float64[LEN_1D], idx: dace.int64[LEN_1D], dst: dace.float64[LEN_1D],
                      scale: dace.float64):
    """``dst[idx[i]] = src[i] * scale``. Safe parallelization requires
    proving that ``idx`` is a permutation -- the ScatterToGuardedMaps
    pass emits a sort+duplicate-count check that lets the lift fire
    only when the runtime indices are distinct."""
    for i, in dace.map[0:LEN_1D:1]:
        dst[idx[i]] = src[i] * scale


# ==========================================================================
#  %D  Quasi-affine offsets (//, floor-div, modular wraparound)
# ==========================================================================


@dace.program
def ext_floordiv_offset(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """``a[i] = a[i + LEN_1D // 2] + b[i]`` -- forward read across the
    array midpoint. Polyhedral dependence analysis fails because the
    offset is a floor-div of the trip count, not an affine integer
    constant."""
    for i in range(LEN_1D // 2):
        a[i] = a[i + LEN_1D // 2] + b[i]


@dace.program
def ext_floordiv_offset_m(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """Generalised ``a[i] = a[i + LEN_1D // M] + b[i]`` with ``M`` a
    runtime symbol. The offset is a quasi-affine function of two
    symbols and is the canonical Pluto-defeat case."""
    for i in range(LEN_1D // M):
        a[i] = a[i + LEN_1D // M] + b[i]


@dace.program
def ext_modular_wrap(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """``a[(i + K) % LEN_1D] = b[i]`` -- modulo wraparound write. The
    write index is data-dependent through ``K``; the canonicalize
    pipeline's ``peel_limit`` knob unlocks parallelization by peeling
    the boundary iteration."""
    for i in range(LEN_1D):
        a[(i + K) % LEN_1D] = b[i]


# ==========================================================================
#  %E  Read-ahead WAR (anti-dep with symbolic offset)
# ==========================================================================


@dace.program
def ext_war_unit(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """TSVC ``s121`` shape: ``a[i] = a[i+1] + b[i]``. ``LoopToMap`` refuses
    without ``break_anti_dependence=True``; the canonicalize knob
    snapshot-renames ``a`` so the loop lifts."""
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def ext_war_sym(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """Symbolic-offset WAR: ``a[i] = a[i + K] + b[i]`` with ``K`` runtime.
    Same snapshot-rename trick lifts the loop when ``K > 0``; ``K`` may
    require a runtime guard to prove non-negativity."""
    for i in range(LEN_1D - K):
        a[i] = a[i + K] + b[i]


# ==========================================================================
#  %F  Boundary-conflict peeling (multi-front)
# ==========================================================================


@dace.program
def ext_peel_multi_back(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """Two tail iterations write conflicting elements; peeling them off
    leaves a disjoint-write remainder that maps cleanly. Anchors the
    ``peel_limit >= 2`` requirement."""
    for i in range(LEN_1D):
        a[i] = b[i] * 2.0
        if i == LEN_1D - 1:
            a[LEN_1D - 2] = a[LEN_1D - 2] + 1.0
        elif i == LEN_1D - 2:
            a[LEN_1D - 3] = a[LEN_1D - 3] + 1.0


# ==========================================================================
#  %G  Multi-dim symbolic tile
# ==========================================================================


@dace.program
def ext_tile_2d_sym(a: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D, LEN_2D]):
    """Two-axis tile with symbolic tile size ``S``. The untile pass
    must detect the (outer_i, inner_i) and (outer_j, inner_j) tile
    pairs across the multi-dim ascent. Requires both the cascade and
    the multi-dim ascent extensions."""
    for ti in range(0, LEN_2D, S):
        for tj in range(0, LEN_2D, S):
            for i in range(ti, ti + S):
                for j in range(tj, tj + S):
                    b[i, j] = a[i, j] * 2.0


# ==========================================================================
#  %H  TSVC-named symbolic-step variants (parallel-naming with tsvc_2/)
# ==========================================================================
#
# Each kernel mirrors a specific TSVC-2 kernel's loop shape but takes a
# symbolic offset / stride. The naming convention matches ``tsvc_2/``'s
# ``s<id>`` prefix so a join across the kernel tables identifies the
# symbolic-variant of each TSVC-2 kernel directly.


@dace.program
def s121_sym_k(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """TSVC ``s121`` with symbolic offset ``K``:
    ``a[i] = a[i + K] + b[i]``. The original ``s121`` uses ``K = 1``
    (a unit-offset read-ahead WAR); here ``K`` is a runtime symbol, so
    the snapshot-rename guard in ``break_anti_dependence`` must add a
    ``K > 0`` runtime check before lifting to a Map.
    """
    for i in range(LEN_1D - K):
        a[i] = a[i + K] + b[i]


@dace.program
def s4113_ssym(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], ip: dace.int64[LEN_1D]):
    """TSVC ``s4113`` with symbolic stride on the index array:
    ``a[ip[i * SSYM]] = b[ip[i * SSYM]] + c[i]``. The original
    ``s4113`` reads ``ip[i]`` (unit stride). Here the gather index
    is itself strided by ``SSYM``, breaking the ``ip`` permutation
    proof at any constant offset and exposing the gather/scatter
    runtime check.
    """
    for i in range(LEN_1D // SSYM):
        a[ip[i * SSYM]] = b[ip[i * SSYM]] + c[i]


@dace.program
def vas_ssym(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int64[LEN_1D]):
    """TSVC ``vas`` with symbolic-stride scatter:
    ``a[ip[i * SSYM]] = b[i]``. Pure write-scatter form. Symbolic
    stride means even known-permutation ``ip`` arrays no longer prove
    distinct writes statically; the
    ``ScatterToGuardedMaps`` sort+dup-count guard is required for the
    lift.
    """
    for i in range(LEN_1D // SSYM):
        a[ip[i * SSYM]] = b[i]


# ==========================================================================
#  %I  Loop-fission family (sequential `for` with multiple bodies)
# ==========================================================================
#
# These kernels exercise the LoopFission canonicalize pass: a body that
# either pairs two independent statements (so they can vectorize
# separately if reuse pressure forces a split) or one carried-dep
# statement next to an independent statement (where fission MUST fire
# for the independent body to vectorize).


@dace.program
def fission_indep_2body(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], x: dace.float64[LEN_1D],
                        y: dace.float64[LEN_1D], z: dace.float64[LEN_1D]):
    """Two independent writes sharing three reads. Either fused or
    fissioned bodies are correct; fission gives both bodies independent
    vector loops if register / reuse pressure forces the split."""
    for i in range(LEN_1D):
        a[i] = x[i] * y[i] + z[i]
        b[i] = x[i] - y[i] * z[i]


@dace.program
def fission_dep_then_indep(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], x: dace.float64[LEN_1D],
                           y: dace.float64[LEN_1D]):
    """Body A carries a unit-offset dependence (prefix-sum on ``a``),
    body B is independent. LoopFission must fire so that the
    independent body vectorizes while the prefix-sum body stays scalar
    (or lifts to a Scan)."""
    a[0] = x[0]
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + x[i]
        b[i] = y[i] * 2.0


@dace.program
def fission_dep_const_offset(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], x: dace.float64[LEN_1D],
                             y: dace.float64[LEN_1D], z: dace.float64[LEN_1D]):
    """Body A carries a constant-offset (stride 2) dependence on ``a``,
    body B is independent. After fission the independent body
    vectorizes; the carried-dep body needs offset-2 software pipelining
    or stays scalar."""
    a[0] = x[0]
    a[1] = x[1]
    for i in range(2, LEN_1D):
        a[i] = a[i - 2] + x[i]
        b[i] = y[i] * z[i]


@dace.program
def fission_dep_sym_offset(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], x: dace.float64[LEN_1D],
                           y: dace.float64[LEN_1D], z: dace.float64[LEN_1D]):
    """Same shape as :func:`fission_dep_const_offset` but the offset is
    the runtime symbol ``K``. Caller initializes ``a[0..K-1]`` before
    invocation."""
    for i in range(K, LEN_1D):
        a[i] = a[i - K] + x[i]
        b[i] = y[i] * z[i]


# ==========================================================================
#  %J  Already-tiled stencils (constant + symbolic tile size)
# ==========================================================================
#
# Outer tile loops + inner stencil. Both the constant- and the
# symbolic-tile-size variants are written explicitly so the
# vectorizer's tile-untile + multi-dim ascent passes have stable anchor
# kernels for benchmarking.


@dace.program
def jacobi2d_tiled_const(a: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D, LEN_2D]):
    """2D Jacobi 5-point stencil pre-tiled with constant tile size 64.

    Outer ``ii``/``jj`` walk tile origins, inner ``i``/``j`` walk the
    in-tile coordinates.
    """
    for ii in range(1, LEN_2D - 1 - 64, 64):
        for jj in range(1, LEN_2D - 1 - 64, 64):
            for i in range(ii, ii + 64):
                for j in range(jj, jj + 64):
                    b[i, j] = 0.2 * (a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


@dace.program
def jacobi2d_tiled_sym(a: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D, LEN_2D]):
    """2D Jacobi 5-point stencil pre-tiled with symbolic tile size
    ``T``. Same body as :func:`jacobi2d_tiled_const` with the literal
    ``64`` replaced by the runtime symbol ``T``."""
    for ii in range(1, LEN_2D - 1 - T, T):
        for jj in range(1, LEN_2D - 1 - T, T):
            for i in range(ii, ii + T):
                for j in range(jj, jj + T):
                    b[i, j] = 0.2 * (a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


@dace.program
def jacobi2d_double_tiled_const(a: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D, LEN_2D]):
    """2D Jacobi 5-point stencil with two levels of constant tiling
    (outer tile 64, inner tile 8). Anchors the two-level untile pass."""
    for ii in range(1, LEN_2D - 1 - 64, 64):
        for jj in range(1, LEN_2D - 1 - 64, 64):
            for iii in range(ii, ii + 64, 8):
                for jjj in range(jj, jj + 64, 8):
                    for i in range(iii, iii + 8):
                        for j in range(jjj, jjj + 8):
                            b[i, j] = 0.2 * (a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


@dace.program
def jacobi2d_double_tiled_sym(a: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D, LEN_2D]):
    """Two-level tiling with symbolic outer tile ``T1`` and symbolic
    inner tile ``T2``."""
    for ii in range(1, LEN_2D - 1 - T1, T1):
        for jj in range(1, LEN_2D - 1 - T1, T1):
            for iii in range(ii, ii + T1, T2):
                for jjj in range(jj, jj + T1, T2):
                    for i in range(iii, iii + T2):
                        for j in range(jjj, jjj + T2):
                            b[i, j] = 0.2 * (a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


@dace.program
def heat3d_tiled_const(a: dace.float64[LEN_3D, LEN_3D, LEN_3D], b: dace.float64[LEN_3D, LEN_3D, LEN_3D]):
    """3D 7-point heat stencil pre-tiled with constant tile size 8 on
    all three axes."""
    for kk in range(1, LEN_3D - 1 - 8, 8):
        for jj in range(1, LEN_3D - 1 - 8, 8):
            for ii in range(1, LEN_3D - 1 - 8, 8):
                for k in range(kk, kk + 8):
                    for j in range(jj, jj + 8):
                        for i in range(ii, ii + 8):
                            b[k, j, i] = 0.125 * (a[k + 1, j, i] - 2.0 * a[k, j, i] + a[k - 1, j, i]) + \
                                         0.125 * (a[k, j + 1, i] - 2.0 * a[k, j, i] + a[k, j - 1, i]) + \
                                         0.125 * (a[k, j, i + 1] - 2.0 * a[k, j, i] + a[k, j, i - 1]) + a[k, j, i]


@dace.program
def heat3d_tiled_sym(a: dace.float64[LEN_3D, LEN_3D, LEN_3D], b: dace.float64[LEN_3D, LEN_3D, LEN_3D]):
    """3D 7-point heat stencil pre-tiled with symbolic tile size ``T``
    on all three axes."""
    for kk in range(1, LEN_3D - 1 - T, T):
        for jj in range(1, LEN_3D - 1 - T, T):
            for ii in range(1, LEN_3D - 1 - T, T):
                for k in range(kk, kk + T):
                    for j in range(jj, jj + T):
                        for i in range(ii, ii + T):
                            b[k, j, i] = 0.125 * (a[k + 1, j, i] - 2.0 * a[k, j, i] + a[k - 1, j, i]) + \
                                         0.125 * (a[k, j + 1, i] - 2.0 * a[k, j, i] + a[k, j - 1, i]) + \
                                         0.125 * (a[k, j, i + 1] - 2.0 * a[k, j, i] + a[k, j, i - 1]) + a[k, j, i]


# ==========================================================================
#  %K  ECRAD-style clamped reduction
# ==========================================================================


@dace.program
def ecrad_clamped_reduction(x: dace.float64[LEN_1D], y: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
                            out: dace.float64[LEN_1D]):
    """ECRAD-shaped per-element clamped transmittance:
    ``out[i] = clamp(exp(-sqrt(max(x*x + y*y, 1e-12)) * d), 0, 1)``.

    Two ``max``/``min`` clamps + an ``exp`` + a ``sqrt`` in the body
    stress the transcendental-clamp recognizer and the SLEEF / libmvec
    intrinsic lowerings.
    """
    for i in dace.map[0:LEN_1D]:
        k = sqrt(max(x[i] * x[i] + y[i] * y[i], 1e-12))
        e = exp(-k * d[i])
        out[i] = max(0.0, min(e, 1.0))


# ==========================================================================
#  %L  Conditional masked stores
# ==========================================================================


@dace.program
def masked_store_const(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], mask: dace.int64[LEN_1D]):
    """Predicated store with an integer mask: ``if mask[i] > 0: a[i] = b[i]``.
    Requires masked-store / blend-store vector intrinsics."""
    for i in dace.map[0:LEN_1D]:
        if mask[i] > 0:
            a[i] = b[i]


@dace.program
def masked_store_sym(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], threshold_data: dace.float64[LEN_1D]):
    """Predicated store keyed on a comparison against the symbolic
    threshold ``K`` (treated as a double scalar): ``if threshold_data[i]
    > K: a[i] = b[i]``."""
    for i in dace.map[0:LEN_1D]:
        if threshold_data[i] > K:
            a[i] = b[i]


# ==========================================================================
#  %M  Quasi-affine subscript ranges (even/odd, pairwise, mod-K, floor-div)
# ==========================================================================
#
# These kernels exercise quasi-affine subscript and iteration patterns
# that polyhedral analysis still struggles with -- a striding subset
# (even-only / odd-only), pairwise reads at ``2*i`` and ``2*i+1``, a
# data-flow split by ``i % K``, and a write-conflict scatter via
# ``b[i // 2] += a[i]`` where pairs of source iterations land in the
# same output cell.


@dace.program
def quasi_affine_reduce_even(a: dace.float64[LEN_1D], out: dace.float64[1]):
    """Reduce only the even-indexed entries: ``sum(a[i] for i in
    range(0, LEN_1D, 2))``. The stride-2 access subset survives the
    front end as ``range(0, N, 2)``; the auto-vectorizer must spot
    that the iteration space is contiguous after a /2 strength-
    reduction (and a contig-load proof on ``a[2*i]``)."""
    out[0] = 0.0
    for i in range(0, LEN_1D, 2):
        out[0] = out[0] + a[i]


@dace.program
def quasi_affine_reduce_odd(a: dace.float64[LEN_1D], out: dace.float64[1]):
    """Sibling of :func:`quasi_affine_reduce_even` with a non-zero
    base: ``sum(a[i] for i in range(1, LEN_1D, 2))``. The non-zero
    starting offset is the extra hop the polyhedral check has to
    canonicalize."""
    out[0] = 0.0
    for i in range(1, LEN_1D, 2):
        out[0] = out[0] + a[i]


@dace.program
def quasi_affine_pairwise_sum(a: dace.float64[2 * LEN_1D], b: dace.float64[LEN_1D]):
    """``b[i] = a[2*i] + a[2*i + 1]`` -- two quasi-affine reads per
    iteration. The compiler should recognise this as a half-stride
    gather + a shuffle (or a deinterleave load), but in practice both
    Clang and GCC frequently scalarise the ``a[2*i + 1]`` read."""
    for i in dace.map[0:LEN_1D]:
        b[i] = a[2 * i] + a[2 * i + 1]


@dace.program
def quasi_affine_mod_k_stripe(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    """Every ``K``-th iteration takes a different branch:
    ``a[i] = b[i] * 2.0 if i % K == 0 else c[i]``. The branch
    predicate is a quasi-affine function of ``i`` and a symbolic
    divisor; the masked-store optimization has to either peel a
    finite period or emit two predicated stores per vector chunk."""
    for i in dace.map[0:LEN_1D]:
        if (i % K) == 0:
            a[i] = b[i] * 2.0
        else:
            a[i] = c[i]


@dace.program
def quasi_affine_floor_div_scatter(a: dace.float64[2 * LEN_1D], b: dace.float64[LEN_1D]):
    """``b[i // 2] += a[i]`` -- write-conflict scatter where pairs of
    source iterations (``i, i+1``) land in the same output cell. This
    pattern is genuinely sequential under naive vectorization (it has
    a length-2 reduction stripe) and must lower to either a pairwise
    horizontal add or a sequential loop."""
    for i in range(2 * LEN_1D):
        b[i // 2] = b[i // 2] + a[i]


# ==========================================================================
#  %N  Wavefront / loop-skew (2D anti-diagonal parallelism)
# ==========================================================================
#
# A perfectly-nested 2D update whose body reads the left (``a[i, j-1]``)
# and top (``a[i-1, j]``) neighbours carries dependence vectors ``(0, 1)``
# and ``(1, 0)``. Neither loop is parallel on its own, but the
# anti-diagonal ``i + j = const`` is: the ``WavefrontSkew`` canonicalize
# pass applies the classical ``(i, j) -> (i + j, j)`` skew so the inner
# loop becomes a parallel Map. The base TSVC kernel is ``s2111``.


@dace.program
def wavefront2d(a: dace.float64[LEN_2D, LEN_2D]):
    """2D in-place relaxation with left + top + corner reads:
    ``a[i, j] = 0.25 * (a[i, j] + a[i-1, j] + a[i, j-1] + a[i-1, j-1])``.

    Dependence vectors ``(0, 1)``, ``(1, 0)``, ``(1, 1)`` make both loops
    sequential as written; only the ``i + j`` anti-diagonal is parallel,
    so ``WavefrontSkew`` must skew before ``LoopToMap`` can fire."""
    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            a[i, j] = 0.25 * (a[i, j] + a[i - 1, j] + a[i, j - 1] + a[i - 1, j - 1])


# ==========================================================================
#  %O  Early-exit / find-first (break loops)
# ==========================================================================
#
# ``for i: ... if cond(i): break; ...`` loops lower to a sequential scan
# the ``EarlyExitToFindIndex`` pass rewrites into a parallel find-first
# reduction plus body Maps clipped to the discovered bound. Base TSVC
# kernels: ``s481`` (guard before body), ``s482`` (guard after body),
# ``s332`` (find-first-above-threshold with index/value capture).


@dace.program
def ext_break_find_first(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D],
                         d: dace.float64[LEN_1D]):
    """TSVC ``s481``: guard checked *before* the body. ``if d[i] < 0: break``
    then ``a[i] = a[i] + b[i] * c[i]``. The break bound is data-dependent
    on ``d``; the lift needs a find-first ``min`` reduction over
    ``{i : d[i] < 0}`` before the body can run as a clipped parallel Map."""
    for i in range(LEN_1D):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]


@dace.program
def ext_break_post_body(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    """TSVC ``s482``: body runs *before* the guard. ``a[i] = a[i] + b[i]*c[i]``
    then ``if c[i] > b[i]: break``. The breaking iteration's write is
    retained, so the find-first bound is inclusive -- a different clip
    than :func:`ext_break_find_first`."""
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break


@dace.program
def ext_break_capture(a: dace.float64[LEN_1D], out_index: dace.int64[1], out_value: dace.float64[1]):
    """TSVC ``s332`` with a symbolic threshold ``K`` (bound as a double):
    find the first ``i`` with ``a[i] > K``, capture its index and value,
    and break. The scalar rebind at the exit edge is what
    ``EarlyExitToFindIndex`` must reconstruct as an argmin-of-index."""
    out_index[0] = -1
    out_value[0] = -1.0
    for i in range(LEN_1D):
        if a[i] > K:
            out_index[0] = i
            out_value[0] = a[i]
            break


# ==========================================================================
#  %P  Conditional reduction (predicated accumulate)
# ==========================================================================
#
# ``if cond(i): acc = acc OP expr`` sits inside a ConditionalBlock, so the
# state-level ``AugAssignToWCR`` matcher cannot reach it and the carry on
# ``acc`` blocks ``LoopToMap``. ``LoopToConditionalReduce`` masks the
# addend with the OP identity on the false branch so the loop lifts to a
# parallel Map with a WCR-on-scalar reduction. Base TSVC kernel: ``s3111``.


@dace.program
def cond_reduce_sum(a: dace.float64[LEN_1D], out: dace.float64[1]):
    """TSVC ``s3111``: ``if a[i] > 0: out += a[i]``. Conditional ``+=``
    accumulator; the false branch contributes the additive identity 0."""
    out[0] = 0.0
    for i in range(LEN_1D):
        if a[i] > 0.0:
            out[0] = out[0] + a[i]


@dace.program
def cond_reduce_sym(a: dace.float64[LEN_1D], out: dace.float64[1]):
    """Symbolic-threshold sibling of :func:`cond_reduce_sum`:
    ``if a[i] > K: out += a[i]`` with ``K`` bound as a double. The
    predicate's symbolic comparison forces the mask to be computed at
    runtime before the WCR reduction."""
    out[0] = 0.0
    for i in range(LEN_1D):
        if a[i] > K:
            out[0] = out[0] + a[i]


# ==========================================================================
#  %Q  Induction-variable closed form (scalar evolution)
# ==========================================================================
#
# ``acc = acc OP const`` over ``N`` iterations is a scalar recurrence with
# a closed form (Aho/Lam/Sethi/Ullman Ch. 9.6, LLVM ``IndVarSimplify``).
# ``InductionVariableSubstitution`` collapses the ``O(N)`` loop to ``O(1)``
# straight-line code. The addend/factor must be a literal constant (not a
# per-element read) for the closed form to exist.


@dace.program
def iv_additive(out: dace.float64[1]):
    """Additive induction variable: ``s = 0; for i in range(LEN_1D): s += 1.5``.
    Closed form ``s = 1.5 * LEN_1D``. The trip count is the symbol
    ``LEN_1D``; there is no per-element data, so the loop is a pure
    recurrence the substitution eliminates."""
    s = 0.0
    for i in range(LEN_1D):
        s = s + 1.5
    out[0] = s


@dace.program
def iv_multiplicative(out: dace.float64[1]):
    """Multiplicative induction variable: ``s = 1; for i: s *= 0.99``.
    Closed form ``s = 0.99 ** LEN_1D`` -- the geometric-product case that
    distinguishes scalar evolution from a plain reduction."""
    s = 1.0
    for i in range(LEN_1D):
        s = s * 0.99
    out[0] = s


# ==========================================================================
#  %R  Argmax / argmin value reduction (conditional carry -> Reduce)
# ==========================================================================
#
# ``x = a[0]; for i: if a[i] OP x: x = a[i]`` is a max/min reduction hidden
# behind a conditional scalar carry. ``ArgMaxLift`` recognises the chain
# and replaces the loop with a ``Reduce`` libnode (``Max`` for ``>``/``>=``,
# ``Min`` for ``<``/``<=``). Base TSVC kernels: ``s314`` / ``s316``.


@dace.program
def argmax_value(a: dace.float64[LEN_1D], out: dace.float64[1]):
    """TSVC ``s314``: running maximum carried in a scalar.
    ``x = a[0]; for i in range(1, LEN_1D): if a[i] > x: x = a[i]``.
    ``ArgMaxLift`` rewrites this to ``Reduce(Max, a)``."""
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] > x:
            x = a[i]
    out[0] = x


@dace.program
def argmin_value(a: dace.float64[LEN_1D], out: dace.float64[1]):
    """TSVC ``s316``: running minimum sibling of :func:`argmax_value`.
    ``x = a[0]; for i: if a[i] < x: x = a[i]`` -> ``Reduce(Min, a)``."""
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] < x:
            x = a[i]
    out[0] = x


# ==========================================================================
#  %S  Negative-stride loop + manually-unrolled lane chain
# ==========================================================================
#
# Two normalization anchors that block ``LoopToMap`` until rewritten:
# ``NormalizeNegativeStride`` flips a literal negative-stride loop to
# positive form (base TSVC ``s112``), and ``RerollUnrolledLoops`` collapses
# a hand-unrolled step-``m`` lane chain back to a unit-step loop (base
# TSVC ``s351``).


@dace.program
def neg_stride_rev(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """Reverse-iteration write with no carried dependence:
    ``for i in range(LEN_1D - 1, -1, -1): a[i] = b[i] + 1``. Parallel in
    principle, but the negative literal stride defeats ``LoopToMap``'s
    affine-subset classifier until ``NormalizeNegativeStride`` rewrites it
    to positive form."""
    for i in range(LEN_1D - 1, -1, -1):
        a[i] = b[i] + 1.0


@dace.program
def reroll_saxpy7(a: dace.float64[LEN_R7], b: dace.float64[LEN_R7]):
    """TSVC ``s351``: a saxpy hand-unrolled 7x. Seven structurally-identical
    lanes at offsets ``{0..6}`` over a step-7 loop look like one strided
    ``7*i + k`` access that blocks ``LoopToMap``; ``RerollUnrolledLoops``
    re-rolls to a unit-step loop first. The unroll factor is deliberately
    a **prime** (7) so it cannot coincide with any vector width -- the
    lane chain never accidentally tiles a SIMD register, so the reroll is
    genuinely required rather than a lucky alignment. Length is its own
    symbol ``LEN_R7`` so it can be bound to a multiple of 7."""
    for i in range(0, LEN_R7 - 6, 7):
        a[i] = a[i] + b[i] * 2.0
        a[i + 1] = a[i + 1] + b[i + 1] * 2.0
        a[i + 2] = a[i + 2] + b[i + 2] * 2.0
        a[i + 3] = a[i + 3] + b[i + 3] * 2.0
        a[i + 4] = a[i + 4] + b[i + 4] * 2.0
        a[i + 5] = a[i + 5] + b[i + 5] * 2.0
        a[i + 6] = a[i + 6] + b[i + 6] * 2.0


# ==========================================================================
#  %T  Strided / multiple scans (prefix recurrences -> Scan libnodes)
# ==========================================================================
#
# A prefix recurrence ``a[i] = a[i - stride] OP x[i]`` is the textbook
# ``LoopToScan`` target. Two extensions force the pipeline to emit *more
# than one* Scan libnode: a stride > 1 splits the array into independent
# per-residue-class subsequences (one Scan each), and a body with two
# distinct recurrences needs one Scan per carry. Base TSVC scan kernels
# (``s242`` / ``s1221`` / ``s221``) are all single unit-stride scans.


@dace.program
def scan_strided_2(a: dace.float64[LEN_1D], x: dace.float64[LEN_1D]):
    """Stride-2 prefix sum: ``a[i] = a[i-2] + x[i]``. The even- and
    odd-indexed subsequences are two INDEPENDENT prefix sums, so
    ``LoopToScan`` must emit two Scan libnodes (one per residue class
    mod 2) rather than one. Caller initializes ``a[0]`` and ``a[1]``."""
    for i in range(2, LEN_1D):
        a[i] = a[i - 2] + x[i]


@dace.program
def scan_strided_sym(a: dace.float64[LEN_1D], x: dace.float64[LEN_1D]):
    """Symbolic-stride prefix sum: ``a[i] = a[i-K] + x[i]``. Decomposes
    into ``K`` independent prefix sums (one per residue class mod ``K``),
    so the Scan count is a runtime symbol -- the pipeline lifts it to a
    single stride-``K`` vector Scan. Caller initializes ``a[0..K-1]``."""
    for i in range(K, LEN_1D):
        a[i] = a[i - K] + x[i]


@dace.program
def scan_multi_carry(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], x: dace.float64[LEN_1D],
                     y: dace.float64[LEN_1D]):
    """Two distinct unit-stride recurrences in one loop body: an additive
    scan on ``a`` and a multiplicative scan on ``b``. ``LoopToScan`` must
    emit two Scan libnodes with different operators (Add and Mul) from the
    same loop. Caller initializes ``a[0]`` and ``b[0]``."""
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + x[i]
        b[i] = b[i - 1] * y[i]


# ==========================================================================
#  %U  Canonicalize unit-test gap kernels
# ==========================================================================
#
# Patterns drawn from the DaCe canonicalize unit tests that the families
# above do not isolate: a guarded prefix scan, many parallel scan carries
# (cloudsc pfsqrf), argmax with index capture (s315), an indirect-gather
# manual unroll (s353), the tridiagonal Thomas two-sweep solve, an
# outer-parallel/inner-carried reduction, and a loop-invariant config-flag
# branch select.


@dace.program
def scan_conditional(out: dace.float64[LEN_1D], delta: dace.float64[LEN_1D], mask: dace.int64[LEN_1D]):
    """Masked prefix scan: the running sum advances only where ``mask[i]``
    is set, otherwise it holds. ``LoopToScan`` must descend into the
    ConditionalBlock and treat the false branch as the additive identity.
    Caller seeds ``out[0]``."""
    for i in range(1, LEN_1D):
        if mask[i] > 0:
            out[i] = out[i - 1] + delta[i]
        else:
            out[i] = out[i - 1]


@dace.program
def scan_multi_5carry(acc: dace.float64[5, LEN_1D], delta: dace.float64[5, LEN_1D]):
    """Five INDEPENDENT prefix sums carried in one loop body (the cloudsc
    ``pfsqrf`` shape): ``acc[r, i] = acc[r, i-1] + delta[r, i]`` for
    ``r = 0..4``. ``LoopToScan`` must match all five carries and emit five
    Scan libnodes (or one vectorized row-Scan). Caller seeds ``acc[:, 0]``."""
    for i in range(1, LEN_1D):
        acc[0, i] = acc[0, i - 1] + delta[0, i]
        acc[1, i] = acc[1, i - 1] + delta[1, i]
        acc[2, i] = acc[2, i - 1] + delta[2, i]
        acc[3, i] = acc[3, i - 1] + delta[3, i]
        acc[4, i] = acc[4, i - 1] + delta[4, i]


@dace.program
def argmax_with_index(a: dace.float64[LEN_1D], out_value: dace.float64[1], out_index: dace.int64[1]):
    """TSVC ``s315``: running maximum carrying BOTH the value and its
    index. ``x = a[0]; idx = 0; for i: if a[i] > x: x = a[i]; idx = i``.
    The two-accumulator conditional (value + index) is the ``ArgMaxLift``
    index-capture variant that value-only :func:`argmax_value` does not
    exercise."""
    x = a[0]
    idx = 0
    for i in range(1, LEN_1D):
        if a[i] > x:
            x = a[i]
            idx = i
    out_value[0] = x
    out_index[0] = idx


@dace.program
def reroll_gather(a: dace.float64[LEN_R7], b: dace.float64[LEN_R7], ip: dace.int64[LEN_R7]):
    """TSVC ``s353``: a saxpy hand-unrolled 7x whose source is an indirect
    gather ``b[ip[i+k]]``. ``RerollUnrolledLoops`` collapses the seven
    lanes to a unit-step loop; ``LoopToMap`` then needs the data-dependent
    gather handled. The gather variant of :func:`reroll_saxpy7`. Length is
    its own symbol ``LEN_R7`` so it can be bound to a multiple of 7."""
    for i in range(0, LEN_R7 - 6, 7):
        a[i] = a[i] + b[ip[i]] * 2.0
        a[i + 1] = a[i + 1] + b[ip[i + 1]] * 2.0
        a[i + 2] = a[i + 2] + b[ip[i + 2]] * 2.0
        a[i + 3] = a[i + 3] + b[ip[i + 3]] * 2.0
        a[i + 4] = a[i + 4] + b[ip[i + 4]] * 2.0
        a[i + 5] = a[i + 5] + b[ip[i + 5]] * 2.0
        a[i + 6] = a[i + 6] + b[ip[i + 6]] * 2.0


@dace.program
def thomas_solve(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
                 x: dace.float64[LEN_1D]):
    """Tridiagonal Thomas algorithm: a forward elimination sweep followed
    by a backward substitution sweep on the same axis -- two sequential
    recurrences, the second descending and reading the first's results.
    ``a`` / ``b`` / ``c`` are the sub / main / super diagonals (``c``,
    ``d`` are overwritten as scratch), ``d`` the RHS, ``x`` the solution.
    No single-direction scan covers the reverse second sweep."""
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]
    for i in range(1, LEN_1D):
        m = b[i] - a[i] * c[i - 1]
        c[i] = c[i] / m
        d[i] = (d[i] - a[i] * d[i - 1]) / m
    x[LEN_1D - 1] = d[LEN_1D - 1]
    for i in range(LEN_1D - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]


@dace.program
def reduce_inner_carry(a: dace.float64[LEN_2D, LEN_2D], out: dace.float64[LEN_2D]):
    """Outer loop is parallel over independent rows; the inner loop
    carries a scalar reduction: ``out[i] = sum_j a[i, j]``. The outer
    ``i`` lifts to a Map while the inner ``j`` stays a sequential
    reduction (or a per-row ``Reduce``). Distinct from the flat
    :func:`cond_reduce_sum` scalar accumulators."""
    for i in range(LEN_2D):
        s = 0.0
        for j in range(LEN_2D):
            s = s + a[i, j]
        out[i] = s


@dace.program
def config_select_branch(out_a: dace.float64[LEN_1D], out_b: dace.float64[LEN_1D], src: dace.float64[LEN_1D]):
    """Loop-invariant config flag ``K`` selects which output array each
    iteration writes (incompatible writes to two distinct arrays):
    ``if K > 0: out_a[i] = src[i]*2 else: out_b[i] = src[i]+1``.
    ``MoveLoopInvariantIfUp`` hoists the ``K``-guard out of the loop,
    splitting it into two clean parallel Maps. ``K`` is bound at call
    time."""
    for i in range(LEN_1D):
        if K > 0:
            out_a[i] = src[i] * 2.0
        else:
            out_b[i] = src[i] + 1.0


@dace.program
def move_if_data_dep_nest(out: dace.float64[LEN_2D, LEN_2D], src: dace.float64[LEN_2D, LEN_2D],
                          cond: dace.float64[LEN_2D]):
    """A DATA-DEPENDENT guard ``cond[i]`` sits in the MIDDLE of a 2D loop
    nest, between the outer ``i`` loop and the inner ``j`` loop, gating the
    whole inner sweep of row ``i``. As written the inner loop is
    conditionally executed per row, so the nest cannot lift to a clean
    parallel Map. Moving the ``if`` INTO the inner loop body
    (``MoveIfIntoLoop``) rewrites it to ``for i: for j: if cond[i] > 0:``,
    a single 2D parallel Map with a per-row data-dependent predicate -- on
    GPU one parallel grid over ``(i, j)`` instead of a per-row branch that
    serializes the inner sweep. Rows with ``cond[i] <= 0`` leave
    ``out[i, :]`` untouched (caller pre-fills ``out``)."""
    for i in range(LEN_2D):
        if cond[i] > 0.0:
            for j in range(LEN_2D):
                out[i, j] = src[i, j] * 2.0


@dace.program
def fuse_move_ifs(a: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D, LEN_2D], src: dace.float64[LEN_2D, LEN_2D],
                  cond: dace.float64[LEN_2D]):
    """Follow-up to :func:`move_if_data_dep_nest`: two loop nests whose
    guards block fusion. The first nest has a data-dependent guard
    ``cond[i]`` in the middle (``for i: if cond[i] > 0: for j: ...``); the
    second has a loop-invariant guard ``K`` wrapping the whole nest
    (``if K > 0: for i: for j: ...``). Moving BOTH guards to the innermost
    position rewrites each to the same ``for i: for j: if ...:`` shape,
    after which the two nests -- now sharing one iteration space -- fuse
    into a single ``for i: for j:`` carrying both predicated bodies: one
    parallel Map / GPU grid instead of two. ``K`` is bound at call time;
    rows/cells whose guard is false leave their output untouched (caller
    pre-fills ``a`` and ``b``)."""
    for i in range(LEN_2D):
        if cond[i] > 0.0:
            for j in range(LEN_2D):
                a[i, j] = src[i, j] * 2.0
    if K > 0:
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                b[i, j] = src[i, j] + 1.0


# ==========================================================================
#  %V  Transformation-test gap kernels (map fusion / loop-to-map / fission)
# ==========================================================================
#
# Patterns drawn from the DaCe transformation tests: map fusion (vertical
# producer-consumer + horizontal sibling), loop-to-map write-disjointness
# certification (parallel vs sequential), a cloudsc-style conditional
# gather, and loop fission through gather/scatter indirection.


@dace.program
def fuse_stencil_through_transient(out: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
    """Non-pointwise vertical fusion (the offset-correction case). The
    producer is a 3-point stencil ``tmp[i] = a[i-1] + a[i] + a[i+1]``; the
    consumer reads the transient at an OFFSET: ``out[i] = tmp[i] * tmp[i+1]``.
    Because the consumer needs ``tmp[i+1]``, the maps are not a 1:1 merge --
    ``MapFusionVertical`` must apply offset correction (widen the producer
    read window) before it can collapse them and drop ``tmp``. Interior
    only; caller pre-fills the boundary cells of ``out``."""
    tmp = np.empty(LEN_1D, dtype=np.float64)
    for i in dace.map[1:LEN_1D - 1]:
        tmp[i] = a[i - 1] + a[i] + a[i + 1]
    for i in dace.map[1:LEN_1D - 2]:
        out[i] = tmp[i] * tmp[i + 1]


@dace.program
def fuse_diamond(out: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
    """Diamond producer-consumer fusion: one producer ``t = a*a`` feeds
    TWO consumers (``u = t + 1``, ``v = t - 1``) whose results join in a
    final map ``out = u * v``. The shared transient ``t`` is read by two
    downstream maps, so the fuser must fuse the diamond without
    duplicating the producer's work or serializing the two consumers --
    harder than a linear producer-consumer chain. All three transients
    (``t``, ``u``, ``v``) are eliminated when the diamond collapses to one
    map."""
    t = np.empty(LEN_1D, dtype=np.float64)
    u = np.empty(LEN_1D, dtype=np.float64)
    v = np.empty(LEN_1D, dtype=np.float64)
    for i in dace.map[0:LEN_1D]:
        t[i] = a[i] * a[i]
    for i in dace.map[0:LEN_1D]:
        u[i] = t[i] + 1.0
    for i in dace.map[0:LEN_1D]:
        v[i] = t[i] - 1.0
    for i in dace.map[0:LEN_1D]:
        out[i] = u[i] * v[i]


@dace.program
def loop_to_map_disjoint_strided(a: dace.float64[2 * LEN_1D], b: dace.float64[LEN_1D]):
    """Two strided writes per iteration to disjoint slots ``a[2*i]`` and
    ``a[2*i+1]``. A gcd-based disjointness proof (the two write index sets
    never collide) lets ``LoopToMap`` parallelize despite the
    two-writes-per-iteration shape."""
    for i in range(LEN_1D):
        a[2 * i] = b[i] + 1.0
        a[2 * i + 1] = b[i] * 2.0


@dace.program
def loop_to_map_overlap_seq(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """Counter-case to :func:`loop_to_map_disjoint_strided`: write index
    sets ``5*i`` and ``3*i`` collide across iterations (``gcd(5, 3) = 1``),
    so the loop carries a write-after-write conflict and ``LoopToMap`` must
    refuse -- the result depends on sequential iteration order. Iterates to
    ``LEN_1D // 5`` to keep both writes in range."""
    for i in range(LEN_1D // 5):
        a[5 * i] = b[i] + 1.0
        a[3 * i] = b[i] * 2.0


@dace.program
def loop_to_map_threshold_gather(out: dace.float64[LEN_2D, LEN_2D], x: dace.float64[LEN_2D, LEN_2D],
                                 y: dace.float64[LEN_2D, LEN_2D], w: dace.float64[LEN_2D, LEN_2D],
                                 idx: dace.int64[LEN_2D]):
    """cloudsc-style column physics: for each ``(i, k)`` a threshold on
    GATHERED data ``w[idx[i], k]`` selects which elementwise update writes
    ``out[i, k]``. Every ``(i, k)`` owns a distinct output cell, so
    ``LoopToMap`` parallelizes the whole 2D nest even though the predicate
    reads through the indirection ``idx``."""
    for i in range(LEN_2D):
        for k in range(LEN_2D):
            if w[idx[i], k] > 0.5:
                out[i, k] = x[i, k] * 2.0
            else:
                out[i, k] = y[i, k] + 1.0


@dace.program
def fission_gather_2body(b: dace.float64[LEN_1D], e: dace.float64[LEN_1D], a: dace.float64[LEN_1D],
                         c: dace.float64[LEN_1D], idx: dace.int64[LEN_1D]):
    """Two independent gathers sharing one index table: ``b[i] = a[idx[i]]``
    and ``e[i] = c[idx[i]]``. The shared ``idx`` read normally blocks
    ``MapFission``; the canonicalize path replicates the index read per
    output so the two gather bodies fission into independent maps. The
    indirect sibling of :func:`fission_indep_2body`."""
    for i in dace.map[0:LEN_1D]:
        b[i] = a[idx[i]]
        e[i] = c[idx[i]]


@dace.program
def fission_scatter_2body(b: dace.float64[LEN_1D], e: dace.float64[LEN_1D], a: dace.float64[LEN_1D],
                          c: dace.float64[LEN_1D], idx: dace.int64[LEN_1D]):
    """Two independent scatters sharing a permutation index:
    ``b[idx[i]] = a[i]*2`` and ``e[idx[i]] = c[i]+1``. Disjoint because
    ``idx`` is a permutation, so after fission each scatter is its own
    parallel map (guarded by the permutation proof)."""
    for i in dace.map[0:LEN_1D]:
        b[idx[i]] = a[i] * 2.0
        e[idx[i]] = c[i] + 1.0


__all__ = [
    "ext_strided_load_ssym",
    "ext_strided_load_2",
    "ext_strided_store_ssym",
    "ext_strided_store_2",
    "ext_gather_load",
    "ext_scatter_store",
    "ext_floordiv_offset",
    "ext_floordiv_offset_m",
    "ext_modular_wrap",
    "ext_war_unit",
    "ext_war_sym",
    "ext_peel_multi_back",
    "s121_sym_k",
    "s4113_ssym",
    "vas_ssym",
    "ext_tile_2d_sym",
    "fission_indep_2body",
    "fission_dep_then_indep",
    "fission_dep_const_offset",
    "fission_dep_sym_offset",
    "jacobi2d_tiled_const",
    "jacobi2d_tiled_sym",
    "jacobi2d_double_tiled_const",
    "jacobi2d_double_tiled_sym",
    "heat3d_tiled_const",
    "heat3d_tiled_sym",
    "ecrad_clamped_reduction",
    "masked_store_const",
    "masked_store_sym",
    "quasi_affine_reduce_even",
    "quasi_affine_reduce_odd",
    "quasi_affine_pairwise_sum",
    "quasi_affine_mod_k_stripe",
    "quasi_affine_floor_div_scatter",
    "wavefront2d",
    "ext_break_find_first",
    "ext_break_post_body",
    "ext_break_capture",
    "cond_reduce_sum",
    "cond_reduce_sym",
    "iv_additive",
    "iv_multiplicative",
    "argmax_value",
    "argmin_value",
    "neg_stride_rev",
    "reroll_saxpy7",
    "scan_strided_2",
    "scan_strided_sym",
    "scan_multi_carry",
    "scan_conditional",
    "scan_multi_5carry",
    "argmax_with_index",
    "reroll_gather",
    "thomas_solve",
    "reduce_inner_carry",
    "config_select_branch",
    "move_if_data_dep_nest",
    "fuse_move_ifs",
    "fuse_stencil_through_transient",
    "fuse_diamond",
    "loop_to_map_disjoint_strided",
    "loop_to_map_overlap_seq",
    "loop_to_map_threshold_gather",
    "fission_gather_2body",
    "fission_scatter_2body",
]


def collect():
    """All TSVC-2.5 kernels as ``@dace.program`` objects, in ``__all__`` order."""
    return [globals()[name] for name in __all__]


#: Concrete symbol values for correctness runs. ``LEN_3D < LEN_2D < LEN_1D`` so
#: the cubic 3D arrays stay small; tile/offset symbols are small but keep every
#: kernel's loop ranges in bounds.
SIZES = {
    "LEN_1D": 128, "LEN_2D": 32, "LEN_3D": 16, "ITERATIONS": 4,
    "S": 4, "SSYM": 3, "K": 3, "M": 4, "T": 4, "T1": 8, "T2": 4,
    # The reroll kernels are valid as written, but a non-multiple-of-7 tail triggers a
    # *separate, unfixed* canonicalize reroll-remainder bug (TODO). Bind their dedicated
    # length to a multiple of 7 so the suite isn't blocked on that fix; revisit once it lands.
    "LEN_R7": 7 * 18,  # 126
}


def make_inputs(program, seed: int = 1234):
    """Allocate inputs for one kernel from its ``@dace.program`` annotations.

    Array extents are the kernel's declared shapes evaluated at :data:`SIZES`;
    float arrays are random, integer arrays are a permutation of ``range(n)`` so
    gather/scatter indices stay in bounds, and scalars are random floats. The
    same values feed both the numpy oracle and the compiled SDFG.

    :returns: ``(arrays, scalars)`` keyed by parameter name.
    """
    import inspect
    rng = np.random.default_rng(seed)
    arrays, scalars = {}, {}
    for name, par in inspect.signature(program.f).parameters.items():
        ann = par.annotation
        if getattr(ann, "shape", None):  # dace array descriptor
            shape = tuple(int(dace.symbolic.evaluate(d, SIZES)) for d in ann.shape)
            if np.issubdtype(ann.dtype.as_numpy_dtype(), np.integer):
                arrays[name] = rng.permutation(shape[0]).astype(ann.dtype.as_numpy_dtype())
            else:
                arrays[name] = rng.standard_normal(shape).astype(ann.dtype.as_numpy_dtype())
        else:  # scalar
            scalars[name] = float(rng.standard_normal())
    return arrays, scalars
