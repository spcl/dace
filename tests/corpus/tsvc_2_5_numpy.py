# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numpy oracles for ``tsvc_2_5_core``.

One function per kernel, taking the same arguments as the DaCe / C++
counterpart (input arrays in, output arrays in-place). Used by the
correctness harness to verify the compiled kernel output.
"""
import numpy as np


def ref_strided_load_ssym(dst: np.ndarray, src: np.ndarray, scale: float, ssym: int) -> None:
    n = dst.shape[0]
    for i in range(n):
        dst[i] = src[i * ssym] * scale


def ref_strided_load_2(dst: np.ndarray, src: np.ndarray, scale: float) -> None:
    n = dst.shape[0]
    for i in range(n):
        dst[i] = src[i * 2] * scale


def ref_strided_store_ssym(dst: np.ndarray, src: np.ndarray, scale: float, ssym: int) -> None:
    n = src.shape[0]
    for i in range(n):
        dst[i * ssym] = src[i] * scale


def ref_strided_store_2(dst: np.ndarray, src: np.ndarray, scale: float) -> None:
    n = src.shape[0]
    for i in range(n):
        dst[i * 2] = src[i] * scale


def ref_gather_load(dst: np.ndarray, src: np.ndarray, idx: np.ndarray, scale: float) -> None:
    n = dst.shape[0]
    for i in range(n):
        dst[i] = src[idx[i]] * scale


def ref_scatter_store(dst: np.ndarray, src: np.ndarray, idx: np.ndarray, scale: float) -> None:
    n = src.shape[0]
    for i in range(n):
        dst[idx[i]] = src[i] * scale


def ref_floordiv_offset(a: np.ndarray, b: np.ndarray) -> None:
    n = a.shape[0]
    half = n // 2
    for i in range(half):
        a[i] = a[i + half] + b[i]


def ref_floordiv_offset_m(a: np.ndarray, b: np.ndarray, m: int) -> None:
    n = a.shape[0]
    chunk = n // m
    for i in range(chunk):
        a[i] = a[i + chunk] + b[i]


def ref_modular_wrap(a: np.ndarray, b: np.ndarray, k: int) -> None:
    n = a.shape[0]
    for i in range(n):
        a[(i + k) % n] = b[i]


def ref_war_unit(a: np.ndarray, b: np.ndarray) -> None:
    n = a.shape[0]
    for i in range(n - 1):
        a[i] = a[i + 1] + b[i]


def ref_war_sym(a: np.ndarray, b: np.ndarray, k: int) -> None:
    n = a.shape[0]
    for i in range(n - k):
        a[i] = a[i + k] + b[i]


def ref_peel_multi_back(a: np.ndarray, b: np.ndarray) -> None:
    n = a.shape[0]
    for i in range(n):
        a[i] = b[i] * 2.0
        if i == n - 1:
            a[n - 2] = a[n - 2] + 1.0
        elif i == n - 2:
            a[n - 3] = a[n - 3] + 1.0


def ref_tile_2d_sym(b: np.ndarray, a: np.ndarray, s: int) -> None:
    n = a.shape[0]  # assume square
    for ti in range(0, n, s):
        for tj in range(0, n, s):
            for i in range(ti, ti + s):
                for j in range(tj, tj + s):
                    b[i, j] = a[i, j] * 2.0


# TSVC-named symbolic-step variants


def ref_s121_sym_k(a: np.ndarray, b: np.ndarray, k: int) -> None:
    """TSVC s121 with symbolic offset ``k``: ``a[i] = a[i+k] + b[i]``."""
    n = a.shape[0]
    for i in range(n - k):
        a[i] = a[i + k] + b[i]


def ref_s4113_ssym(a: np.ndarray, b: np.ndarray, c: np.ndarray, ip: np.ndarray, ssym: int) -> None:
    """TSVC s4113 with strided index access:
    ``a[ip[i*ssym]] = b[ip[i*ssym]] + c[i]``."""
    n = a.shape[0]
    for i in range(n // ssym):
        a[ip[i * ssym]] = b[ip[i * ssym]] + c[i]


def ref_vas_ssym(a: np.ndarray, b: np.ndarray, ip: np.ndarray, ssym: int) -> None:
    """TSVC vas with strided index scatter: ``a[ip[i*ssym]] = b[i]``."""
    n = a.shape[0]
    for i in range(n // ssym):
        a[ip[i * ssym]] = b[i]


# Loop-fission family


def ref_fission_indep_2body(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Two independent writes sharing three reads."""
    n = a.shape[0]
    for i in range(n):
        a[i] = x[i] * y[i] + z[i]
        b[i] = x[i] - y[i] * z[i]


def ref_fission_dep_then_indep(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Body A unit-offset carried dep, body B independent."""
    n = a.shape[0]
    a[0] = x[0]
    for i in range(1, n):
        a[i] = a[i - 1] + x[i]
        b[i] = y[i] * 2.0


def ref_fission_dep_const_offset(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Body A offset-2 carried dep, body B independent."""
    n = a.shape[0]
    a[0] = x[0]
    a[1] = x[1]
    for i in range(2, n):
        a[i] = a[i - 2] + x[i]
        b[i] = y[i] * z[i]


def ref_fission_dep_sym_offset(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               k: int) -> None:
    """Body A symbolic-offset (``k``) carried dep, body B independent.
    Caller must initialize ``a[0..k-1]``."""
    n = a.shape[0]
    for i in range(k, n):
        a[i] = a[i - k] + x[i]
        b[i] = y[i] * z[i]


# Already-tiled stencils


def _jacobi2d_tiled(b: np.ndarray, a: np.ndarray, t: int) -> None:
    n = a.shape[0]
    for ii in range(1, n - 1 - t, t):
        for jj in range(1, n - 1 - t, t):
            for i in range(ii, ii + t):
                for j in range(jj, jj + t):
                    b[i, j] = 0.2 * (a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


def ref_jacobi2d_tiled_const(b: np.ndarray, a: np.ndarray) -> None:
    """2D Jacobi 5-point stencil pre-tiled with constant tile size 64."""
    _jacobi2d_tiled(b, a, 64)


def ref_jacobi2d_tiled_sym(b: np.ndarray, a: np.ndarray, t: int) -> None:
    """2D Jacobi 5-point stencil pre-tiled with symbolic tile size ``t``."""
    _jacobi2d_tiled(b, a, t)


def _jacobi2d_double_tiled(b: np.ndarray, a: np.ndarray, t1: int, t2: int) -> None:
    n = a.shape[0]
    for ii in range(1, n - 1 - t1, t1):
        for jj in range(1, n - 1 - t1, t1):
            for iii in range(ii, ii + t1, t2):
                for jjj in range(jj, jj + t1, t2):
                    for i in range(iii, iii + t2):
                        for j in range(jjj, jjj + t2):
                            b[i, j] = 0.2 * (a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


def ref_jacobi2d_double_tiled_const(b: np.ndarray, a: np.ndarray) -> None:
    """2D Jacobi 5-point stencil pre-tiled with constant outer (64) and inner (8) tiles."""
    _jacobi2d_double_tiled(b, a, 64, 8)


def ref_jacobi2d_double_tiled_sym(b: np.ndarray, a: np.ndarray, t1: int, t2: int) -> None:
    """Two-level Jacobi tiling with symbolic outer ``t1`` and inner ``t2``."""
    _jacobi2d_double_tiled(b, a, t1, t2)


def _heat3d_tiled(b: np.ndarray, a: np.ndarray, t: int) -> None:
    n = a.shape[0]
    for kk in range(1, n - 1 - t, t):
        for jj in range(1, n - 1 - t, t):
            for ii in range(1, n - 1 - t, t):
                for k in range(kk, kk + t):
                    for j in range(jj, jj + t):
                        for i in range(ii, ii + t):
                            b[k, j, i] = (0.125 * (a[k + 1, j, i] - 2.0 * a[k, j, i] + a[k - 1, j, i]) + 0.125 *
                                          (a[k, j + 1, i] - 2.0 * a[k, j, i] + a[k, j - 1, i]) + 0.125 *
                                          (a[k, j, i + 1] - 2.0 * a[k, j, i] + a[k, j, i - 1]) + a[k, j, i])


def ref_heat3d_tiled_const(b: np.ndarray, a: np.ndarray) -> None:
    """3D 7-point heat stencil pre-tiled with constant tile size 8."""
    _heat3d_tiled(b, a, 8)


def ref_heat3d_tiled_sym(b: np.ndarray, a: np.ndarray, t: int) -> None:
    """3D 7-point heat stencil pre-tiled with symbolic tile size ``t``."""
    _heat3d_tiled(b, a, t)


# ECRAD-style clamped reduction


def ref_ecrad_clamped_reduction(out: np.ndarray, x: np.ndarray, y: np.ndarray, d: np.ndarray) -> None:
    """clamp(exp(-sqrt(max(x*x+y*y, 1e-12)) * d), 0, 1)."""
    n = out.shape[0]
    for i in range(n):
        k_val = np.sqrt(max(x[i] * x[i] + y[i] * y[i], 1e-12))
        e = np.exp(-k_val * d[i])
        out[i] = max(0.0, min(e, 1.0))


# Masked stores


def ref_masked_store_const(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> None:
    """Predicated store: ``if mask[i] > 0: a[i] = b[i]``."""
    n = a.shape[0]
    for i in range(n):
        if mask[i] > 0:
            a[i] = b[i]


def ref_masked_store_sym(a: np.ndarray, b: np.ndarray, threshold_data: np.ndarray, k: float) -> None:
    """Predicated store: ``if threshold_data[i] > k: a[i] = b[i]``."""
    n = a.shape[0]
    for i in range(n):
        if threshold_data[i] > k:
            a[i] = b[i]


# ---------------------------------------------------------------------------
#  Quasi-affine subscript / iteration patterns
# ---------------------------------------------------------------------------


def ref_quasi_affine_reduce_even(a: np.ndarray, out: np.ndarray) -> None:
    """``out[0] = sum(a[0::2])`` (stride-2 reduction from index 0)."""
    out[0] = float(a[0::2].sum())


def ref_quasi_affine_reduce_odd(a: np.ndarray, out: np.ndarray) -> None:
    """``out[0] = sum(a[1::2])`` (stride-2 reduction from index 1)."""
    out[0] = float(a[1::2].sum())


def ref_quasi_affine_pairwise_sum(a: np.ndarray, b: np.ndarray) -> None:
    """``b[i] = a[2*i] + a[2*i + 1]`` -- pairwise gather + add."""
    n = b.shape[0]
    b[:] = a[0:2 * n:2] + a[1:2 * n:2]


def ref_quasi_affine_mod_k_stripe(a: np.ndarray, b: np.ndarray, c: np.ndarray, k: int) -> None:
    """``a[i] = b[i] * 2.0 if i % k == 0 else c[i]`` -- mod-k stripe."""
    n = a.shape[0]
    idx = np.arange(n)
    mask = (idx % k) == 0
    a[mask] = b[mask] * 2.0
    a[~mask] = c[~mask]


def ref_quasi_affine_floor_div_scatter(a: np.ndarray, b: np.ndarray) -> None:
    """``b[i // 2] += a[i]`` -- pair-stripe reduction. The numpy
    oracle uses ``np.add.at`` for the unbuffered scatter so each pair
    of source indices accumulates into the same output cell."""
    n2 = a.shape[0]
    idx = np.arange(n2) // 2
    np.add.at(b, idx, a)


# ---------------------------------------------------------------------------
#  Wavefront / loop-skew
# ---------------------------------------------------------------------------


def ref_wavefront2d(a: np.ndarray) -> None:
    """``a[i,j] = 0.25*(a[i,j] + a[i-1,j] + a[i,j-1] + a[i-1,j-1])``.
    Sequential by row/column: the in-place reads of already-updated
    neighbours make the scalar loop order significant, so the oracle
    keeps the explicit nested sweep."""
    n = a.shape[0]
    for i in range(1, n):
        for j in range(1, n):
            a[i, j] = 0.25 * (a[i, j] + a[i - 1, j] + a[i, j - 1] + a[i - 1, j - 1])


# ---------------------------------------------------------------------------
#  Early-exit / find-first (break loops)
# ---------------------------------------------------------------------------


def ref_break_find_first(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> None:
    """TSVC ``s481``: ``if d[i] < 0: break`` then ``a[i] += b[i]*c[i]``."""
    n = a.shape[0]
    for i in range(n):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]


def ref_break_post_body(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """TSVC ``s482``: ``a[i] += b[i]*c[i]`` then ``if c[i] > b[i]: break``."""
    n = a.shape[0]
    for i in range(n):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break


def ref_break_capture(a: np.ndarray, out_index: np.ndarray, out_value: np.ndarray, k: float) -> None:
    """TSVC ``s332``: first ``i`` with ``a[i] > k`` -> capture index +
    value, break. ``out_index``/``out_value`` stay at ``-1`` if no
    element exceeds ``k``."""
    n = a.shape[0]
    out_index[0] = -1
    out_value[0] = -1.0
    for i in range(n):
        if a[i] > k:
            out_index[0] = i
            out_value[0] = a[i]
            break


# ---------------------------------------------------------------------------
#  Conditional reduction
# ---------------------------------------------------------------------------


def ref_cond_reduce_sum(a: np.ndarray, out: np.ndarray) -> None:
    """TSVC ``s3111``: ``out[0] = sum(a[a > 0])``."""
    out[0] = float(a[a > 0.0].sum())


def ref_cond_reduce_sym(a: np.ndarray, out: np.ndarray, k: float) -> None:
    """Symbolic-threshold conditional sum: ``out[0] = sum(a[a > k])``."""
    out[0] = float(a[a > k].sum())


# ---------------------------------------------------------------------------
#  Induction-variable closed form
# ---------------------------------------------------------------------------


def ref_iv_additive(out: np.ndarray, n: int) -> None:
    """Additive IV closed form: ``out[0] = 1.5 * n``. ``n`` is the trip
    count (the ``LEN_1D`` symbol), not derivable from ``out``'s shape."""
    s = 0.0
    for _ in range(n):
        s = s + 1.5
    out[0] = s


def ref_iv_multiplicative(out: np.ndarray, n: int) -> None:
    """Multiplicative IV closed form: ``out[0] = 0.99 ** n``."""
    s = 1.0
    for _ in range(n):
        s = s * 0.99
    out[0] = s


# ---------------------------------------------------------------------------
#  Argmax / argmin value
# ---------------------------------------------------------------------------


def ref_argmax_value(a: np.ndarray, out: np.ndarray) -> None:
    """TSVC ``s314``: ``out[0] = max(a)``."""
    out[0] = float(a.max())


def ref_argmin_value(a: np.ndarray, out: np.ndarray) -> None:
    """TSVC ``s316``: ``out[0] = min(a)``."""
    out[0] = float(a.min())


# ---------------------------------------------------------------------------
#  Negative stride + manual unroll
# ---------------------------------------------------------------------------


def ref_neg_stride_rev(a: np.ndarray, b: np.ndarray) -> None:
    """Reverse-iteration write (no carried dep): ``a[i] = b[i] + 1``."""
    n = a.shape[0]
    for i in range(n - 1, -1, -1):
        a[i] = b[i] + 1.0


def ref_reroll_saxpy7(a: np.ndarray, b: np.ndarray) -> None:
    """TSVC ``s351``: 7x (prime) hand-unrolled saxpy. Net effect is
    ``a[i] += b[i] * 2`` for every ``i`` in a full group of 7; any
    ``n % 7`` remainder elements at the tail are left untouched."""
    n = a.shape[0]
    for i in range(0, n - 6, 7):  # full groups of 7; remainder tail left untouched (matches kernel)
        a[i] = a[i] + b[i] * 2.0
        a[i + 1] = a[i + 1] + b[i + 1] * 2.0
        a[i + 2] = a[i + 2] + b[i + 2] * 2.0
        a[i + 3] = a[i + 3] + b[i + 3] * 2.0
        a[i + 4] = a[i + 4] + b[i + 4] * 2.0
        a[i + 5] = a[i + 5] + b[i + 5] * 2.0
        a[i + 6] = a[i + 6] + b[i + 6] * 2.0


# ---------------------------------------------------------------------------
#  Strided / multiple scans
# ---------------------------------------------------------------------------


def ref_scan_strided_2(a: np.ndarray, x: np.ndarray) -> None:
    """Stride-2 prefix sum ``a[i] = a[i-2] + x[i]``. Caller seeds
    ``a[0]``/``a[1]``; the even/odd subsequences are two scans."""
    n = a.shape[0]
    for i in range(2, n):
        a[i] = a[i - 2] + x[i]


def ref_scan_strided_sym(a: np.ndarray, x: np.ndarray, k: int) -> None:
    """Stride-``k`` prefix sum ``a[i] = a[i-k] + x[i]``. Caller seeds
    ``a[0..k-1]``; the ``k`` residue classes are ``k`` scans."""
    n = a.shape[0]
    for i in range(k, n):
        a[i] = a[i - k] + x[i]


def ref_scan_multi_carry(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Two scans in one body: additive on ``a``, multiplicative on ``b``.
    Caller seeds ``a[0]``/``b[0]``."""
    n = a.shape[0]
    for i in range(1, n):
        a[i] = a[i - 1] + x[i]
        b[i] = b[i - 1] * y[i]


# ---------------------------------------------------------------------------
#  Canonicalize unit-test gap kernels
# ---------------------------------------------------------------------------


def ref_scan_conditional(out: np.ndarray, delta: np.ndarray, mask: np.ndarray) -> None:
    """Masked prefix scan: advance where ``mask[i] > 0``, else hold.
    Caller seeds ``out[0]``."""
    n = out.shape[0]
    for i in range(1, n):
        if mask[i] > 0:
            out[i] = out[i - 1] + delta[i]
        else:
            out[i] = out[i - 1]


def ref_scan_multi_5carry(acc: np.ndarray, delta: np.ndarray) -> None:
    """Five independent prefix sums ``acc[r, i] = acc[r, i-1] + delta[r, i]``
    (``acc``/``delta`` are ``(5, n)``). Caller seeds ``acc[:, 0]``."""
    n = acc.shape[1]
    for i in range(1, n):
        for r in range(5):
            acc[r, i] = acc[r, i - 1] + delta[r, i]


def ref_argmax_with_index(a: np.ndarray, out_value: np.ndarray, out_index: np.ndarray) -> None:
    """TSVC ``s315``: running max value + (first) index."""
    out_value[0] = float(a.max())
    out_index[0] = int(a.argmax())


def ref_reroll_gather(a: np.ndarray, b: np.ndarray, ip: np.ndarray) -> None:
    """TSVC ``s353``: 7x hand-unrolled gather saxpy. Net effect over the
    full array is ``a[i] += b[ip[i]] * 2`` for every ``i`` (``n``
    divisible by 7)."""
    n = a.shape[0]
    for i in range(0, n - 6, 7):  # full groups of 7; remainder tail left untouched (matches kernel)
        for k in range(7):
            a[i + k] = a[i + k] + b[ip[i + k]] * 2.0


def ref_thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, x: np.ndarray) -> None:
    """Tridiagonal Thomas solve: forward elimination then backward
    substitution. ``c`` and ``d`` are overwritten as scratch; ``x`` holds
    the solution."""
    n = a.shape[0]
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * c[i - 1]
        c[i] = c[i] / m
        d[i] = (d[i] - a[i] * d[i - 1]) / m
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]


def ref_reduce_inner_carry(a: np.ndarray, out: np.ndarray) -> None:
    """Row-wise reduction: ``out[i] = sum_j a[i, j]`` (outer parallel,
    inner carried)."""
    out[:] = a.sum(axis=1)


def ref_config_select_branch(out_a: np.ndarray, out_b: np.ndarray, src: np.ndarray, k: int) -> None:
    """Loop-invariant config flag selects the output array:
    ``if k > 0: out_a = src*2 else: out_b = src+1``. The unselected output
    is left untouched (caller pre-fills)."""
    if k > 0:
        out_a[:] = src * 2.0
    else:
        out_b[:] = src + 1.0


def ref_move_if_data_dep_nest(out: np.ndarray, src: np.ndarray, cond: np.ndarray) -> None:
    """Data-dependent per-row guard over a 2D nest: rows with
    ``cond[i] > 0`` get ``out[i, :] = src[i, :] * 2``; other rows are left
    untouched (caller pre-fills ``out``)."""
    n = src.shape[0]
    for i in range(n):
        if cond[i] > 0.0:
            out[i, :] = src[i, :] * 2.0


def ref_fuse_move_ifs(a: np.ndarray, b: np.ndarray, src: np.ndarray, cond: np.ndarray, k: int) -> None:
    """Two guarded nests: data-dependent ``cond[i]`` on ``a``, then
    loop-invariant ``k`` on ``b``. Cells whose guard is false are left
    untouched (caller pre-fills ``a`` and ``b``)."""
    n = src.shape[0]
    for i in range(n):
        if cond[i] > 0.0:
            a[i, :] = src[i, :] * 2.0
    if k > 0:
        b[:, :] = src + 1.0


# ---------------------------------------------------------------------------
#  Transformation-test gap kernels (fusion / loop-to-map / indirect fission)
# ---------------------------------------------------------------------------


def ref_fuse_stencil_through_transient(out: np.ndarray, a: np.ndarray) -> None:
    """Non-pointwise fusion: ``tmp[i] = a[i-1]+a[i]+a[i+1]`` (interior),
    then ``out[i] = tmp[i] * tmp[i+1]`` for ``i`` in ``[1, n-2)``. Boundary
    cells of ``out`` are left untouched (caller pre-fills)."""
    n = a.shape[0]
    tmp = np.empty(n)
    for i in range(1, n - 1):
        tmp[i] = a[i - 1] + a[i] + a[i + 1]
    for i in range(1, n - 2):
        out[i] = tmp[i] * tmp[i + 1]


def ref_fuse_diamond(out: np.ndarray, a: np.ndarray) -> None:
    """Diamond fusion result: ``t = a*a; out = (t+1) * (t-1)``."""
    t = a * a
    out[:] = (t + 1.0) * (t - 1.0)


def ref_loop_to_map_disjoint_strided(a: np.ndarray, b: np.ndarray) -> None:
    """Disjoint strided writes: ``a[2*i] = b[i] + 1``; ``a[2*i+1] = b[i] * 2``."""
    n = b.shape[0]
    a[0:2 * n:2] = b + 1.0
    a[1:2 * n:2] = b * 2.0


def ref_loop_to_map_overlap_seq(a: np.ndarray, b: np.ndarray) -> None:
    """Overlapping writes, order-dependent: ``a[5*i] = b[i]+1``;
    ``a[3*i] = b[i]*2`` in sequential iteration order."""
    n = a.shape[0]
    for i in range(n // 5):
        a[5 * i] = b[i] + 1.0
        a[3 * i] = b[i] * 2.0


def ref_loop_to_map_threshold_gather(out: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                                     idx: np.ndarray) -> None:
    """Per-cell threshold on gathered ``w[idx[i], k]`` selects the update
    for ``out[i, k]``."""
    n = out.shape[0]
    for i in range(n):
        for k in range(n):
            if w[idx[i], k] > 0.5:
                out[i, k] = x[i, k] * 2.0
            else:
                out[i, k] = y[i, k] + 1.0


def ref_fission_gather_2body(b: np.ndarray, e: np.ndarray, a: np.ndarray, c: np.ndarray, idx: np.ndarray) -> None:
    """Two independent gathers: ``b[i] = a[idx[i]]``; ``e[i] = c[idx[i]]``."""
    b[:] = a[idx]
    e[:] = c[idx]


def ref_fission_scatter_2body(b: np.ndarray, e: np.ndarray, a: np.ndarray, c: np.ndarray, idx: np.ndarray) -> None:
    """Two independent scatters (``idx`` a permutation): ``b[idx[i]] = a[i]*2``;
    ``e[idx[i]] = c[i]+1``."""
    b[idx] = a * 2.0
    e[idx] = c + 1.0
