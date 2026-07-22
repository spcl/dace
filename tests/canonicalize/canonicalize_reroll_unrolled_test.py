# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on manually-unrolled (lane-chain) loops -- TSVC ``s353`` shape.

A loop with step ``S != 1`` whose body is ``S`` manually-unrolled lanes (the
lane-``k`` statement is lane 0 with every index shifted by ``+k``) should be
re-rolled (un-tiled) to a step-1 loop so ``LoopToMap`` can parallelize it. Two
forms are covered:

* **dense** -- ``a[i+k] += alpha * b[i+k]``
* **indirect** (TSVC ``s353``) -- ``a[i+k] += alpha * b[ip[i+k]]`` (gather)

Canonicalize is value-correct on both today (the value tests pass). The
re-roll-to-a-parallel-map step is a documented gap (CORE_BUGFIXES.md L-E):
canonicalize normalizes the step-``S`` loop to step 1 but keeps the ``S`` lanes
(``a[S*i + k]``), and ``LoopToMap`` then refuses on the multi-lane read-write
pattern. The structural tests are strict xfails pinning that target.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


@dace.program
def unrolled_dense(a: dace.float64[N], b: dace.float64[N], alpha: dace.float64):
    for i in range(0, N - 3, 4):
        a[i] = a[i] + alpha * b[i]
        a[i + 1] = a[i + 1] + alpha * b[i + 1]
        a[i + 2] = a[i + 2] + alpha * b[i + 2]
        a[i + 3] = a[i + 3] + alpha * b[i + 3]


@dace.program
def unrolled_indirect(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N], alpha: dace.float64):
    for i in range(0, N - 3, 4):
        a[i] = a[i] + alpha * b[ip[i]]
        a[i + 1] = a[i + 1] + alpha * b[ip[i + 1]]
        a[i + 2] = a[i + 2] + alpha * b[ip[i + 2]]
        a[i + 3] = a[i + 3] + alpha * b[ip[i + 3]]


def test_unrolled_dense_value_preserving():
    n = 16
    rng = np.random.default_rng(0)
    a0, b = rng.standard_normal(n), rng.standard_normal(n)
    alpha = np.float64(2.5)
    sdfg = unrolled_dense.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b, alpha=alpha, N=n)
    assert np.allclose(got, a0 + alpha * b)


def test_unrolled_indirect_value_preserving():
    n = 16
    rng = np.random.default_rng(1)
    a0, b = rng.standard_normal(n), rng.standard_normal(n)
    ip = rng.permutation(n).astype(np.int32)
    alpha = np.float64(1.3)
    sdfg = unrolled_indirect.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b, ip=ip, alpha=alpha, N=n)
    assert np.allclose(got, a0 + alpha * b[ip])


def test_unrolled_dense_becomes_map():
    n = 16
    sdfg = unrolled_dense.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1, 'expected the re-rolled loop to parallelize into a map'


def test_unrolled_indirect_becomes_map():
    n = 16
    sdfg = unrolled_indirect.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1, 'expected the re-rolled gather loop to parallelize into a map'


M = dace.symbol('M')


@dace.program
def unrolled_unit_step2(a: dace.float64[M], b: dace.float64[M]):
    """Step 2, lanes at offsets {0, 1} (spacing 1) -- re-rolls to step 1."""
    for i in range(0, M, 2):
        a[i] = b[i] * 2.0
        a[i + 1] = b[i + 1] * 2.0


@dace.program
def unrolled_strided(a: dace.float64[M], b: dace.float64[M]):
    """Step 2, lanes at offsets {0, 2} (spacing 2, overlapping pure writes) --
    re-rolls to step 2 (the offset spacing), not step 1."""
    for i in range(0, M - 2, 2):
        a[i] = b[i] * 3.0
        a[i + 2] = b[i + 2] * 3.0


def test_unrolled_unit_step2_value_and_map():
    n = 12
    rng = np.random.default_rng(5)
    b = rng.standard_normal(n)
    a0 = rng.standard_normal(n)
    sdfg = unrolled_unit_step2.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b, M=n)
    exp = a0.copy()
    for i in range(0, n, 2):
        exp[i] = b[i] * 2.0
        exp[i + 1] = b[i + 1] * 2.0
    assert np.allclose(got, exp)
    assert _nmaps(sdfg) >= 1, 'step-2 / offset-spacing-1 unroll should re-roll to a step-1 map'


def test_unrolled_strided_value_and_map():
    n = 12
    rng = np.random.default_rng(6)
    b = rng.standard_normal(n)
    a0 = rng.standard_normal(n)
    sdfg = unrolled_strided.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b, M=n)
    exp = a0.copy()
    for i in range(0, n - 2, 2):
        exp[i] = b[i] * 3.0
        exp[i + 2] = b[i + 2] * 3.0
    assert np.allclose(got, exp)
    assert _nmaps(sdfg) >= 1, 'step-2 / offset-spacing-2 unroll should re-roll to a step-2 map'


@dace.program
def unrolled_dot_product(a: dace.float64[N], b: dace.float64[N], c: dace.float64[2]):
    """TSVC ``s352``: a single-expression ``m``-term dot product with manual unroll.

    The body's ``m=5`` lanes share a left-folded ``_Add_`` reduction tree -- the
    associative-merge generalization of :class:`RerollUnrolledLoops` allows that
    overlap and collapses the tree to lane 0, leaving a step-1 dot product that
    ``LoopToReduce`` / ``LoopToMap`` can parallelize."""
    dot = 0.0
    for i in range(0, N - 4, 5):
        dot = dot + (a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] +
                     a[i + 4] * b[i + 4])
    c[0] = dot


def test_unrolled_dot_product_value_preserving():
    n = 25
    rng = np.random.default_rng(7)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    sdfg = unrolled_dot_product.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    c = np.zeros(2)
    sdfg(a=a.copy(), b=b.copy(), c=c, N=n)
    # The rerolled loop computes ``sum(a[0:n] * b[0:n])`` -- the same value as
    # the original lane-summed form, even though the access pattern changed.
    assert np.isclose(c[0], float(np.dot(a, b)))


def test_unrolled_dot_product_becomes_map_or_reduce():
    """After re-roll, the body is a single-lane dot accumulator; canonicalize
    should turn the loop into either a parallel reduction map or a ``Reduce``."""
    from dace.sdfg.state import LoopRegion
    sdfg = unrolled_dot_product.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    n_maps = _nmaps(sdfg)
    n_reduces = sum(1 for n, _ in sdfg.all_nodes_recursive()
                    if isinstance(n, nodes.LibraryNode) and 'Reduce' in type(n).__name__)
    n_loops = sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
    assert (n_maps + n_reduces) >= 1 and n_loops == 0, (
        f'expected a map or reduce, got maps={n_maps}, reduces={n_reduces}, loops={n_loops}')


@dace.program
def unrolled_dot_nonaligned(a: dace.float64[N], b: dace.float64[N], c: dace.float64[2]):
    """Step-5 dot whose iteration range is NOT a multiple of the step, so the
    source loop skips the final partial group. Pins the re-roll bound: the
    re-rolled step-1 loop must cover exactly the original positions (alignment-
    aware ``last_i``). A naive ``end + m*g`` bound would extend over the skipped
    tail and add spurious reduction terms (the s352 corpus miscompile)."""
    dot = 0.0
    for i in range(0, N - 4, 5):
        dot = dot + (a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] +
                     a[i + 4] * b[i + 4])
    c[0] = dot


def test_unrolled_dot_nonaligned_skips_tail():
    n = 27  # range(0, 23, 5) = {0,5,10,15,20} -> covers 0..24; positions 25,26 skipped
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    sdfg = unrolled_dot_nonaligned.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    c = np.zeros(2)
    sdfg(a=a.copy(), b=b.copy(), c=c, N=n)
    # Faithful reference: grouped-by-5, the unaligned tail (25, 26) skipped.
    ref = 0.0
    for i in range(0, n - 4, 5):
        ref += sum(a[i + k] * b[i + k] for k in range(5))
    assert np.isclose(c[0], ref), f'got {c[0]} expected {ref} (re-roll must skip the unaligned tail)'


# --------------------------------------------------------------------------
# Manual-unroll variants with an explicit remainder loop (the unrolled main
# body covers the largest multiple-of-K prefix; a step-1 remainder handles the
# up-to-(K-1) trailing elements). ``(N // K) * K`` makes the tiling exact for
# any N, so the re-rolled result must cover every position exactly once.
# --------------------------------------------------------------------------
@dace.program
def unroll_body_plus_remainder(a: dace.float64[N], b: dace.float64[N]):
    """Elementwise square, unrolled by 4 with a scalar remainder loop."""
    for i in range(0, N - 3, 4):
        a[i] = b[i] * b[i]
        a[i + 1] = b[i + 1] * b[i + 1]
        a[i + 2] = b[i + 2] * b[i + 2]
        a[i + 3] = b[i + 3] * b[i + 3]
    for i in range((N // 4) * 4, N):
        a[i] = b[i] * b[i]


@dace.program
def unroll_partial_5_then_12(a: dace.float64[N], b: dace.float64[N], s: dace.float64):
    """Elementwise scale, a 17-wide unrolled body expressed as 5 lanes then 12
    lanes, with a scalar remainder."""
    for i in range(0, N - 16, 17):
        a[i] = b[i] * s
        a[i + 1] = b[i + 1] * s
        a[i + 2] = b[i + 2] * s
        a[i + 3] = b[i + 3] * s
        a[i + 4] = b[i + 4] * s
        a[i + 5] = b[i + 5] * s
        a[i + 6] = b[i + 6] * s
        a[i + 7] = b[i + 7] * s
        a[i + 8] = b[i + 8] * s
        a[i + 9] = b[i + 9] * s
        a[i + 10] = b[i + 10] * s
        a[i + 11] = b[i + 11] * s
        a[i + 12] = b[i + 12] * s
        a[i + 13] = b[i + 13] * s
        a[i + 14] = b[i + 14] * s
        a[i + 15] = b[i + 15] * s
        a[i + 16] = b[i + 16] * s
    for i in range((N // 17) * 17, N):
        a[i] = b[i] * s


@dace.program
def unroll_prime_17_uniform(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Elementwise add, uniformly unrolled by the prime 17 with a remainder."""
    for i in range(0, N - 16, 17):
        a[i] = b[i] + c[i]
        a[i + 1] = b[i + 1] + c[i + 1]
        a[i + 2] = b[i + 2] + c[i + 2]
        a[i + 3] = b[i + 3] + c[i + 3]
        a[i + 4] = b[i + 4] + c[i + 4]
        a[i + 5] = b[i + 5] + c[i + 5]
        a[i + 6] = b[i + 6] + c[i + 6]
        a[i + 7] = b[i + 7] + c[i + 7]
        a[i + 8] = b[i + 8] + c[i + 8]
        a[i + 9] = b[i + 9] + c[i + 9]
        a[i + 10] = b[i + 10] + c[i + 10]
        a[i + 11] = b[i + 11] + c[i + 11]
        a[i + 12] = b[i + 12] + c[i + 12]
        a[i + 13] = b[i + 13] + c[i + 13]
        a[i + 14] = b[i + 14] + c[i + 14]
        a[i + 15] = b[i + 15] + c[i + 15]
        a[i + 16] = b[i + 16] + c[i + 16]
    for i in range((N // 17) * 17, N):
        a[i] = b[i] + c[i]


@dace.program
def unroll_reduction_11_accs(a: dace.float64[N], out: dace.float64[1]):
    """Sum reduction unrolled by 11 into 11 independent partial accumulators,
    combined at the end; plus a scalar remainder into the first accumulator."""
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    s6 = 0.0
    s7 = 0.0
    s8 = 0.0
    s9 = 0.0
    s10 = 0.0
    for i in range(0, N - 10, 11):
        s0 = s0 + a[i]
        s1 = s1 + a[i + 1]
        s2 = s2 + a[i + 2]
        s3 = s3 + a[i + 3]
        s4 = s4 + a[i + 4]
        s5 = s5 + a[i + 5]
        s6 = s6 + a[i + 6]
        s7 = s7 + a[i + 7]
        s8 = s8 + a[i + 8]
        s9 = s9 + a[i + 9]
        s10 = s10 + a[i + 10]
    for i in range((N // 11) * 11, N):
        s0 = s0 + a[i]
    out[0] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10


def test_unroll_body_plus_remainder_value_and_map():
    n = 18  # remainder 2
    rng = np.random.default_rng(20)
    b = rng.standard_normal(n)
    sdfg = unroll_body_plus_remainder.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = np.zeros(n)
    sdfg(a=got, b=b.copy(), N=n)
    assert np.allclose(got, b * b)
    assert _nmaps(sdfg) >= 1, 'the re-rolled unrolled body must parallelize into a map'


def test_unroll_partial_5_then_12_value_and_map():
    n = 39  # remainder 5
    rng = np.random.default_rng(21)
    b = rng.standard_normal(n)
    s = np.float64(1.7)
    sdfg = unroll_partial_5_then_12.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = np.zeros(n)
    sdfg(a=got, b=b.copy(), s=s, N=n)
    assert np.allclose(got, b * s)
    assert _nmaps(sdfg) >= 1, 'the re-rolled unrolled body must parallelize into a map'


def test_unroll_prime_17_uniform_value_and_map():
    n = 39  # remainder 5
    rng = np.random.default_rng(22)
    b, c = rng.standard_normal(n), rng.standard_normal(n)
    sdfg = unroll_prime_17_uniform.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = np.zeros(n)
    sdfg(a=got, b=b.copy(), c=c.copy(), N=n)
    assert np.allclose(got, b + c)
    assert _nmaps(sdfg) >= 1, 'the re-rolled unrolled body must parallelize into a map'


def _nreduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, nodes.LibraryNode) and 'Reduce' in type(n).__name__)


def test_unroll_reduction_11_accs_value_and_reduce():
    """A sum reduction hand-unrolled into 11 partial accumulators (over a
    strided main body) plus a step-1 remainder is a *tiled reduction*: re-roll
    collapses the 11-lane main body to a single-accumulator unit-stride loop,
    and ``FuseConsecutiveLoops`` re-joins that loop with its adjacent remainder
    so the whole thing lifts to ONE ``Reduce`` over ``a[0:N]`` -- exact, not the
    former half-sum (two un-chained reduces writing the same accumulator)."""
    n = 25  # remainder 3
    rng = np.random.default_rng(23)
    a = rng.standard_normal(n)
    sdfg = unroll_reduction_11_accs.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(1)
    sdfg(a=a.copy(), out=out, N=n)
    assert np.isclose(out[0], np.sum(a)), f'got {out[0]} expected {np.sum(a)}'
    assert (_nmaps(sdfg) + _nreduces(sdfg)) >= 1, 'the multi-accumulator unrolled reduction must lift to a map/reduce'


@dace.program
def unrolled_lanes_read_different_arrays(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N],
                                         c: dace.float64[N]):
    for i in range(0, N, 2):
        c[i] = a[i] * b[i]
        c[i + 1] = a[i + 1] * d[i + 1]


@dace.program
def unrolled_lanes_write_different_arrays(a: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in range(0, N, 2):
        c[i] = a[i] * 2.0
        e[i + 1] = a[i + 1] * 2.0


def test_lanes_differing_only_by_array_are_not_rerolled():
    """Two lanes whose tasklet text matches but whose ARRAYS differ are not interchangeable.

    Both lanes here read ``__out = __in1 * __in2``, so comparing lane bodies by tasklet code alone
    calls them identical and drops lane 1 -- after which every odd element of ``c`` is computed from
    ``b`` instead of ``d``.
    """
    n = 16
    rng = np.random.default_rng(31)
    a, b, d = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    expected = np.zeros(n)
    expected[0::2] = a[0::2] * b[0::2]
    expected[1::2] = a[1::2] * d[1::2]

    sdfg = unrolled_lanes_read_different_arrays.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = np.zeros(n)
    sdfg(a=a, b=b, d=d, c=got, N=n)
    assert np.allclose(got, expected), f'odd lane took the wrong source array: {got} != {expected}'


def test_lanes_writing_different_arrays_are_not_rerolled():
    """Same tasklet text and same READ array, but the lanes write different destinations.

    Collapsing onto lane 0 would leave ``e`` entirely unwritten.
    """
    n = 16
    rng = np.random.default_rng(32)
    a = rng.standard_normal(n)
    exp_c, exp_e = np.zeros(n), np.zeros(n)
    exp_c[0::2] = a[0::2] * 2.0
    exp_e[1::2] = a[1::2] * 2.0

    sdfg = unrolled_lanes_write_different_arrays.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got_c, got_e = np.zeros(n), np.zeros(n)
    sdfg(a=a, c=got_c, e=got_e, N=n)
    assert np.allclose(got_c, exp_c)
    assert np.allclose(got_e, exp_e), f'the second lane\'s destination was dropped: {got_e} != {exp_e}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
