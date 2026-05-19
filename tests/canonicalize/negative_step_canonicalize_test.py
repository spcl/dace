# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Canonicalization of descending (negative-step) sequential loops.

    A ``range(hi, lo, -s)`` loop lowers to a ``LoopRegion``. The pipeline
    normalizes it to a ``0:trip:1`` ascending iterator with every access
    rewritten ``i -> start + step*i'``, so the traversal order is unchanged
    and the result holds for loop-carried (vertical) dependencies where the
    order is load-bearing. After canonicalization no negative-step range
    remains. References are pure-numpy oracles.

    (Negative-step ``dace.map`` parallel maps are rejected by SDFG
    validation and are out of scope here.)
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _negative_step_ranges(sdfg):
    """Map ranges / loop strides in ``sdfg`` that are provably negative."""
    bad = []
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry):
                for (_, _, s) in n.map.range:
                    if (s < 0) == True:
                        bad.append(('map', n.map.label))
    for r in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(r, LoopRegion):
            stride = loop_analysis.get_loop_stride(r)
            if stride is not None and (stride < 0) == True:
                bad.append(('loop', r.label))
    return bad


def _assert_canonical_bounds(sdfg, n: int, expected_iterations=None):
    """Every map/loop is zero-based and unit-stride; if
    ``expected_iterations`` is given, every map/loop's concrete trip count
    (at ``N == n``) equals it.

    Checks the *new bounds* explicitly -- start ``0``, step ``1``, and the
    exact iteration count after the ``i -> start + step*i'`` remap -- not
    just the end-to-end numbers.
    """
    subs = {dace.symbol('N'): n}
    seen = 0
    for st in sdfg.all_states():
        for me in st.nodes():
            if isinstance(me, nodes.MapEntry):
                for (b, e, s) in me.map.range:
                    assert dace.symbolic.evaluate(b, subs) == 0, f'map begin {b} != 0'
                    assert dace.symbolic.evaluate(s, subs) == 1, f'map step {s} != 1'
                    if expected_iterations is not None:
                        trip = int(dace.symbolic.evaluate(e, subs)) + 1  # inclusive end
                        assert trip == expected_iterations, f'map trips {trip} != {expected_iterations}'
                    seen += 1
    for r in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(r, LoopRegion):
            start = loop_analysis.get_init_assignment(r)
            stride = loop_analysis.get_loop_stride(r)
            assert dace.symbolic.evaluate(start, subs) == 0, f'loop start {start} != 0'
            assert dace.symbolic.evaluate(stride, subs) == 1, f'loop stride {stride} != 1'
            if expected_iterations is not None:
                trip = int(dace.symbolic.evaluate(loop_analysis.get_loop_end(r), subs)) + 1
                assert trip == expected_iterations, f'loop trips {trip} != {expected_iterations}'
            seen += 1
    assert seen > 0, 'no map/loop found to check bounds on'


@dace.program
def recurrence_down(a: dace.float64[N], b: dace.float64[N]):
    # b[i] reads b[i+1]: a vertical dependency, iterated top -> bottom.
    for i in range(N - 2, -1, -1):
        b[i] = b[i + 1] + a[i]


@dace.program
def recurrence_down_two(a: dace.float64[N], b: dace.float64[N]):
    # Two-deep loop-carried dependency, descending.
    for i in range(N - 3, -1, -1):
        b[i] = 0.5 * b[i + 1] + 0.25 * b[i + 2] + a[i]


@dace.program
def descending_parallel(a: dace.float64[N], b: dace.float64[N]):
    # Descending but data-parallel: canonicalization is free to reorder it.
    for i in range(N - 1, -1, -1):
        b[i] = a[i] * 2.0 + 1.0


@dace.program
def thomas_solve(lo: dace.float64[N], di: dace.float64[N], up: dace.float64[N], rhs: dace.float64[N],
                 x: dace.float64[N]):
    cp = dace.define_local([N], dace.float64)
    dp = dace.define_local([N], dace.float64)
    cp[0] = up[0] / di[0]
    dp[0] = rhs[0] / di[0]
    for i in range(1, N):  # ascending forward sweep
        m = di[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / m
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / m
    x[N - 1] = dp[N - 1]
    for i in range(N - 2, -1, -1):  # descending back substitution
        x[i] = dp[i] - cp[i] * x[i + 1]


@pytest.mark.parametrize('prog,seed_tail,recur', [
    (recurrence_down, 1, lambda b, a, i: b[i + 1] + a[i]),
    (recurrence_down_two, 2, lambda b, a, i: 0.5 * b[i + 1] + 0.25 * b[i + 2] + a[i]),
])
def test_descending_recurrence_canonicalizes_correctly(prog, seed_tail, recur):
    """A descending loop-carried recurrence canonicalizes to an ascending
    iterator, stays numerically exact, and leaves no negative-step range."""
    n = 21
    a = np.random.rand(n)
    exp = np.zeros(n)
    exp[n - 1] = 1.0
    if seed_tail == 2:
        exp[n - 2] = 0.7
    for i in range(n - 1 - seed_tail, -1, -1):
        exp[i] = recur(exp, a, i)

    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not _negative_step_ranges(sdfg), _negative_step_ranges(sdfg)
    # range(n-1-seed_tail, -1, -1) has (n - seed_tail) iterations.
    _assert_canonical_bounds(sdfg, n, expected_iterations=n - seed_tail)

    out = np.zeros(n)
    out[n - 1] = 1.0
    if seed_tail == 2:
        out[n - 2] = 0.7
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, exp)


def test_descending_parallel_loop_canonicalizes_correctly():
    n = 30
    a = np.random.rand(n)
    sdfg = descending_parallel.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not _negative_step_ranges(sdfg), _negative_step_ranges(sdfg)
    # range(n-1, -1, -1) has n iterations.
    _assert_canonical_bounds(sdfg, n, expected_iterations=n)
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, a * 2.0 + 1.0)


def test_cloudsc_style_tridiagonal_vertical_dependency():
    """Thomas algorithm (ascending forward sweep + descending back
    substitution -- the cloudsc-solver shape). The canonicalized result
    matches a dense numpy solve and contains no negative-step range."""
    n = 24
    rng = np.random.default_rng(7)
    lo = np.concatenate([[0.0], rng.uniform(-1, -0.2, n - 1)])
    up = np.concatenate([rng.uniform(-1, -0.2, n - 1), [0.0]])
    di = rng.uniform(4.0, 6.0, n)  # diagonally dominant
    rhs = rng.uniform(-1, 1, n)
    A = np.diag(di) + np.diag(lo[1:], -1) + np.diag(up[:-1], 1)
    exp = np.linalg.solve(A, rhs)

    sdfg = thomas_solve.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not _negative_step_ranges(sdfg), _negative_step_ranges(sdfg)
    # Two sweeps (forward + back-substitution): assert zero-based unit-stride
    # form (trip counts differ per loop, so not pinned to one value here).
    _assert_canonical_bounds(sdfg, n)

    out = np.zeros(n)
    sdfg(lo=lo.copy(), di=di.copy(), up=up.copy(), rhs=rhs.copy(), x=out, N=n)
    assert np.allclose(out, exp, rtol=1e-9, atol=1e-9), f"max err {np.abs(out - exp).max():.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
