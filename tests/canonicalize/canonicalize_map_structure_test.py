# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Structural outcome of the ``canonicalize`` pipeline: the number of
    parallel ``Map`` scopes (and, where relevant, residual sequential
    ``LoopRegion``s) after canonicalization, alongside numerical
    equivalence against pure-numpy oracles.

    Independent computations parallelize into separate maps; a loop-carried
    (vertical) dependency stays a sequential loop while an independent
    sibling becomes a map; producer/consumer through a transient stays two
    maps. Map counts are pinned to the values the pipeline produces so a
    parallelization/fission/fusion regression fails structurally, not only
    numerically.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


def _nmaps(sdfg):
    # all_nodes_recursive so maps inside NestedSDFGs are counted too.
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _nloops(sdfg):
    return len([r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)])


@dace.program
def elemwise(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0 + 1.0


@dace.program
def two_independent(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
        d[i] = c[i] * 3.0


@dace.program
def producer_consumer(a: dace.float64[N], b: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = t[i] + 1.0


@dace.program
def stencil1d(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[1:N - 1]:
        b[i] = a[i - 1] + a[i] + a[i + 1]


@dace.program
def jacobi2d(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[1:N - 1, 1:M - 1]:
        b[i, j] = 0.25 * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


@dace.program
def dependent_plus_independent(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]  # loop-carried recurrence (sequential)
        d[i] = c[i] * 2.0  # independent (parallel)


def test_elementwise_single_map():
    n = 24
    a = np.random.rand(n)
    sdfg = elemwise.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, a * 2.0 + 1.0)


def test_two_independent_statements_single_map():
    """Two data-independent statements over the same iteration space fuse
    into a single map: each statement becomes a sibling component in the
    map body. (Earlier the pipeline left them as two separate maps; with
    the ``UniqueLoopIterators`` post-value epilogue off the
    inter-statement boundary state no longer carries a fragmenting
    interstate-edge assignment, and the maps fuse.)"""
    n = 20
    a, c = np.random.rand(n), np.random.rand(n)
    sdfg = two_independent.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1
    ob, od = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=ob, c=c.copy(), d=od, N=n)
    assert np.allclose(ob, a + 1.0) and np.allclose(od, c * 3.0)


def test_producer_consumer_single_map():
    """Producer and consumer over the same iteration space, communicating
    through a transient, fuse into a single map. (Earlier the pipeline
    kept them as two maps; with the post-value epilogue off the producer's
    closing state no longer separates them with an interstate-edge
    assignment, enabling vertical fusion.)"""
    n = 18
    a = np.random.rand(n)
    sdfg = producer_consumer.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, a * 2.0 + 1.0)


def test_stencil_single_map():
    n = 32
    a = np.random.rand(n)
    sdfg = stencil1d.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    exp = np.zeros(n)
    exp[1:n - 1] = a[0:n - 2] + a[1:n - 1] + a[2:n]
    assert np.allclose(out, exp)


def test_jacobi2d_single_map():
    n, m = 16, 12
    a = np.random.rand(n, m)
    sdfg = jacobi2d.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1
    out = np.zeros((n, m))
    sdfg(a=a.copy(), b=out, N=n, M=m)
    exp = np.zeros((n, m))
    exp[1:n - 1, 1:m - 1] = 0.25 * (a[0:n - 2, 1:m - 1] + a[2:n, 1:m - 1] + a[1:n - 1, 0:m - 2] + a[1:n - 1, 2:m])
    assert np.allclose(out, exp)


def test_dependency_aware_split_one_map_one_loop():
    """A loop-carried recurrence and an independent statement in the same
    loop split: the recurrence stays a sequential ``LoopRegion`` while the
    independent statement becomes a parallel ``Map`` (one map, one loop)."""
    n = 25
    a, c = np.random.rand(n), np.random.rand(n)
    sdfg = dependent_plus_independent.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1, f'expected the independent part as one map, got {_nmaps(sdfg)}'
    assert _nloops(sdfg) >= 1, 'the carried recurrence must remain a sequential loop'

    eb = np.zeros(n)
    eb[0] = 1.0
    for i in range(1, n):
        eb[i] = eb[i - 1] + a[i]
    ed = np.zeros(n)
    for i in range(1, n):
        ed[i] = c[i] * 2.0

    ob, od = np.zeros(n), np.zeros(n)
    ob[0] = 1.0
    sdfg(a=a.copy(), b=ob, c=c.copy(), d=od, N=n)
    assert np.allclose(ob, eb) and np.allclose(od, ed)


def _count_promoted_arith_symbols(sdfg, base: str) -> int:
    """Count symbols whose name matches ``<base>_plus_<digits>`` / ``<base>_minus_<digits>``
    (the Python frontend's naming heuristic for promoted ``base + k`` / ``base - k``
    expressions). The post-value epilogue used to materialise a fresh
    instance per loop, accumulating per canonicalize call; with the epilogue
    off the count must stay bounded by the number of distinct arithmetic
    bounds the source program actually uses."""
    import re
    pat = re.compile(rf'^{re.escape(base)}_(plus|minus)_\d+(_\d+)?$')
    syms = set(sdfg.symbols.keys())
    for n, _ in sdfg.all_nodes_recursive():
        if hasattr(n, 'sdfg') and n.sdfg is not None and n.sdfg is not sdfg:
            syms |= set(n.sdfg.symbols.keys())
    return sum(1 for s in syms if pat.match(s))


@dace.program
def mixed_direct_indirect_stencil(a: dace.float64[N, M], idx: dace.int32[N], b: dace.float64[N, M]):
    """Mixed direct stencil + indirect gather over the same iteration: the
    ``a[i, j]`` and ``a[i, j-1]`` reads are direct (regular stencil), while
    ``a[idx[i], j]`` is an indirect gather through ``idx``. Canonicalize
    must preserve both access shapes and not promote unrelated subexpressions
    into duplicate symbols."""
    for i, j in dace.map[1:N - 1, 1:M - 1]:
        b[i, j] = 0.25 * (a[i, j] + a[i, j - 1] + a[i, j + 1] + a[idx[i], j])


def test_mixed_direct_indirect_stencil_value_preserving():
    n, m = 12, 9
    a = np.random.rand(n, m)
    idx = np.random.randint(0, n, size=n).astype(np.int32)
    sdfg = mixed_direct_indirect_stencil.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1
    out = np.zeros((n, m))
    sdfg(a=a.copy(), idx=idx.copy(), b=out, N=n, M=m)
    exp = np.zeros((n, m))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            exp[i, j] = 0.25 * (a[i, j] + a[i, j - 1] + a[i, j + 1] + a[idx[i], j])
    assert np.allclose(out, exp)


@dace.program
def two_ranges_same_arith_bound(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Two loops whose ranges share an arithmetic upper bound ``N + 1``
    (clipped by the array shape). The Python frontend promotes ``N + 1`` to
    an auxiliary symbol (``N_plus_1`` style). With the postamble OFF inside
    canonicalize, that promotion happens once and is reused, not duplicated
    per loop -- ``_count_promoted_arith_symbols(sdfg, 'N')`` must stay
    small."""
    for i in dace.map[0:N - 1]:
        b[i] = a[i] + a[i + 1]
    for i in dace.map[1:N]:
        c[i] = a[i] - a[i - 1]


def test_two_ranges_share_arith_bound_no_symbol_duplication():
    n = 16
    a = np.random.rand(n)
    sdfg = two_ranges_same_arith_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # Both loops should fuse to a single Map (knob-off enables this) or stay
    # as two maps if dependency analysis keeps them apart; either is OK -- the
    # test's structural assertion is on symbol count, not map count.
    assert _nmaps(sdfg) in (1, 2)
    # Bound by source-program arithmetic distinct expressions (``N - 1``,
    # ``N``): at most a handful of promoted ``N_*`` symbols.
    assert _count_promoted_arith_symbols(sdfg, 'N') <= 4, \
        f"unexpected duplication of N_plus/minus symbols: {sorted(sdfg.symbols)}"
    ob, oc = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=ob, c=oc, N=n)
    exp_b, exp_c = np.zeros(n), np.zeros(n)
    for i in range(0, n - 1):
        exp_b[i] = a[i] + a[i + 1]
    for i in range(1, n):
        exp_c[i] = a[i] - a[i - 1]
    assert np.allclose(ob, exp_b) and np.allclose(oc, exp_c)


@dace.program
def guarded_arith_bound(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1]):
    """Guard whose body uses a loop with an arithmetic-bound range
    ``range(0, N - 1)``. Canonicalize pushes the guard into the loop and
    must keep the promoted ``N - 1`` symbol clean -- no duplicate per
    canonicalize iteration, no leaked free symbol on the resulting
    NestedSDFG."""
    if c[0] > 0:
        for i in range(0, N - 1):
            b[i] = a[i] + a[i + 1]


def test_guarded_arith_bound_clean_after_canonicalize():
    n = 10
    a = np.random.rand(n)
    sdfg = guarded_arith_bound.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # Validates after canonicalize: any leaked-symbol leak (e.g. an
    # arith-bound symbol that the guard's NestedSDFG references but the
    # outer mapping does not provide) would fail validation here.
    sdfg.validate()
    # Re-running canonicalize must reach a fixed point in symbol count: no
    # additional arith-bound symbol minted on each pass.
    n_minus_before = _count_promoted_arith_symbols(sdfg, 'N')
    canonicalize(sdfg, validate=True)
    assert _count_promoted_arith_symbols(sdfg, 'N') == n_minus_before, \
        "canonicalize is not idempotent on N_minus symbol count"

    for cv in (1, 0):
        sdfg_run = guarded_arith_bound.to_sdfg(simplify=True)
        canonicalize(sdfg_run, validate=True)
        ob = np.zeros(n)
        sdfg_run(a=a.copy(), b=ob, c=np.array([cv], np.int32), N=n)
        exp = np.zeros(n)
        if cv > 0:
            for i in range(0, n - 1):
                exp[i] = a[i] + a[i + 1]
        assert np.allclose(ob, exp), f"value mismatch for c={cv}"


@dace.program
def stencil_reduction_mixed(a: dace.float64[N, M], b: dace.float64[N]):
    """Per-row reduction with stencil-style neighbour accesses: each row
    independently sums ``a[i, 1:M-1]`` plus the boundary contributions.
    Mixes per-iteration map-style outer with reduction-style inner."""
    for i in dace.map[0:N]:
        s = a[i, 0] + a[i, M - 1]
        for j in range(1, M - 1):
            s += a[i, j - 1] + a[i, j] + a[i, j + 1]
        b[i] = s


@pytest.mark.xfail(strict=True,
                   reason="Inner-accumulator reduction post-canonicalize produces slightly different "
                   "values (per-row reduction semantics drift through canonicalize's NSDFG / "
                   "interstate-edge remixing). Same family as the deferred "
                   "CascadeInterstateEdgeAssignmentsUp design; the j-leak validation issue is "
                   "fixed (the SDFG now validates and compiles cleanly) but the reduction's "
                   "numerical equivalence is not yet preserved.")
def test_stencil_reduction_mixed_value_preserving():
    n, m = 8, 11
    a = np.random.rand(n, m)
    sdfg = stencil_reduction_mixed.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1
    # No duplicated ``M_*`` symbols even after canonicalize promoted
    # ``M - 1`` for the inner loop bound.
    assert _count_promoted_arith_symbols(sdfg, 'M') <= 2, \
        f"unexpected M_plus/minus duplication: {sorted(sdfg.symbols)}"
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n, M=m)
    exp = np.zeros(n)
    for i in range(n):
        s = a[i, 0] + a[i, m - 1]
        for j in range(1, m - 1):
            s += a[i, j - 1] + a[i, j] + a[i, j + 1]
        exp[i] = s
    assert np.allclose(out, exp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
