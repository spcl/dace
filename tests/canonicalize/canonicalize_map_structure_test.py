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
    return len([n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)])


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


def test_two_independent_statements_two_maps():
    """Two data-independent statements parallelize into two separate
    maps."""
    n = 20
    a, c = np.random.rand(n), np.random.rand(n)
    sdfg = two_independent.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 2
    ob, od = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=ob, c=c.copy(), d=od, N=n)
    assert np.allclose(ob, a + 1.0) and np.allclose(od, c * 3.0)


def test_producer_consumer_two_maps():
    """Producer and consumer communicate through a transient; the pipeline
    keeps them as two maps."""
    n = 18
    a = np.random.rand(n)
    sdfg = producer_consumer.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 2
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
