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


@pytest.mark.xfail(strict=True,
                   reason=('The dense case re-rolls (RerollUnrolledLoops), but the indirect ``b[ip[i + k]]`` '
                           'gather lowers the loop body to TWO states (a slice state + the gather/compute '
                           'state), and the re-roll matcher currently handles only a single-state body. '
                           'Cross-state lane matching is the remaining work (CORE_BUGFIXES.md L-E). '
                           'Value-correct today.'))
def test_unrolled_indirect_becomes_map():
    n = 16
    sdfg = unrolled_indirect.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) >= 1, 'expected the re-rolled gather loop to parallelize into a map'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
