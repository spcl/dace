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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
