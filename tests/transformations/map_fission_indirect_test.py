# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" MapFission on indirect-access maps.

    The dace frontend lowers ``for i: A[i]=B[idx[i]]; C[i]=D[idx[i]]`` to a
    map whose body is one ``NestedSDFG`` whose ``idx[i]`` indirection is an
    interstate-edge symbol assignment depending on the map iterator.
    ``MapFission`` refuses to split that in place (it cannot hoist the
    assignment out of the fissioned maps). ``ConditionalComponentFission``
    replicates the NestedSDFG per independent output group first -- the
    deep-copy carries every interstate-edge indirection-symbol assignment
    and the index inputs into each clone -- after which ordinary
    ``MapFission`` splits the map into one map per independent output.

    Every test checks numerical equivalence against a deep-copied pre-pass
    run, and that the split actually happened (applied when possible) or
    provably did not (single group / coupled outputs -> still correct).
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.conditional_component_fission import ConditionalComponentFission
from dace.transformation.dataflow.map_fission import MapFission

N = dace.symbol('N')
M = dace.symbol('M')


def _nmaps(sdfg):
    return len([n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)])


@dace.program
def gather_two(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i]]
        e[i] = c[idx[i]]


@dace.program
def scatter_two(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in dace.map[0:N]:
        b[idx[i]] = a[i] * 2.0
        e[idx[i]] = c[i] + 1.0


@dace.program
def semi_indirect_two(a: dace.float64[N, M], col: dace.int32[M], b: dace.float64[N, M],
                      c: dace.float64[N, M], e: dace.float64[N, M]):
    # dim 0 structured, dim 1 gathered through ``col``.
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i, col[j]]
        e[i, j] = c[i, col[j]]


@dace.program
def gather_single(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i]]


@dace.program
def gather_coupled(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], e: dace.float64[N]):
    # Both outputs read a shared (non-input) transient -> one group: the
    # outputs are NOT independent, so no split is possible. Must stay correct.
    for i in dace.map[0:N]:
        t = a[idx[i]] * 3.0
        b[i] = t + 1.0
        e[i] = t - 1.0


def _fission(sdfg):
    """The fuse-stage recipe for indirect maps: replicate the blocking
    NestedSDFG per independent output, then MapFission."""
    ccf = ConditionalComponentFission().apply_pass(sdfg, {})
    mf = sdfg.apply_transformations_repeated(MapFission)
    return ccf, mf


def test_mapfission_splits_two_independent_gathers():
    """`b=a[idx]; e=c[idx]` -> two independent maps; value-preserving."""
    n = 20
    a = np.random.rand(n)
    c = np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)
    base = gather_two.to_sdfg(simplify=True)
    assert _nmaps(base) == 1

    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n)

    sdfg = gather_two.to_sdfg(simplify=True)
    ccf, mf = _fission(sdfg)
    assert ccf is not None, "ConditionalComponentFission must replicate the indirect NestedSDFG"
    assert mf is not None and _nmaps(sdfg) == 2, "MapFission must split into two maps"
    sdfg.validate()

    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, c=c.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    assert np.allclose(out_b, a[idx]) and np.allclose(out_e, c[idx])


def test_mapfission_splits_two_independent_scatters():
    """`b[idx]=2a; e[idx]=c+1` -> two independent maps; value-preserving."""
    n = 18
    a = np.random.rand(n)
    c = np.random.rand(n)
    idx = np.random.permutation(n).astype(np.int32)
    base = scatter_two.to_sdfg(simplify=True)

    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n)

    sdfg = scatter_two.to_sdfg(simplify=True)
    ccf, mf = _fission(sdfg)
    assert ccf is not None
    assert mf is not None and _nmaps(sdfg) == 2
    sdfg.validate()

    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, c=c.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    exp_b, exp_e = np.zeros(n), np.zeros(n)
    exp_b[idx] = a * 2.0
    exp_e[idx] = c + 1.0
    assert np.allclose(out_b, exp_b) and np.allclose(out_e, exp_e)


def test_mapfission_splits_semi_indirect():
    """Semi-indirect 2-D (structured dim 0, gathered dim 1) splits into two
    maps; value-preserving."""
    n, m = 8, 12
    a = np.random.rand(n, m)
    c = np.random.rand(n, m)
    col = np.random.randint(0, m, size=m).astype(np.int32)
    base = semi_indirect_two.to_sdfg(simplify=True)

    ref_b, ref_e = np.zeros((n, m)), np.zeros((n, m))
    copy.deepcopy(base)(a=a.copy(), col=col.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n, M=m)

    sdfg = semi_indirect_two.to_sdfg(simplify=True)
    ccf, mf = _fission(sdfg)
    assert ccf is not None
    assert mf is not None and _nmaps(sdfg) == 2
    sdfg.validate()

    out_b, out_e = np.zeros((n, m)), np.zeros((n, m))
    sdfg(a=a.copy(), col=col.copy(), b=out_b, c=c.copy(), e=out_e, N=n, M=m)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    assert np.allclose(out_b, a[:, col]) and np.allclose(out_e, c[:, col])


def test_mapfission_single_output_indirect_is_noop():
    """A single-output indirect map has nothing to fission: no-op, correct."""
    n = 14
    a = np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)
    base = gather_single.to_sdfg(simplify=True)
    ref = np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref, N=n)

    sdfg = gather_single.to_sdfg(simplify=True)
    assert ConditionalComponentFission().apply_pass(sdfg, {}) is None, "single group -> no-op"
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out, N=n)
    assert np.allclose(out, ref) and np.allclose(out, a[idx])


def test_mapfission_coupled_outputs_not_split_but_correct():
    """Outputs sharing a non-input transient are one group: no independent
    split is possible, and the result stays numerically correct."""
    n = 16
    a = np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)
    base = gather_coupled.to_sdfg(simplify=True)
    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref_b, e=ref_e, N=n)

    sdfg = gather_coupled.to_sdfg(simplify=True)
    # Coupled outputs -> a single independent group -> no replication.
    assert ConditionalComponentFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    t = a[idx] * 3.0
    assert np.allclose(out_b, t + 1.0) and np.allclose(out_e, t - 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
