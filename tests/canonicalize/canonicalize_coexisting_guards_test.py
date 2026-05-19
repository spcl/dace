# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Two map-nests with different inner guards under canonicalization.

    ``map i: if i: map j: A`` and ``map i: map j: if j: B`` -- distinct
    index-dependent guards on the same N x M iteration space. The guards
    must survive, each still depending on its own map index (so neither can
    be hoisted past the map that defines that index), and the result must be
    value-preserving. The stated ideal is full cross-nest collapse to a
    single ``map i / map j`` containing both ``if i: A`` and ``if j: B``;
    that is not yet reached (tracked as an xfail).
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def two_guarded_nests(a: dace.float64[N, M], A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in dace.map[0:N]:
        if i % 2 == 0:  # guard depends on i
            for j in dace.map[0:M]:
                A[i, j] = a[i, j] + 1.0
    for i in dace.map[0:N]:
        for j in dace.map[0:M]:
            if j % 3 == 0:  # guard depends on j
                B[i, j] = a[i, j] * 2.0


def _nmaps(sdfg):
    return len([n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)])


def _conds(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _map_params(sdfg):
    ps = set()
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry):
                ps.update(str(p) for p in n.map.params)
    return ps


def _oracle(a, n, m):
    eA, eB = np.full((n, m), 7.0), np.full((n, m), 5.0)
    for i in range(n):
        if i % 2 == 0:
            for j in range(m):
                eA[i, j] = a[i, j] + 1.0
    for i in range(n):
        for j in range(m):
            if j % 3 == 0:
                eB[i, j] = a[i, j] * 2.0
    return eA, eB


def test_coexisting_index_guards_survive_and_are_value_preserving():
    """Both distinct guards survive, each still depending on its own map
    index; no guard is hoisted to SDFG top level; value-preserving. The map
    count is the current (non-minimal) outcome -- see the xfail below for
    the full-fusion ideal."""
    n, m = 8, 9
    a = np.random.rand(n, m)
    eA, eB = _oracle(a, n, m)

    sdfg = two_guarded_nests.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    cbs = _conds(sdfg)
    assert len(cbs) == 2, f"both guards must survive: {len(cbs)}"
    assert not [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)], \
        "an index-dependent guard was hoisted to SDFG top level"
    map_params = _map_params(sdfg)
    for cb in cbs:
        for cond, _ in cb.branches:
            if cond is not None:
                syms = {str(s) for s in cond.get_free_symbols()}
                assert syms & map_params, f"guard {syms} no longer depends on a map index"
    assert _nmaps(sdfg) == 3, f"current (non-minimal) map count changed: {_nmaps(sdfg)}"

    oA, oB = np.full((n, m), 7.0), np.full((n, m), 5.0)
    sdfg(a=a.copy(), A=oA, B=oB, N=n, M=m)
    assert np.allclose(oA, eA) and np.allclose(oB, eB)


@pytest.mark.xfail(strict=True,
                   reason="Ideal: the two nests collapse to a single map i / map j carrying "
                   "both `if i: A` and `if j: B` (2 maps). Cross-nest horizontal fusion of "
                   "differently-guarded map nests is not yet achieved (stays 3 maps).")
def test_coexisting_index_guards_collapse_to_single_nest():
    """Ideal full cross-nest fusion: one ``map i / map j`` with both guards
    coexisting -> 2 maps."""
    n, m = 8, 9
    a = np.random.rand(n, m)
    sdfg = two_guarded_nests.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
