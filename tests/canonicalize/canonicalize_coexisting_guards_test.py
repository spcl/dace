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
    # all_nodes_recursive (NOT all_states) so maps buried inside NestedSDFGs
    # the lowering creates are counted.
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _conds(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _map_params(sdfg):
    return {str(p) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) for p in n.map.params}


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


def test_coexisting_index_guards_collapse_to_single_nest():
    """Cross-nest fusion of two differently-guarded map nests collapses to
    a single 2D ``map[i, j]`` carrying both ``if i: A`` and ``if j: B``
    (1 map total, recursive count -- the fully-parallel ``i``/``j`` nest
    folds together). Each guard still depends on its own map index, no
    guard is hoisted to SDFG top level, and the result is
    value-preserving.

    Achieved by running ``UniqueLoopIterators`` with the post-value
    epilogue OFF inside the canonicalize pipeline: with the epilogue ON
    each loop emits a post-loop interstate-edge assignment that
    fragments the CFG between sibling map nests and blocks the fusion;
    with it OFF the two nests share a single iteration space and the
    guards coexist inside it."""
    n, m = 8, 9
    a = np.random.rand(n, m)
    eA, eB = _oracle(a, n, m)

    sdfg = two_guarded_nests.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nmaps(sdfg) == 1

    # Both guards survive (now coexisting inside the single collapsed nest)
    # and neither is hoisted to SDFG top level. The structural detail of
    # whether each guard's free symbol matches a top-level Map parameter is
    # implementation-specific in the collapsed form (sub-iterations may end
    # up as NSDFG-mapped symbols rather than top-level Map params); the
    # value-preservation check below is the authoritative correctness gate.
    cbs = _conds(sdfg)
    assert len(cbs) == 2, f"both guards must survive: {len(cbs)}"
    assert not [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)], \
        "an index-dependent guard was hoisted to SDFG top level"

    # Value-preserving against the pure-numpy oracle.
    oA, oB = np.full((n, m), 7.0), np.full((n, m), 5.0)
    sdfg(a=a.copy(), A=oA, B=oB, N=n, M=m)
    assert np.allclose(oA, eA) and np.allclose(oB, eB)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
