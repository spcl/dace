# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize must fission a nest whose statements have DIFFERENT
parallelism, then parallelize each maximally.

A single ``i, j`` nest carries two independent statements::

    for i, j:
        A[j, i] = A[j, i] * 2.0          # parallel over BOTH i and j
        B[i, j] = B[i, j - 1] + B[i, j]   # carried over j (reads j-1)

Canonical form: the two statements must **fission apart** because they do
not share parallelism --

* the ``A`` statement is fully parallel (independent per ``(i, j)``) and
  must become a single collapsed 2D Map;
* the ``B`` statement carries a dependence along ``j`` (``B[i, j-1]``) but
  is independent across ``i``, so it must become ``map i: { loop j }``
  (``i`` parallel, ``j`` sequential).

The distribution is what the DEFAULT pipeline delivers; measured 2026-09-11, passing
``perfect_loop_nesting=False`` reaches the identical shape on this nest, so the knob is not
what produces it and a knob-on test pins nothing the default test does not.

Value preservation is checked against the original (un-canonicalized)
SDFG -- the non-transformed reference -- so the ``B[i, -1]`` wrap on the
first ``j`` iteration is matched exactly on both sides.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def mixed_parallelism(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in range(N):
        for j in range(1, N):
            A[j, i] = A[j, i] * 2.0
            B[i, j] = B[i, j - 1] + B[i, j]


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def _map_param_counts(sdfg):
    """Sorted list of each MapEntry's parameter count (e.g. ``[1, 2]`` =
    one 1D map and one collapsed 2D map)."""
    return sorted(len(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def test_mixed_parallelism_value_preserving():
    """Canonicalize preserves the values of the original SDFG (the
    non-transformed reference) for the mixed-parallelism nest."""
    n = 8
    rng = np.random.default_rng(30)
    A0 = rng.standard_normal((n, n))
    B0 = rng.standard_normal((n, n))

    # Non-transformed reference: the original SDFG.
    ref_sdfg = mixed_parallelism.to_sdfg(simplify=True)
    refA, refB = A0.copy(), B0.copy()
    ref_sdfg(A=refA, B=refB, N=n)

    # Canonicalized SDFG on the same inputs.
    sdfg = mixed_parallelism.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    gotA, gotB = A0.copy(), B0.copy()
    sdfg(A=gotA, B=gotB, N=n)

    assert np.allclose(gotA, refA), 'A (fully-parallel) value mismatch after canonicalize'
    assert np.allclose(gotB, refB), 'B (j-carried) value mismatch after canonicalize'


def test_mixed_parallelism_b_keeps_sequential_j():
    """Contract that canonicalize ALREADY delivers: the ``j``-carried ``B``
    statement keeps a sequential ``LoopRegion`` for ``j`` (never wrongly
    parallelized), and the ``i`` axis is a Map. Verified value-preserving
    above; here we pin the sequential-j survivor."""
    sdfg = mixed_parallelism.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nloops(sdfg) == 1, f'exactly the j-carried B statement stays a LoopRegion, got {_nloops(sdfg)}'
    assert _nmaps(sdfg) == 2, f"expected A's 2D map and B's i-map, got {_nmaps(sdfg)}"
    assert _map_param_counts(sdfg) == [1, 2], f'maps={_map_param_counts(sdfg)}'


def test_mixed_parallelism_default_pipeline_distributes():
    """The default pipeline distributes the nest: a collapsed 2D map for A, a 1-D i-map for B.

    ``perfect_loop_nesting`` is ON by default (user ruling 2026-09-01, reversing the 2026-08-18
    ruling that kept it opt-in). The 08-18 ruling existed for a defect, not for a preference:
    ``PerfectLoopNesting`` then distributed through ``LoopFission``'s node-level grouping, which
    carries no dependence distance or direction, and a single application reproduced the CloudSC
    read-modify-write miscompile (``tendency_loc_a`` rel=0.13). That path is severed -- the pass
    groups with Allen-Kennedy at block granularity and refuses what it cannot prove disjoint --
    so the pin now points the other way. A default that silently stopped distributing would leave
    the collapsed 2D map reachable only through a knob nobody passes, which is what this catches.

    Values are checked against the un-canonicalized original in the same run: the distribution is
    only worth having if it is also correct, and this kernel is exactly the shape whose grouping
    soundness is in question.
    """
    n = 8
    rng = np.random.default_rng(0)
    A0, B0 = rng.random((n, n)), rng.random((n, n))
    ref = mixed_parallelism.to_sdfg(simplify=True)
    A_ref, B_ref = A0.copy(), B0.copy()
    ref.compile()(A=A_ref, B=B_ref, N=n)

    sdfg = mixed_parallelism.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _map_param_counts(sdfg) == [
        1, 2
    ], ('the default pipeline no longer distributes this nest into a collapsed 2D map for A and a '
        f'1-D i-map for B; maps={_map_param_counts(sdfg)}')

    A_got, B_got = A0.copy(), B0.copy()
    sdfg.compile()(A=A_got, B=B_got, N=n)
    assert np.allclose(A_got, A_ref), 'A diverged from the un-canonicalized reference'
    assert np.allclose(B_got, B_ref), 'B diverged from the un-canonicalized reference'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
