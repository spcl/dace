# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Producer and consumer maps under one invariant guard fuse into one map with one guarded body.

Canonicalization pushes ``if K > 0`` into every statement loop, so a producer ``t[i] = ...`` and its consumer
``c[i] = t[i] ...`` become two maps whose bodies each hold the guard. The producer's write is conditional (a dynamic
memlet), which vertical fusion refuses unless every reader runs under a provably equal guard; then
``NormalizeMapBody`` sequences the two bodies into one nested SDFG and ``MergeGuards`` folds the guards.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes import canonicalize

N = dace.symbol("N")
K = dace.symbol("K")


@dace.program
def guarded_chain(a: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    t = np.empty_like(a)
    for i in range(N):
        if K > 0:
            t[i] = a[i] + 1.0
    for i in range(N):
        if K > 0:
            c[i] = t[i] * 2.0
    for i in range(N):
        if K > 0:
            d[i] = t[i] * 3.0


@dace.program
def differently_guarded_chain(a: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in range(N):
        if K > 0:
            t[i] = a[i] + 1.0
    for i in range(N):
        if K > 1:
            c[i] = t[i] * 2.0


def top_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None]


def nested_sdfgs(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def conditional_blocks(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock))


def test_equally_guarded_producer_fuses_into_its_readers_as_one_guarded_body():
    sdfg = guarded_chain.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    assert len(top_maps(sdfg)) == 1, "the producer and both readers share the guard: one map"
    assert nested_sdfgs(sdfg) == 1, "the fused body is one nested SDFG, not one per statement"
    assert conditional_blocks(sdfg) == 1, "the three equal guards fold into one"

    n = 16
    a = np.random.default_rng(0).random(n)
    for k in (1, 0):
        c, d = np.full(n, -1.0), np.full(n, -1.0)
        sdfg(a=a.copy(), c=c, d=d, N=n, K=k)
        assert np.allclose(c, (a + 1.0) * 2.0 if k > 0 else -1.0), f"c mismatch K={k}"
        assert np.allclose(d, (a + 1.0) * 3.0 if k > 0 else -1.0), f"d mismatch K={k}"


def test_differently_guarded_producer_stays_apart():
    sdfg = differently_guarded_chain.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    assert len(top_maps(sdfg)) == 2, "a reader under another guard may read what the conditional write skipped"

    n = 16
    a = np.random.default_rng(1).random(n)
    for k in (2, 1, 0):
        c = np.full(n, -1.0)
        sdfg(a=a.copy(), c=c, N=n, K=k)
        assert np.allclose(c, (a + 1.0) * 2.0 if k > 1 else -1.0), f"c mismatch K={k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
