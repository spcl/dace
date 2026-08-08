# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A happens-before edge must not cost a parallel loop.

TSVC ``s1251`` is fully parallel -- every access is at ``i`` and the only dependences are
INTRA-iteration write-after-read, satisfied by statement order::

    for i in range(N):
        s = b[i] + c[i]      # reads b[i]
        b[i] = a[i] + d[i]   # overwrites b[i], reads a[i]
        a[i] = s * e[i]      # overwrites a[i]

``StateFusionExtended`` fuses the three statement states into one and keeps each read
ordered before its overwrite with EMPTY memlet edges. An empty memlet moves no data, so it
cannot carry a dependence from one iteration into the next, and the intra-iteration order it
encodes survives verbatim inside a map body. But it also has no subset, and ``LoopToMap``
used to take that missing subset for an unindexed whole-array access -- refusing a perfectly
parallel loop, which then stayed sequential for the rest of the pipeline (no vectorization,
no OpenMP).
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')


@dace.program
def intra_iteration_war_on_two_arrays(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                                      e: dace.float64[N]):
    for i in range(N):
        s = b[i] + c[i]
        b[i] = a[i] + d[i]
        a[i] = s * e[i]


def _fused(tag: str) -> dace.SDFG:
    sdfg = intra_iteration_war_on_two_arrays.to_sdfg(simplify=True)
    sdfg.name = tag
    PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
    return sdfg


def _empty_edges(sdfg: dace.SDFG):
    return [(state.label, str(edge.src), str(edge.dst)) for sd in sdfg.all_sdfgs_recursive() for state in sd.states()
            for edge in state.edges() if edge.data is None or edge.data.is_empty()]


def test_state_fusion_records_the_war_ordering():
    """Guards the premise: the fusion really does add happens-before edges here."""
    sdfg = _fused('war_two_arrays_struct')
    assert _empty_edges(sdfg), 'StateFusionExtended added no happens-before edge -- test no longer covers the shape'


def test_still_parallelizes_after_state_fusion():
    sdfg = _fused('war_two_arrays_l2m')
    applied = sdfg.apply_transformations_repeated(LoopToMap, validate=False)
    assert applied >= 1, 'LoopToMap refused a fully parallel loop because of a happens-before edge'


def test_value_preserving():
    n = 64
    rng = np.random.default_rng(11)
    a, b, c, d, e = (rng.random(n) for _ in range(5))
    want_b = a + d
    want_a = (b + c) * e

    sdfg = _fused('war_two_arrays_value')
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)
    sdfg.validate()
    got_a, got_b = a.copy(), b.copy()
    sdfg.compile()(a=got_a, b=got_b, c=copy.deepcopy(c), d=copy.deepcopy(d), e=copy.deepcopy(e), N=n)
    assert np.allclose(got_b, want_b, rtol=1e-12, atol=1e-12)
    assert np.allclose(got_a, want_a, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
