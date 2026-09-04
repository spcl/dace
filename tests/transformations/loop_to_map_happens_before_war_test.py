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


def test_parallelizes_with_the_ordering_edge_inside_a_nested_body():
    """The happens-before edge can sit INSIDE a NestedSDFG loop body (canonicalize's map ->
    loop round trip nests the fused body), where the outer connector write is the propagated
    whole-array union. The nested write walk must skip the empty memlet exactly like the outer
    walk does -- reading it as a write refused this fully parallel loop."""
    from dace.sdfg.state import LoopRegion

    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    inner.add_symbol('N', dace.int64)
    inner.add_array('a', [N], dace.float64)
    inner.add_array('d', [N], dace.float64)
    state = inner.add_state('s', is_start_block=True)
    read_d = state.add_read('d')
    tasklet = state.add_tasklet('w', {'inp'}, {'out'}, 'out = inp + 1')
    write_a = state.add_write('a')
    state.add_edge(read_d, None, tasklet, 'inp', dace.Memlet('d[i]'))
    state.add_edge(tasklet, 'out', write_a, None, dace.Memlet('a[i]'))
    ordered_a = state.add_access('a')
    state.add_edge(read_d, None, ordered_a, None, dace.Memlet())  # happens-before, not a write

    outer = dace.SDFG('nested_ordering')
    outer.add_array('a', [N], dace.float64)
    outer.add_array('d', [N], dace.float64)
    loop = LoopRegion('l', 'i < N', 'i', 'i = 0', 'i = i + 1')
    outer.add_node(loop, is_start_block=True)
    body = loop.add_state('b', is_start_block=True)
    nsdfg = body.add_nested_sdfg(inner, {'d'}, {'a'}, symbol_mapping={'i': 'i', 'N': 'N'})
    body.add_edge(body.add_read('d'), None, nsdfg, 'd', dace.Memlet('d[0:N]'))
    body.add_edge(nsdfg, 'a', body.add_write('a'), None, dace.Memlet('a[0:N]'))
    outer.validate()

    applied = outer.apply_transformations_repeated(LoopToMap, validate=False)
    assert applied == 1, 'LoopToMap refused: the nested ordering edge was read as an unindexed write'


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
