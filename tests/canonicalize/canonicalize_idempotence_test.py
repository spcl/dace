# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalizing an already-canonical SDFG must not undo its parallelization.

The recipe lowers every Map back to a LoopRegion at its ``lower`` stage and re-parallelizes at
``parallelize``, so a second run is a full round trip through the loop form. Anything that
widened a memlet on the way through -- a scope summary recomputed against a body that indexes
absolutely, say -- shows up here as a map that does not come back, and nowhere else: the first
run still looks perfect.

The vectorizer runs ``canonicalize`` at its OWN entry, so a caller that canonicalizes first
takes exactly this path on every kernel.
"""
import dace
import pytest

from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def guarded_elementwise(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """A guard the body needs for its range: ``b[i + 1]`` runs off the end at ``i = N - 1``."""
    for i in range(N):
        if i + 1 < N:
            a[i] = a[i] + b[i + 1] * c[i]


@dace.program
def nested_rows(a: dace.float64[M, N], b: dace.float64[M, N]):
    for j in range(M):
        for i in range(N):
            a[j, i] = a[j, i] + b[j, i] * 2.0


def map_params(sdfg):
    return sorted(tuple(m.map.params) for m, _ in sdfg.all_nodes_recursive() if isinstance(m, nd.MapEntry))


@pytest.mark.parametrize('program', [guarded_elementwise, nested_rows], ids=['guarded', 'nested'])
def test_second_canonicalize_keeps_every_map(program):
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    first = map_params(sdfg)
    assert first, 'fixture stopped producing a map -- the round trip is no longer under test'

    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    assert map_params(sdfg) == first, 'a second canonicalize dropped a map the first one had proven parallel'


def test_second_canonicalize_leaves_no_growing_subset():
    """No memlet inside a map scope may span ``0`` up to the map's own parameter.

    ``a[0:i + 1]`` says iteration ``i`` touches everything written so far -- a triangular write.
    It is a sound over-approximation of the single element ``a[i]``, which is why it survives
    validation, and it is exactly what makes the next ``LoopToMap`` refuse the loop. Pinned as a
    shape, not as a specific subset string, so it catches the pathology wherever it appears.
    """
    sdfg = guarded_elementwise.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)

    checked = 0
    for entry, state in sdfg.all_nodes_recursive():
        if not isinstance(entry, nd.MapEntry):
            continue
        params = {dace.symbolic.symbol(p) for p in entry.map.params}
        for node in state.scope_subgraph(entry, include_entry=True, include_exit=True).nodes():
            for edge in state.all_edges(node):
                if edge.data is None or edge.data.subset is None:
                    continue
                checked += 1
                for begin, end, _ in edge.data.subset.ndrange():
                    growing = begin == 0 and (dace.symbolic.pystr_to_symbolic(str(end)).free_symbols & params)
                    assert not growing, (f'{edge.src} -> {edge.dst}: subset {edge.data.subset} grows with the '
                                         f'map parameter instead of naming one element')
    assert checked, 'no in-scope memlet was inspected -- fixture no longer covers the case'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
