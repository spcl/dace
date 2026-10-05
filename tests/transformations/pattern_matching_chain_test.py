# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The chain matcher (a pattern that is one directed path, e.g. map fusion's ``MapExit -> AccessNode -> MapEntry``)
finds exactly VF2's induced matches, in graph order rather than in the networkx version's VF2 order."""
import dace
from dace.transformation.dataflow import MapFusionVertical
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes import pattern_matching as pm
from dace.transformation.passes.pattern_matching import collapse_multigraph_to_nx, type_match

N = dace.symbol('N')


@dace.program
def stencil_chain(A: dace.float64[N, N], B: dace.float64[N, N]):
    for _ in range(3):
        B[1:-1, 1:-1] = 0.25 * (A[:-2, 1:-1] + A[2:, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:])
        A[1:-1, 1:-1] = 0.25 * (B[:-2, 1:-1] + B[2:, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:])


def match_keys(matches):
    return [tuple(sorted((str(pattern), id(node)) for node, pattern in match.items())) for match in matches]


def test_the_chain_matcher_finds_exactly_the_vf2_matches():
    sdfg = stencil_chain.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated([LoopToMap])
    sdfg.simplify()
    checked = 0
    for expr in MapFusionVertical.expressions():
        pattern = collapse_multigraph_to_nx(expr)
        assert pm.chain_order(pattern) is not None
        for state in sdfg.states():
            graph = collapse_multigraph_to_nx(state)
            vf2 = match_keys(pm._subgraph_isomorphism_matcher(graph, pattern, type_match, None))
            chain = match_keys(pm._chain_matcher(graph, pattern, type_match, None))
            assert sorted(vf2) == sorted(chain)
            checked += len(chain)
    assert checked > 0, 'no fusion candidate in the fixture, so the comparison proves nothing'


def test_the_chain_matcher_yields_in_graph_order():
    """Start nodes come in the state's insertion order, whatever networkx's VF2 would pick first."""
    sdfg = stencil_chain.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated([LoopToMap])
    sdfg.simplify()
    pattern = collapse_multigraph_to_nx(MapFusionVertical.expressions()[0])
    start = pm.chain_order(pattern)[0]
    for state in sdfg.states():
        graph = collapse_multigraph_to_nx(state)
        starts = [
            next(n for n, p in match.items() if p == start)
            for match in pm._chain_matcher(graph, pattern, type_match, None)
        ]
        positions = {n: i for i, n in enumerate(graph)}
        assert [positions[n] for n in starts] == sorted(positions[n] for n in starts)


if __name__ == '__main__':
    test_the_chain_matcher_finds_exactly_the_vf2_matches()
    test_the_chain_matcher_yields_in_graph_order()
