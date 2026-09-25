# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that patterns of two unconnected nodes match exactly as with the VF2 subgraph matcher. """
import random

import networkx as nx
import numpy as np

import dace
from dace import subsets
from dace.sdfg import nodes
from dace.transformation import transformation as xf
from dace.transformation.dataflow import MapFusionHorizontal
from dace.transformation.passes import pattern_matching as pm


def _random_digraph(rng: random.Random, num_nodes: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(num_nodes):
        kind = rng.choice(['map', 'access', 'tasklet'])
        if kind == 'map':
            node = nodes.MapEntry(nodes.Map(f'map_{i}', ['i'], subsets.Range([(0, 9, 1)])))
        elif kind == 'access':
            node = nodes.AccessNode(f'data_{i}')
        else:
            node = nodes.Tasklet(f'tasklet_{i}')
        graph.add_node(i, node=node)
    for _ in range(rng.randint(0, 3 * num_nodes)):
        src, dst = rng.randrange(num_nodes), rng.randrange(num_nodes)
        graph.add_edge(src, dst)  # includes self-edges
    return graph


def _pattern(first_type, second_type) -> nx.DiGraph:
    pattern = nx.DiGraph()
    pattern.add_node(0, node=xf.PatternNode(first_type))
    pattern.add_node(1, node=xf.PatternNode(second_type))
    return pattern


def test_same_matches_as_vf2_on_random_graphs():
    rng = random.Random(42)
    checked = 0
    for _ in range(300):
        graph = _random_digraph(rng, rng.randint(2, 40))
        for pattern in (_pattern(nodes.MapEntry, nodes.MapEntry), _pattern(nodes.MapEntry, nodes.AccessNode)):
            expected = list(pm._subgraph_isomorphism_matcher(graph, pattern, pm.type_match, None))
            actual = list(pm._unconnected_pair_matcher(graph, pattern, pm.type_match, None))
            assert actual == expected
            checked += len(expected)
    assert checked > 1000


def test_same_matches_as_vf2_on_long_path():
    """ Two nodes at distance > 2 along a chain are not adjacent, so both matchers pair them up.

        This documents that VF2 (and hence the fast matcher) matches an unconnected two-node
        pattern based on direct adjacency only, not reachability: chained map entries several
        hops apart still form a match.
    """
    # A chain map_0 -> access_1 -> map_2 -> ... -> map_8 with 5 map entries at even indices,
    # so the map entries are pairwise at distance >= 2 and never directly connected.
    graph = nx.DiGraph()
    types = [nodes.MapEntry, nodes.AccessNode] * 4 + [nodes.MapEntry]
    map_indices = [i for i, kind in enumerate(types) if kind is nodes.MapEntry]
    assert len(map_indices) == 5
    for i, kind in enumerate(types):
        if kind is nodes.MapEntry:
            node = nodes.MapEntry(nodes.Map(f'map_{i}', ['i'], subsets.Range([(0, 9, 1)])))
        else:
            node = nodes.AccessNode(f'data_{i}')
        graph.add_node(i, node=node)
    for src in range(len(types) - 1):
        graph.add_edge(src, src + 1)

    pattern = _pattern(nodes.MapEntry, nodes.MapEntry)
    expected = list(pm._subgraph_isomorphism_matcher(graph, pattern, pm.type_match, None))
    actual = list(pm._unconnected_pair_matcher(graph, pattern, pm.type_match, None))

    # VF2 pairs every ordered pair of distinct, non-adjacent map entries. None of the map
    # entries are directly connected in the chain, so all 5 * 4 ordered pairs match.
    expected_pairs = sorted((u, v) for u in map_indices for v in map_indices if u != v)
    assert sorted((u, v) for m in expected for u, v in [sorted(m)]) == \
        sorted(tuple(sorted(p)) for p in expected_pairs)
    assert len(expected) == len(map_indices) * (len(map_indices) - 1)
    assert actual == expected


def test_metadata_selects_pair_matcher():
    _, singlestate = pm.get_transformation_metadata([MapFusionHorizontal()])
    assert [matcher for _, _, _, matcher, _ in singlestate] == [pm._unconnected_pair_matcher]


def _parallel_maps_sdfg() -> dace.SDFG:

    @dace.program
    def parallel_maps(a: dace.float64[20], b: dace.float64[20], c: dace.float64[20], d: dace.float64[20]):
        for i in dace.map[0:20]:
            b[i] = a[i] + 1.0
        for i in dace.map[0:20]:
            c[i] = a[i] * 2.0
        for i in dace.map[0:10]:
            d[i] = a[i] - 1.0

    sdfg = parallel_maps.to_sdfg(simplify=True)
    return sdfg


def test_same_matches_as_vf2_on_sdfg():
    sdfg = _parallel_maps_sdfg()
    xform = MapFusionHorizontal()
    metadata = pm.get_transformation_metadata([xform])
    vf2_metadata = (metadata[0], [(x, i, p, pm._subgraph_isomorphism_matcher, o) for x, i, p, _, o in metadata[1]])

    def matched(meta):
        return [(match.state_id, sorted(match.subgraph.values()))
                for match in pm.match_patterns(sdfg, [xform], metadata=meta)]

    expected = matched(vf2_metadata)
    assert len(expected) > 0
    assert matched(metadata) == expected


def _fuse_horizontally(sdfg: dace.SDFG, force_vf2: bool):
    pipeline = pm.PatternMatchAndApplyRepeated([MapFusionHorizontal()], validate=True)
    if force_vf2:
        interstate, singlestate = pipeline._metadata
        pipeline._metadata = (interstate, [(x, i, p, pm._subgraph_isomorphism_matcher, o)
                                           for x, i, p, _, o in singlestate])
    return pipeline.apply_pass(sdfg, {})


def test_fusion_result_unchanged():
    sdfg_vf2 = _parallel_maps_sdfg()
    sdfg_pair = _parallel_maps_sdfg()

    applied_vf2 = _fuse_horizontally(sdfg_vf2, force_vf2=True)
    applied_pair = _fuse_horizontally(sdfg_pair, force_vf2=False)

    assert applied_vf2 is not None and applied_pair == applied_vf2
    assert sdfg_pair.hash_sdfg() == sdfg_vf2.hash_sdfg()

    a = np.random.rand(20)
    b, c, d = np.zeros(20), np.zeros(20), np.zeros(20)
    sdfg_pair(a=a, b=b, c=c, d=d)
    assert np.allclose(b, a + 1.0) and np.allclose(c, a * 2.0) and np.allclose(d[:10], a[:10] - 1.0)


if __name__ == '__main__':
    test_same_matches_as_vf2_on_random_graphs()
    test_same_matches_as_vf2_on_long_path()
    test_metadata_selects_pair_matcher()
    test_same_matches_as_vf2_on_sdfg()
    test_fusion_result_unchanged()
