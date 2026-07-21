# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" copy.deepcopy() of a dace.graphlib graph must preserve node-identity sharing across a
    memo-threaded deepcopy, under every backend -- the property SDFG.__deepcopy__'s generic
    per-attribute sweep already relies on for its `_nx` mirror. """
import copy
import importlib.util

import pytest

import dace.graphlib as gl

_BACKENDS = ['networkx'] + (['rustworkx'] if importlib.util.find_spec('rustworkx') is not None else [])


class _Node:
    """ Plain stand-in for a real DaCe node -- deepcopy must work regardless of what's stored. """

    def __init__(self, label):
        self.label = label


@pytest.mark.parametrize('backend', _BACKENDS)
def test_topology_round_trip(backend):
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        a, b, c = _Node('a'), _Node('b'), _Node('c')
        G.add_edge(a, b)
        G.add_edge(b, c)

        G2 = copy.deepcopy(G)
        assert G2.number_of_nodes() == G.number_of_nodes() == 3
        assert G2.number_of_edges() == G.number_of_edges() == 2
        a2, b2, c2 = (n for n in sorted(G2.nodes(), key=lambda n: n.label))
        assert {n.label for n in G2.nodes()} == {'a', 'b', 'c'}
        assert gl.has_path(G2, a2, c2)
        # it's a real copy, not the same objects
        assert a2 is not a and b2 is not b and c2 is not c


@pytest.mark.parametrize('backend', _BACKENDS)
def test_shared_node_reference_across_two_edges_stays_shared(backend):
    """ A node referenced from two edges' payloads must resolve to the SAME copied instance. """
    with gl.set_default_backend(backend):
        shared = _Node('shared')
        a, b = _Node('a'), _Node('b')
        G = gl.DiGraph()
        G.add_edge(a, shared, ref=shared)
        G.add_edge(shared, b, ref=shared)

        G2 = copy.deepcopy(G)
        nodes2 = {n.label: n for n in G2.nodes()}
        a2, shared2, b2 = nodes2['a'], nodes2['shared'], nodes2['b']

        assert G2[a2][shared2]['ref'] is shared2
        assert G2[shared2][b2]['ref'] is shared2
        assert G2[a2][shared2]['ref'] is G2[shared2][b2]['ref']


@pytest.mark.parametrize('backend', _BACKENDS)
def test_cross_structure_memo_sharing(backend):
    """ SDFG.__deepcopy__'s pattern: a wrapper holding a graph and a direct node ref stays consistent. """

    class Container:

        def __init__(self, graph, direct_ref):
            self.graph = graph
            self.direct_ref = direct_ref

    with gl.set_default_backend(backend):
        shared = _Node('shared')
        other = _Node('other')
        G = gl.DiGraph()
        G.add_edge(other, shared)

        container = Container(G, shared)
        container2 = copy.deepcopy(container)

        shared2 = next(n for n in container2.graph.nodes() if n.label == 'shared')
        assert container2.direct_ref is shared2


@pytest.mark.parametrize('backend', _BACKENDS)
def test_deepcopy_does_not_mutate_original(backend):
    with gl.set_default_backend(backend):
        a, b = _Node('a'), _Node('b')
        G = gl.DiGraph()
        G.add_edge(a, b, weight=1)

        G2 = copy.deepcopy(G)
        b2 = next(n for n in G2.nodes() if n.label == 'b')
        a2 = next(n for n in G2.nodes() if n.label == 'a')
        G2[a2][b2]['weight'] = 999

        assert G[a][b]['weight'] == 1


class _CountingRx:
    """Forwards to a real PyDiGraph, tallying the per-element payload fetches the deepcopy must NOT
    do. Only the methods the deepcopy touches are defined, so an unexpected fetch surfaces here."""

    def __init__(self, inner, calls):
        self.inner = inner
        self.calls = calls

    def get_node_data(self, idx):
        self.calls['get_node_data'] += 1
        return self.inner.get_node_data(idx)

    def get_edge_data(self, u, v):
        self.calls['get_edge_data'] += 1
        return self.inner.get_edge_data(u, v)

    def node_indices(self):
        return self.inner.node_indices()

    def nodes(self):
        return self.inner.nodes()

    def out_edges(self, idx):
        return self.inner.out_edges(idx)

    # Per-NODE (like out_edges), not per-edge: the deepcopy pairs each edge with its multigraph
    # key, which has to survive the rebuild.
    def out_edge_indices(self, idx):
        return self.inner.out_edge_indices(idx)


@pytest.mark.skipif(importlib.util.find_spec('rustworkx') is None, reason='rustworkx backend not installed')
def test_rustworkx_deepcopy_fetches_payloads_in_bulk():
    """Perf regression guard: the deepcopy must pull payloads in BULK, not per-node/per-edge round-trips."""
    with gl.set_default_backend('rustworkx'):
        G = gl.DiGraph()
        for i in range(200):
            G.add_node(i, k=i)
        for i in range(200):
            G.add_edge(i, (i + 1) % 200, w=i)

        calls = {'get_node_data': 0, 'get_edge_data': 0}
        real = G._rx
        G._rx = _CountingRx(real, calls)
        try:
            G2 = copy.deepcopy(G)
        finally:
            G._rx = real

        assert G2.number_of_nodes() == 200 and G2.number_of_edges() == 200
        assert calls['get_node_data'] < G.number_of_nodes(), calls  # bulk, not per-node
        assert calls['get_edge_data'] < G.number_of_edges(), calls  # bulk, not per-edge
