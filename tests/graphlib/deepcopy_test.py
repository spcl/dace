# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" copy.deepcopy() of a dace.graphlib graph must keep working correctly under every backend --
    not just "doesn't crash", but preserve node-identity-sharing across a memo-threaded
    deepcopy. dace.sdfg.sdfg.SDFG.__deepcopy__ already depends on exactly that property today
    for its real-networkx `_nx` mirror (a generic per-attribute `copy.deepcopy(v, memo)` sweep
    that does NOT special-case `_nx`); this suite proves dace.graphlib graphs are equally
    well-behaved citizens of Python's deepcopy protocol, so any future code that embeds one in a
    larger structure (the same shape SDFG.__deepcopy__ already relies on) stays correct too. """
import copy
import importlib.util

import pytest

import dace.graphlib as gl

_BACKENDS = ['networkx'] + (['rustworkx'] if importlib.util.find_spec('rustworkx') is not None else [])


class _Node:
    """ A plain, non-dace object standing in for a real (possibly unhashable) DaCe node --
        deepcopy must work the same way regardless of what's stored. """

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
    """ The exact property SDFG.__deepcopy__ relies on for its real-networkx `_nx` mirror: a
        node referenced from two different edges' payloads must resolve to the SAME copied
        instance after deepcopy, not two independent copies. """
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
    """ Mimics SDFG.__deepcopy__'s own pattern: a wrapper object holding both a graphlib graph
        AND a direct reference to one of its nodes. A plain copy.deepcopy() of the wrapper must
        keep that reference pointing at the SAME copied node the graph itself now holds -- this
        is what makes it safe to embed a graphlib graph as a plain attribute inside any larger
        structure that relies on Python's generic per-attribute deepcopy sweep (as
        SDFG.__deepcopy__ does for `_nx` today, and as a possible future backend-aware `.nx`
        follow-up would need for graphlib-backed graphs too). """

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
    """Forwards to a real rustworkx PyDiGraph, tallying the two per-element payload fetches the
    deepcopy must NOT do per node/edge. Only the read methods the deepcopy touches on the SOURCE
    graph are defined (no getattr passthrough) -- an unexpected fetch would surface here."""

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
    # key, which has to survive the rebuild -- see test_multigraph_key_survives_deepcopy.
    def out_edge_indices(self, idx):
        return self.inner.out_edge_indices(idx)


@pytest.mark.skipif(importlib.util.find_spec('rustworkx') is None, reason='rustworkx backend not installed')
def test_rustworkx_deepcopy_fetches_payloads_in_bulk():
    """Perf regression guard: the rustworkx deepcopy must pull node/edge payloads in BULK (one
    rx.nodes()/rx.node_indices() plus the per-source out_edges it already needs), NOT via a per-node
    get_node_data / per-edge get_edge_data round-trip into rust -- the redundant-hop antipattern that
    made it slower than plain networkx. Deterministic (call-count, not wall-clock), so it stays
    stable under load: assert the per-element fetches stay well below O(V)/O(E)."""
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
