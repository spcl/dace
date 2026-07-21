# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Differential tests for dace.graphlib: every assertion must hold identically under
    backend='networkx' and backend='rustworkx' (skipped if rustworkx is not installed). """
import copy
import importlib.util

import networkx
import pytest

import dace.graphlib as gl
from dace.graphlib.algorithms.flow import edmondskarp

_BACKENDS = ['networkx'] + (['rustworkx'] if importlib.util.find_spec('rustworkx') is not None else [])


def _diamond():
    """ a -> b -> d and a -> c -> d, plus an isolated node e. """
    G = gl.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('a', 'c')
    G.add_edge('b', 'd')
    G.add_edge('c', 'd')
    G.add_node('e')
    return G


@pytest.mark.parametrize('backend', _BACKENDS)
def test_has_path(backend):
    with gl.set_default_backend(backend):
        G = _diamond()
        assert gl.has_path(G, 'a', 'd')
        assert not gl.has_path(G, 'd', 'a')
        assert not gl.has_path(G, 'a', 'e')


@pytest.mark.parametrize('backend', _BACKENDS)
def test_has_path_trivial_self_path_matches_real_networkx(backend):
    """ networkx counts the trivial length-0 path, so has_path(G, x, x) is True; rustworkx's does not. """
    with gl.set_default_backend(backend):
        G = _diamond()
        assert gl.has_path(G, 'a', 'a')  # reachable via the trivial path, no cycle through 'a'
        assert gl.has_path(G, 'e', 'e')  # isolated node, still trivially reaches itself

        cyclic = gl.DiGraph()
        cyclic.add_edge('x', 'y')
        cyclic.add_edge('y', 'x')
        assert gl.has_path(cyclic, 'x', 'x')  # genuine cycle back to self


@pytest.mark.parametrize('backend', _BACKENDS)
def test_immediate_dominators(backend):
    with gl.set_default_backend(backend):
        G = _diamond()
        idom = gl.immediate_dominators(G, 'a')
        assert idom == {'a': 'a', 'b': 'a', 'c': 'a', 'd': 'a'}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_weakly_connected_components(backend):
    with gl.set_default_backend(backend):
        G = _diamond()
        comps = {frozenset(c) for c in gl.weakly_connected_components(G)}
        assert comps == {frozenset({'a', 'b', 'c', 'd'}), frozenset({'e'})}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_topological_sort(backend):
    with gl.set_default_backend(backend):
        G = _diamond()
        order = list(gl.topological_sort(G))
        assert order.index('a') < order.index('b') < order.index('d')
        assert order.index('a') < order.index('c') < order.index('d')


@pytest.mark.parametrize('backend', _BACKENDS)
def test_topological_sort_cycle_raises_networkx_unfeasible(backend):
    """ rustworkx raises DAGHasCycle where networkx raises NetworkXUnfeasible, which dace/library.py catches. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('b', 'a')
        with pytest.raises(gl.NetworkXUnfeasible):
            list(gl.topological_sort(G))


@pytest.mark.parametrize('backend', _BACKENDS)
def test_descendants_and_ancestors(backend):
    with gl.set_default_backend(backend):
        G = _diamond()
        assert gl.descendants(G, 'a') == {'b', 'c', 'd'}
        assert gl.ancestors(G, 'd') == {'a', 'b', 'c'}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_all_simple_paths(backend):
    with gl.set_default_backend(backend):
        G = _diamond()
        paths = {tuple(p) for p in gl.all_simple_paths(G, 'a', 'd')}
        assert paths == {('a', 'b', 'd'), ('a', 'c', 'd')}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_dfs_edges(backend):
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        assert list(gl.dfs_edges(G, 'a')) == [('a', 'b'), ('b', 'c')]


@pytest.mark.parametrize('backend', _BACKENDS)
def test_shortest_path_length(backend):
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        assert gl.shortest_path_length(G, 'a', 'c') == 2


@pytest.mark.parametrize('backend', _BACKENDS)
def test_cycles(backend):
    with gl.set_default_backend(backend):
        acyclic = _diamond()
        with pytest.raises(gl.NetworkXNoCycle):
            gl.find_cycle(acyclic)

        G = gl.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        G.add_edge('c', 'a')
        assert set(gl.find_cycle(G)) == {('a', 'b'), ('b', 'c'), ('c', 'a')}
        cycles = {frozenset(c) for c in gl.simple_cycles(G)}
        assert cycles == {frozenset({'a', 'b', 'c'})}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_transitive_closure(backend):
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('x', 'y')
        G.add_edge('y', 'z')
        closure = gl.transitive_closure(G)
        assert set(closure.edges()) == {('x', 'y'), ('y', 'z'), ('x', 'z')}
        closure_dag = gl.transitive_closure_dag(G)
        assert set(closure_dag.edges()) == {('x', 'y'), ('y', 'z'), ('x', 'z')}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_mutable_adjacency(backend):
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b', capacity=5)
        G['a']['b']['capacity'] += 10
        assert G['a']['b']['capacity'] == 15


@pytest.mark.parametrize('backend', _BACKENDS)
def test_minimum_cut_matches_networkx(backend):
    edges = [('s', 'a', 3), ('s', 'b', 2), ('a', 't', 2), ('b', 't', 3)]

    reference = networkx.DiGraph()
    for u, v, cap in edges:
        reference.add_edge(u, v, capacity=cap)
    expected_value, expected_partition = networkx.minimum_cut(reference, 's', 't')

    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        for u, v, cap in edges:
            G.add_edge(u, v, capacity=cap)
        value, partition = gl.minimum_cut(G, 's', 't', flow_func=edmondskarp.edmonds_karp)
        assert value == expected_value
        assert {frozenset(p) for p in partition} == {frozenset(p) for p in expected_partition}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_isomorphism_matches_networkx(backend):
    # host: n1(A) -> n2(B) -> n3(C); pattern: 0(A) -> 1(B). Expect one subgraph mapping.
    def node_match(a, b):
        return a['kind'] == b['kind']

    with gl.set_default_backend(backend):
        pattern = gl.DiGraph()
        pattern.add_node(0, kind='A')
        pattern.add_node(1, kind='B')
        pattern.add_edge(0, 1)

        host = gl.DiGraph()
        host.add_node('n1', kind='A')
        host.add_node('n2', kind='B')
        host.add_node('n3', kind='C')
        host.add_edge('n1', 'n2')
        host.add_edge('n2', 'n3')

        matcher = gl.isomorphism.DiGraphMatcher(host, pattern, node_match=node_match)
        mappings = list(matcher.subgraph_isomorphisms_iter())
        assert mappings == [{'n1': 0, 'n2': 1}]


@pytest.mark.skipif('rustworkx' not in _BACKENDS, reason='rustworkx not installed')
def test_unhashable_node_rustworkx_only():
    """ DaCe graphs may hold unhashable nodes; real networkx cannot, so only rustworkx is exercised here. """

    class UnhashableNode:
        __hash__ = None

        def __init__(self, label):
            self.label = label

    with gl.set_default_backend('rustworkx'):
        n1, n2 = UnhashableNode('x'), UnhashableNode('y')
        G = gl.DiGraph()
        G.add_edge(n1, n2)
        assert gl.has_path(G, n1, n2)
        assert list(G.nodes()) == [n1, n2]


def test_unhashable_node_networkx_backend_matches_real_networkx_limitation():
    """ The networkx backend is an unwrapped passthrough: it must fail exactly like raw networkx would. """

    class UnhashableNode:
        __hash__ = None

    with gl.set_default_backend('networkx'):
        G = gl.DiGraph()
        with pytest.raises(TypeError):
            G.add_edge(UnhashableNode(), UnhashableNode())


@pytest.mark.parametrize('backend', _BACKENDS)
def test_results_are_lazy_iterators_like_real_networkx(backend):
    """ These all return generators in real networkx so callers can early-break; graphlib must match. """
    with gl.set_default_backend(backend):
        G = _diamond()
        for result in (gl.dfs_edges(G, 'a'), gl.topological_sort(G), gl.all_simple_paths(G, 'a', 'd'),
                       gl.weakly_connected_components(G)):
            assert hasattr(result, '__next__'), f'{result!r} is not a lazy iterator'

        cyclic = gl.DiGraph()
        cyclic.add_edge('a', 'b')
        cyclic.add_edge('b', 'a')
        assert hasattr(gl.simple_cycles(cyclic), '__next__')


@pytest.mark.skipif('rustworkx' not in _BACKENDS, reason='rustworkx not installed')
def test_real_networkx_graph_is_lowered_to_rustworkx():
    """ No mixed backends: under backend='rustworkx', a call on a real networkx.DiGraph still runs accelerated. """
    real_g = networkx.DiGraph()
    real_g.add_edge('a', 'b')
    real_g.add_edge('b', 'c')
    real_g.add_edge('a', 'd')

    with gl.set_default_backend('rustworkx'):
        assert gl.has_path(real_g, 'a', 'c')
        assert not gl.has_path(real_g, 'c', 'a')
        assert gl.immediate_dominators(real_g, 'a') == {'a': 'a', 'b': 'a', 'c': 'b', 'd': 'a'}
        assert gl.descendants(real_g, 'a') == {'b', 'c', 'd'}
        assert list(gl.topological_sort(real_g)).index('a') == 0
        comps = {frozenset(c) for c in gl.weakly_connected_components(real_g)}
        assert comps == {frozenset({'a', 'b', 'c', 'd'})}

        # confirm it really goes through the conversion path: the coerced graph is a distinct
        # temporary handle, not the original mutated or reused.
        from dace.graphlib import rustworkx_backend
        coerced = rustworkx_backend._coerce(real_g)
        assert isinstance(coerced, rustworkx_backend.RustworkxGraphHandle)
        assert coerced is not real_g


@pytest.mark.parametrize('backend', _BACKENDS)
def test_transitive_closure_gap_never_round_trips_an_already_real_graph(backend):
    """ No rustworkx equivalent: a real networkx input runs directly on networkx and the result stays one. """
    real_g = networkx.DiGraph()
    real_g.add_edge('x', 'y')
    with gl.set_default_backend(backend):
        closure = gl.transitive_closure(real_g)
        assert isinstance(closure, networkx.Graph)


@pytest.mark.parametrize('backend', _BACKENDS)
def test_bulk_mutation_methods(backend):
    """ networkx's bulk add/remove methods, including the (item, attr-dict) forms and ignore-missing removal. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_nodes_from([1, (2, {'x': 1}), 3])
        assert set(G.nodes()) == {1, 2, 3}
        G.add_edges_from([(1, 2), (2, 3, {'w': 5})])
        assert set(G.edges()) == {(1, 2), (2, 3)}
        assert G[2][3]['w'] == 5

        G.remove_edges_from([(1, 2), (99, 100)])  # second pair doesn't exist -- silently ignored
        assert set(G.edges()) == {(2, 3)}

        G.remove_nodes_from([1, 42])  # 42 doesn't exist -- silently ignored
        assert set(G.nodes()) == {2, 3}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_instance_methods_successors_predecessors_degree_reverse(backend):
    """ performance_evaluation/helpers.py calls these as instance methods, not via module-level dispatch. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('a', 'c')
        G.add_edge('b', 'd')

        assert set(G.successors('a')) == {'b', 'c'}
        assert set(G.predecessors('d')) == {'b'}
        assert G.in_degree('a') == 0
        assert G.in_degree('d') == 1
        assert G.out_degree('a') == 2
        assert G.out_degree('d') == 0

        R = G.reverse()
        assert set(R.edges()) == {('b', 'a'), ('c', 'a'), ('d', 'b')}
        assert set(G.edges()) == {('a', 'b'), ('a', 'c'), ('b', 'd')}  # original untouched


@pytest.mark.parametrize('backend', _BACKENDS)
def test_nodes_and_edges_are_dual_property_callable_views(backend):
    """ networkx's .nodes/.edges work both bare (`for n in G.nodes:`) and called; helpers.py uses both. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_node('a', level=0)
        G.add_edge('a', 'b', w=1)

        assert set(G.nodes) == {'a', 'b'}
        assert set(G.nodes()) == {'a', 'b'}
        assert dict(G.nodes(data=True))['a'] == {'level': 0}
        assert set(G.edges) == {('a', 'b')}
        assert set(G.edges()) == {('a', 'b')}
        edges_with_data = list(G.edges(data=True))
        assert edges_with_data == [('a', 'b', {'w': 1})]
        assert len(G.nodes) == 2
        assert len(G.edges) == 1
        assert 'a' in G.nodes


@pytest.mark.parametrize('backend', _BACKENDS)
def test_get_node_attributes(backend):
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_node('a', level=0)
        G.add_node('b', level=1)
        G.add_node('c')  # no 'level' attr
        assert gl.get_node_attributes(G, 'level') == {'a': 0, 'b': 1}


@pytest.mark.parametrize('backend', _BACKENDS)
def test_missing_node_exception_types_match_real_networkx(backend):
    """ Exception TYPE must match networkx: redundant_array.py has live NodeNotFound/NetworkXError handlers. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b')

        with pytest.raises(gl.NodeNotFound):
            gl.has_path(G, 'a', 'zz')
        with pytest.raises(gl.NodeNotFound):
            gl.has_path(G, 'zz', 'a')
        with pytest.raises(gl.NetworkXError):
            gl.immediate_dominators(G, 'zz')
        with pytest.raises(gl.NetworkXError):
            gl.descendants(G, 'zz')
        with pytest.raises(gl.NetworkXError):
            gl.ancestors(G, 'zz')
        with pytest.raises(gl.NetworkXError):
            list(gl.dfs_edges(G, 'zz'))
        with pytest.raises(gl.NodeNotFound):
            gl.shortest_path_length(G, 'zz', 'a')
        with pytest.raises(gl.NodeNotFound):
            gl.shortest_path_length(G, 'a', 'zz')
        with pytest.raises(gl.NetworkXNoPath):
            gl.shortest_path_length(G, 'b', 'a')  # both nodes exist, just no path
        with pytest.raises(gl.NodeNotFound):
            list(gl.all_simple_paths(G, 'zz', 'a'))
        assert list(gl.all_simple_paths(G, 'a', 'zz')) == []  # missing TARGET: no error, empty
        # find_cycle: real networkx does NOT raise for a missing source, it just finds no cycle
        with pytest.raises(gl.NetworkXNoCycle):
            gl.find_cycle(G, 'zz')


@pytest.mark.parametrize('backend', _BACKENDS)
def test_value_equal_node_identities_deduplicate_like_real_dicts(backend):
    """ Nodes key by hash+equality, not id(): state_fusion.py rebuilds equal (0, i) tuples as one node. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_nodes_from((0, i) for i in range(3))
        G.add_nodes_from((1, i) for i in range(3))
        # each of these tuples is a FRESH object, unrelated to the ones just added above
        G.add_edge((0, 0), (1, 1))
        G.add_edge((0, 1), (1, 1))

        assert G.number_of_nodes() == 6  # not 8 -- the edge endpoints must reuse existing nodes
        comps = {frozenset(c) for c in gl.weakly_connected_components(G)}
        assert frozenset({(0, 0), (0, 1), (1, 1)}) in comps
        assert frozenset({(0, 2)}) in comps
        assert frozenset({(1, 0)}) in comps
        assert frozenset({(1, 2)}) in comps


@pytest.mark.parametrize('backend', _BACKENDS)
def test_nodes_and_edges_view_subscript(backend):
    """ G.nodes[n] / G.edges[u, v] return the attribute dict by reference; pattern_matching.py uses this. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_node('a', kind='X')
        G.add_edge('a', 'b', w=5)

        assert G.nodes['a'] == {'kind': 'X'}
        assert G.edges['a', 'b'] == {'w': 5}
        assert G.edges[('a', 'b')] == {'w': 5}

        G.nodes['a']['kind'] = 'Y'
        assert G.nodes['a'] == {'kind': 'Y'}
        G.edges['a', 'b']['w'] = 99
        assert G.edges['a', 'b']['w'] == 99


@pytest.mark.parametrize('backend', _BACKENDS)
def test_multidigraph_parallel_edges(backend):
    with gl.set_default_backend(backend):
        G = gl.MultiDiGraph()
        G.add_edge('a', 'b', label='first')
        G.add_edge('a', 'b', label='second')
        assert G.number_of_nodes() == 2
        assert gl.has_path(G, 'a', 'b')


@pytest.mark.parametrize('backend', _BACKENDS)
def test_multidigraph_edges_data_keeps_each_parallel_edge_payload(backend):
    """ edges(data=True) on a multigraph must return EACH parallel edge's own payload, not one collapsed. """
    with gl.set_default_backend(backend):
        G = gl.MultiDiGraph()
        G.add_edge('a', 'b', label='first')
        G.add_edge('a', 'b', label='second')
        G.add_edge('a', 'c', label='third')

        labels = [d['label'] for _, _, d in G.edges(data=True)]
        assert sorted(labels) == ['first', 'second', 'third']
        ab_labels = sorted(d['label'] for u, v, d in G.edges(data=True) if (u, v) == ('a', 'b'))
        assert ab_labels == ['first', 'second']


def _build_ordering_graph():
    G = gl.DiGraph()
    for n in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']:
        G.add_node(n)
    # Interleaved and NOT grouped by source at insertion time, to tell "grouped by source, then
    # insertion order" (networkx) apart from "flat global insertion order" (rustworkx).
    G.add_edge('n0', 'n2')
    G.add_edge('n3', 'n4')
    G.add_edge('n0', 'n1')
    G.add_edge('n3', 'n5')
    return G


@pytest.mark.parametrize('backend', _BACKENDS)
def test_node_and_edge_iteration_order_matches_real_networkx(backend):
    """ pattern_matching.py depends on the exact order: nodes by insertion, edges grouped by source. """
    with gl.set_default_backend('networkx'):
        reference = _build_ordering_graph()
        expected_nodes = list(reference.nodes())
        expected_edges = list(reference.edges())
        expected_successors_n0 = list(reference.successors('n0'))
        expected_out_edges_n3 = list(reference.out_edges('n3'))

    with gl.set_default_backend(backend):
        G = _build_ordering_graph()
        assert list(G.nodes()) == expected_nodes
        assert list(G.edges()) == expected_edges
        assert list(G.successors('n0')) == expected_successors_n0
        assert list(G.out_edges('n3')) == expected_out_edges_n3


@pytest.mark.parametrize('backend', _BACKENDS)
def test_edge_order_survives_remove_and_readd(backend):
    """ rustworkx recycles node/edge indices freed by removal, so order must track insertion explicitly. """
    with gl.set_default_backend('networkx'):
        reference = gl.DiGraph()
        reference.add_edge('a', 'x')
        reference.add_edge('a', 'y')
        reference.add_edge('a', 'z')
        reference.remove_edge('a', 'y')
        reference.add_edge('a', 'y')
        expected = list(reference.out_edges('a'))

    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'x')
        G.add_edge('a', 'y')
        G.add_edge('a', 'z')
        G.remove_edge('a', 'y')
        G.add_edge('a', 'y')
        assert list(G.out_edges('a')) == expected


@pytest.mark.parametrize('backend', _BACKENDS)
def test_add_edge_merges_attributes_like_real_networkx(backend):
    """ networkx's add_edge on an existing (u, v) MERGES attributes; rustworkx's replaces the payload. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b', w=1)
        G.add_edge('a', 'b', z=2)
        assert dict(G['a']['b']) == {'w': 1, 'z': 2}
        assert G.number_of_edges() == 1

        # A repeated add_edge on a MULTIgraph is a genuinely new parallel edge, not an update.
        M = gl.MultiDiGraph()
        M.add_edge('a', 'b', w=1)
        M.add_edge('a', 'b', z=2)
        assert M.number_of_edges() == 2


@pytest.mark.parametrize('backend', _BACKENDS)
def test_neighbors_matches_successors_on_derived_graph(backend):
    """ autodiff/analysis.py calls .neighbors() on a transitive_closure() result; neighbors is successors. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        closure = gl.transitive_closure(G)
        assert sorted(closure.neighbors('a')) == ['b', 'c']
        assert sorted(closure.neighbors('a')) == sorted(closure.successors('a'))


@pytest.mark.parametrize('backend', _BACKENDS)
def test_node_accessors_raise_networkx_error_for_missing_node(backend):
    """ redundant_array.py guards successors() with `except NetworkXError:`; a bare KeyError sails past it. """
    with gl.set_default_backend(backend):
        G = gl.DiGraph()
        G.add_node('a')
        for accessor in ('successors', 'predecessors', 'neighbors'):
            with pytest.raises(gl.NetworkXError):
                list(getattr(G, accessor)('missing'))


@pytest.mark.parametrize('backend', _BACKENDS)
def test_topological_sort_order_matches_real_networkx(backend):
    """ Exact order, not just validity: state_fusion.py picks out of it with `next(n for n in order if ...)`. """
    import random

    random.seed(11)
    for _ in range(25):
        size = random.randint(2, 12)
        order = list(range(size))
        random.shuffle(order)
        edges = [(order[i], order[j]) for i in range(size) for j in range(i + 1, size) if random.random() < 0.3]
        random.shuffle(edges)

        def build():
            G = gl.DiGraph()
            for node in order:
                G.add_node(node)
            for u, v in edges:
                G.add_edge(u, v)
            return G

        with gl.set_default_backend('networkx'):
            expected = list(gl.topological_sort(build()))
        with gl.set_default_backend(backend):
            assert list(gl.topological_sort(build())) == expected


@pytest.mark.parametrize('backend', _BACKENDS)
def test_topological_sort_handles_multigraph_parallel_edges(backend):
    """ in_degree counts EDGES but successors() dedups them, so a per-neighbour Kahn decrement fakes a cycle. """
    with gl.set_default_backend(backend):
        G = gl.MultiDiGraph()
        G.add_edge('a', 'b')
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        assert list(gl.topological_sort(G)) == ['a', 'b', 'c']


@pytest.mark.parametrize('backend', _BACKENDS)
def test_multigraph_key_survives_deepcopy(backend):
    """ A caller-held multigraph key must still name the same edge after a deepcopy; an edge index cannot. """

    def build(graph):
        for node in 'abcd':
            graph.add_node(node)
        key_cd = graph.add_edge('c', 'd', data='cd')
        graph.add_edge('a', 'b', data='ab')
        return key_cd

    with gl.set_default_backend('networkx'):
        reference = gl.MultiDiGraph()
        reference_key = build(reference)
        reference_copy = copy.deepcopy(reference)
        reference_copy.remove_edge('c', 'd', reference_key)
        expected = sorted(reference_copy.edges())

    with gl.set_default_backend(backend):
        G = gl.MultiDiGraph()
        key = build(G)
        H = copy.deepcopy(G)
        H.remove_edge('c', 'd', key)
        assert sorted(H.edges()) == expected == [('a', 'b')]
        # the original must be untouched by its copy's mutation
        assert sorted(G.edges()) == [('a', 'b'), ('c', 'd')]


@pytest.mark.parametrize('backend', _BACKENDS)
def test_multigraph_keys_are_per_node_pair(backend):
    """ The per-(u, v) key numbering rule is itself observable, so compare it against real networkx. """

    def build(graph):
        keys = [
            graph.add_edge('a', 'b', data='1'),
            graph.add_edge('a', 'b', data='2'),
            graph.add_edge('c', 'd', data='3'),
        ]
        graph.remove_edge('a', 'b', keys[0])
        keys.append(graph.add_edge('a', 'b', data='4'))
        return keys

    with gl.set_default_backend('networkx'):
        expected = build(gl.MultiDiGraph())

    with gl.set_default_backend(backend):
        G = gl.MultiDiGraph()
        assert build(G) == expected
        assert expected[0] != expected[1]  # two live parallel edges never share a key
        assert expected[2] == expected[0]  # ... but keys DO restart per node pair
        assert G.number_of_edges() == 3
