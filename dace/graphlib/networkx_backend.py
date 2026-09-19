"""Backend #1: literal, unwrapped passthrough to real networkx. This is dace.graphlib's
zero-risk default -- graphs built via graphlib.DiGraph()/MultiDiGraph() under this backend ARE
plain networkx.DiGraph()/MultiDiGraph() instances, so every call below runs the literal same
networkx code a direct `import networkx as nx` call would.
"""
import networkx
from networkx.algorithms.isomorphism import DiGraphMatcher


class NetworkxBackend:
    name = 'networkx'

    def new_digraph(self):
        return networkx.DiGraph()

    def new_multidigraph(self):
        return networkx.MultiDiGraph()

    def has_path(self, G, source, target):
        return networkx.has_path(G, source, target)

    def immediate_dominators(self, G, start):
        """``{block: its immediate dominator}``, with the START BLOCK MAPPED TO ITSELF.

        Normalising that entry is the whole reason this wrapper is not a bare forward. networkx
        changed the contract in 3.6: it builds ``idom = {start: None}``, computes, and then
        ``del idom[start]`` before returning, so the root is no longer a key. Every caller here
        predates that and reads the map as total over the reachable blocks -- ``cfg.py``,
        ``control_flow_raising.py``, ``analysis.py``, ``loop_detection.py`` and
        ``promote_constant_index_access.py`` all index it directly -- so on 3.6 the first lookup of
        an entry block raises ``KeyError: SDFGState (...)`` and the whole canonicalize pipeline dies
        on a graph it handled fine one release earlier.

        Restoring the entry here rather than teaching fourteen call sites to guard is the point of
        having a backend layer at all, and it keeps the rustworkx backend -- which never dropped the
        root -- and networkx answering the same question.
        """
        idom = networkx.immediate_dominators(G, start)
        idom.setdefault(start, start)
        return idom

    def weakly_connected_components(self, G):
        return networkx.weakly_connected_components(G)

    def weakly_connected_component(self, G, node):
        return networkx.node_connected_component(G.to_undirected(as_view=True), node)

    def topological_sort(self, G):
        return networkx.topological_sort(G)

    def simple_cycles(self, G):
        return networkx.simple_cycles(G)

    def find_cycle(self, G, source=None):
        return networkx.find_cycle(G, source)

    def is_directed_acyclic_graph(self, G):
        return networkx.is_directed_acyclic_graph(G)

    def descendants(self, G, source):
        return networkx.descendants(G, source)

    def ancestors(self, G, source):
        return networkx.ancestors(G, source)

    def all_simple_paths(self, G, source, target):
        return networkx.all_simple_paths(G, source, target)

    def transitive_closure(self, G):
        return networkx.transitive_closure(G)

    def transitive_closure_dag(self, G):
        return networkx.transitive_closure_dag(G)

    def dfs_edges(self, G, source=None):
        return networkx.dfs_edges(G, source)

    def shortest_path_length(self, G, source, target):
        return networkx.shortest_path_length(G, source, target)

    def isomorphism_matcher(self, G1, G2, node_match=None, edge_match=None):
        return DiGraphMatcher(G1, G2, node_match, edge_match)


INSTANCE = NetworkxBackend()
