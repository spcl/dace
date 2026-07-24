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
        return networkx.immediate_dominators(G, start)

    def weakly_connected_components(self, G):
        return networkx.weakly_connected_components(G)

    def topological_sort(self, G):
        return networkx.topological_sort(G)

    def simple_cycles(self, G):
        return networkx.simple_cycles(G)

    def find_cycle(self, G, source=None):
        return networkx.find_cycle(G, source)

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
