# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Backend #1: literal, unwrapped passthrough to real networkx, dace.graphlib's default. Graphs
built via graphlib.DiGraph()/MultiDiGraph() here ARE plain networkx instances, so every call
below runs the same code a direct `import networkx as nx` call would.
"""
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx


class NetworkxBackend:
    name = 'networkx'

    def new_digraph(self) -> networkx.DiGraph:
        return networkx.DiGraph()

    def new_multidigraph(self) -> networkx.MultiDiGraph:
        return networkx.MultiDiGraph()

    def has_path(self, G: Any, source: Any, target: Any) -> bool:
        return networkx.has_path(G, source, target)

    def immediate_dominators(self, G: Any, start: Any) -> Dict[Any, Any]:
        return networkx.immediate_dominators(G, start)

    def weakly_connected_components(self, G: Any) -> Iterable[Set[Any]]:
        return networkx.weakly_connected_components(G)

    def topological_sort(self, G: Any) -> Iterable[Any]:
        return networkx.topological_sort(G)

    def simple_cycles(self, G: Any) -> Iterable[List[Any]]:
        return networkx.simple_cycles(G)

    def find_cycle(self, G: Any, source: Any = None) -> List[Tuple[Any, Any]]:
        return networkx.find_cycle(G, source)

    def descendants(self, G: Any, source: Any) -> Set[Any]:
        return networkx.descendants(G, source)

    def ancestors(self, G: Any, source: Any) -> Set[Any]:
        return networkx.ancestors(G, source)

    def all_simple_paths(self, G: Any, source: Any, target: Any) -> Iterable[List[Any]]:
        return networkx.all_simple_paths(G, source, target)

    def transitive_closure(self, G: Any) -> Any:
        return networkx.transitive_closure(G)

    def transitive_closure_dag(self, G: Any) -> Any:
        return networkx.transitive_closure_dag(G)

    def dfs_edges(self, G: Any, source: Any = None) -> Iterable[Tuple[Any, Any]]:
        return networkx.dfs_edges(G, source)

    def shortest_path_length(self, G: Any, source: Any, target: Any) -> int:
        return networkx.shortest_path_length(G, source, target)

    def isomorphism_matcher(self,
                            G1: Any,
                            G2: Any,
                            node_match: Optional[Callable[..., bool]] = None,
                            edge_match: Optional[Callable[..., bool]] = None) -> Any:
        return networkx.algorithms.isomorphism.DiGraphMatcher(G1, G2, node_match, edge_match)


INSTANCE = NetworkxBackend()
