"""Interface every dace.graphlib backend implements. dace/graphlib/__init__.py dispatches
module-level calls to whichever backend a given graph object was constructed with (see
resolve.py's backend_for()).

Hard requirement, not optional: any object new_digraph()/new_multidigraph() returns MUST
correctly implement Python's copy.deepcopy protocol, including proper `memo` participation.
DaCe's IR clone machinery (SDFG.__deepcopy__ and friends) depends on that pattern pervasively --
see rustworkx_backend.RustworkxGraphHandle.__deepcopy__ for the concrete implementation this
requirement drives.
"""
from typing import Any, Dict, Iterable, List, Protocol, Set, Tuple


class GraphBackend(Protocol):
    name: str

    def new_digraph(self) -> Any:
        ...

    def new_multidigraph(self) -> Any:
        ...

    def has_path(self, G: Any, source: Any, target: Any) -> bool:
        ...

    def immediate_dominators(self, G: Any, start: Any) -> Dict[Any, Any]:
        ...

    def weakly_connected_components(self, G: Any) -> Iterable[Set[Any]]:
        ...

    def topological_sort(self, G: Any) -> Iterable[Any]:
        ...

    def simple_cycles(self, G: Any) -> Iterable[List[Any]]:
        ...

    def find_cycle(self, G: Any, source: Any = None) -> List[Tuple[Any, Any]]:
        ...

    def descendants(self, G: Any, source: Any) -> Set[Any]:
        ...

    def ancestors(self, G: Any, source: Any) -> Set[Any]:
        ...

    def all_simple_paths(self, G: Any, source: Any, target: Any) -> Iterable[List[Any]]:
        ...

    def transitive_closure(self, G: Any) -> Any:
        ...

    def transitive_closure_dag(self, G: Any) -> Any:
        ...

    def dfs_edges(self, G: Any, source: Any = None) -> Iterable[Tuple[Any, Any]]:
        ...

    def shortest_path_length(self, G: Any, source: Any, target: Any) -> int:
        ...

    def isomorphism_matcher(self, G1: Any, G2: Any, node_match=None, edge_match=None) -> Any:
        ...
