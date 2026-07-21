# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""dace.graphlib -- networkx-compatible graph layer with a config-selectable backend
(Config.get('graph', 'backend'): 'networkx', the default passthrough, or 'rustworkx', an
accelerated opt-in). Migration target for `import networkx as nx`: `from dace import graphlib as nx`.

No mixed backends. Graphs built via DiGraph()/MultiDiGraph() remember the backend that produced
them (bound at construction, see resolve.py); real networkx graphs (an SDFG/state's .nx/._nx)
run on whichever backend is currently configured, converted on the fly. DaCe's IR storage itself
is always real, unwrapped networkx, so raw adjacency access that bypasses graphlib.* is
unaffected either way. Transitive closure and max-flow/min-cut always run on real networkx --
rustworkx has neither, so there is nothing to lower to.
"""
import contextlib
import os
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import networkx

import dace.config
import dace.graphlib.resolve as resolve
import dace.graphlib.rustworkx_backend as rustworkx_backend
from dace.graphlib import isomorphism
from dace.graphlib.algorithms.flow import edmondskarp

# Re-exported, so `except nx.NetworkXNoCycle:` call sites keep working after the import swap from
# networkx to dace.graphlib. Aliases rather than `from networkx.exception import ...`, per
# CONTRIBUTING.md's ban on importing classes directly.
NetworkXError = networkx.NetworkXError
NetworkXNoCycle = networkx.NetworkXNoCycle
NetworkXNoPath = networkx.NetworkXNoPath
NetworkXUnfeasible = networkx.NetworkXUnfeasible
NodeNotFound = networkx.NodeNotFound

__all__ = [
    'DiGraph', 'MultiDiGraph', 'has_path', 'immediate_dominators', 'weakly_connected_components', 'topological_sort',
    'simple_cycles', 'find_cycle', 'descendants', 'ancestors', 'all_simple_paths', 'transitive_closure',
    'transitive_closure_dag', 'dfs_edges', 'shortest_path_length', 'minimum_cut', 'get_node_attributes', 'isomorphism',
    'NetworkXError', 'NetworkXNoCycle', 'NetworkXNoPath', 'NetworkXUnfeasible', 'NodeNotFound', 'set_default_backend',
    'get_backend_name'
]


def DiGraph() -> Any:
    return resolve.resolve_backend().new_digraph()


def MultiDiGraph() -> Any:
    return resolve.resolve_backend().new_multidigraph()


def has_path(G: Any, source: Any, target: Any) -> bool:
    return resolve.backend_for(G).has_path(G, source, target)


def immediate_dominators(G: Any, start: Any) -> Dict[Any, Any]:
    return resolve.backend_for(G).immediate_dominators(G, start)


def weakly_connected_components(G: Any) -> Iterable[Set[Any]]:
    return resolve.backend_for(G).weakly_connected_components(G)


def topological_sort(G: Any) -> Iterable[Any]:
    return resolve.backend_for(G).topological_sort(G)


def simple_cycles(G: Any) -> Iterable[List[Any]]:
    return resolve.backend_for(G).simple_cycles(G)


def find_cycle(G: Any, source: Any = None) -> List[Tuple[Any, Any]]:
    return resolve.backend_for(G).find_cycle(G, source)


def descendants(G: Any, source: Any) -> Set[Any]:
    return resolve.backend_for(G).descendants(G, source)


def ancestors(G: Any, source: Any) -> Set[Any]:
    return resolve.backend_for(G).ancestors(G, source)


def all_simple_paths(G: Any, source: Any, target: Any) -> Iterable[List[Any]]:
    return resolve.backend_for(G).all_simple_paths(G, source, target)


def transitive_closure(G: Any) -> Any:
    return resolve.backend_for(G).transitive_closure(G)


def transitive_closure_dag(G: Any) -> Any:
    return resolve.backend_for(G).transitive_closure_dag(G)


def dfs_edges(G: Any, source: Any = None) -> Iterable[Tuple[Any, Any]]:
    return resolve.backend_for(G).dfs_edges(G, source)


def shortest_path_length(G: Any, source: Any, target: Any) -> int:
    return resolve.backend_for(G).shortest_path_length(G, source, target)


def get_node_attributes(G: Any, name: str, default: Any = None) -> Dict[Any, Any]:
    """
    Plain attribute-dict reader, not an accelerated algorithm -- works on either a real networkx
    graph or a RustworkxGraphHandle, via each one's own node-payload access.

    :param G: The graph to read from.
    :param name: The node attribute to read.
    :param default: Value for nodes lacking the attribute; None omits them, as in networkx.
    :return: {node: attribute value}.
    """
    if isinstance(G, rustworkx_backend.RustworkxGraphHandle):
        result = {}
        for node, attrs in G.nodes_with_payload():
            # networkx omits nodes lacking the attribute only when no default is given.
            if name in attrs:
                result[node] = attrs[name]
            elif default is not None:
                result[node] = default
        return result
    return networkx.get_node_attributes(G, name, default)


def minimum_cut(G: Any,
                s: Any,
                t: Any,
                capacity: str = 'capacity',
                flow_func: Optional[Callable[..., Any]] = None,
                **kwargs: Any) -> Tuple[Any, Tuple[Set[Any], Set[Any]]]:
    """
    Always real networkx internally -- rustworkx has no directed s-t max-flow/min-cut.

    :param G: The graph to cut.
    :param s: Source node.
    :param t: Sink node.
    :param capacity: Edge attribute holding each edge's capacity.
    :param flow_func: Max-flow implementation to use, or None for networkx's default.
    :param kwargs: Forwarded to ``networkx.minimum_cut``.
    :return: (cut value, (source-side partition, sink-side partition)).
    """
    if isinstance(G, rustworkx_backend.RustworkxGraphHandle):
        G = rustworkx_backend.to_networkx(G)
    return networkx.minimum_cut(G, s, t, capacity=capacity, flow_func=flow_func, **kwargs)


@contextlib.contextmanager
def set_default_backend(name: str) -> Iterator[None]:
    """
    dace.config.set_temporary plus a matching os.environ override: Config.get() checks
    DACE_graph_backend BEFORE its in-memory value, so set_temporary alone is a silent no-op
    whenever that env var is already set (CI exports it for the whole pytest run). Affects graphs
    constructed inside the context, not ones that already exist (backend binds at construction).

    :param name: Backend to select, 'networkx' or 'rustworkx'.
    """
    envvar = 'DACE_graph_backend'
    had_envvar = envvar in os.environ
    old_envvar = os.environ.get(envvar)
    # set_temporary MUST snapshot the old value before the env var is set: Config.get is
    # env-var-first, so setting the env var first makes it snapshot -- and then "restore" -- the
    # NEW value, leaking the temporary backend into the process permanently.
    with dace.config.set_temporary('graph', 'backend', value=name):
        os.environ[envvar] = name
        try:
            yield
        finally:
            if had_envvar:
                os.environ[envvar] = old_envvar
            else:
                del os.environ[envvar]


def get_backend_name() -> str:
    return resolve.resolve_backend().name
