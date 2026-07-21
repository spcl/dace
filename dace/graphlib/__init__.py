"""dace.graphlib -- networkx-compatible graph layer with a config-selectable backend
(Config.get('graph', 'backend'): 'networkx', the default/safe passthrough, or 'rustworkx', an
accelerated opt-in). Mechanical migration target for every `import networkx as nx` in dace:
`from dace import graphlib as nx`.

No mixed backends: this module is either "full networkx" or "full rustworkx", not a patchwork.
Graphs built directly through DiGraph()/MultiDiGraph() below remember which backend produced
them (resolved once, at construction time, not re-read per call -- see resolve.py). Real
networkx graphs (e.g. reached through an SDFG/state's .nx/._nx escape hatch) are NOT hardcoded
to the networkx backend -- they run on whichever backend is currently configured, same as any
other graph: under backend='rustworkx', a plain call like `has_path(state.nx, a, b)` still runs
accelerated, via an on-the-fly conversion (see rustworkx_backend._coerce). The one thing that
never changes regardless of backend is DaCe's core IR storage itself -- `.nx`/`._nx` is always a
real, unwrapped networkx.DiGraph/MultiDiGraph instance (so SDFG/SDFGState/ControlFlowRegion's
existing deepcopy/serialization behavior, and the raw adjacency-protocol access some external
call sites still use directly -- `.pred`, `.reverse()` chained into a further call, mutable
`G[u][v]['x']` subscripting -- are completely unaffected by this module either way); what's
backend-aware is only the *algorithm implementation* graphlib.* functions dispatch to when
called on it. The rule for what does and doesn't get lowered to rustworkx: algorithms with no
callback/special-adjacency requirement (has_path, immediate_dominators, topological_sort, etc.)
do; the two rustworkx has no implementation of at all (transitive closure, max-flow/min-cut)
always run on real networkx regardless of backend, since there's nothing to lower to.
"""
import contextlib
import os

import dace.config
import dace.graphlib.resolve as resolve
import dace.graphlib.rustworkx_backend as rustworkx_backend
from dace.graphlib import isomorphism
from dace.graphlib.algorithms.flow import edmondskarp
from networkx.exception import NetworkXError, NetworkXNoCycle, NetworkXNoPath, NetworkXUnfeasible, NodeNotFound

__all__ = [
    'DiGraph', 'MultiDiGraph', 'has_path', 'immediate_dominators', 'weakly_connected_components', 'topological_sort',
    'simple_cycles', 'find_cycle', 'descendants', 'ancestors', 'all_simple_paths', 'transitive_closure',
    'transitive_closure_dag', 'dfs_edges', 'shortest_path_length', 'minimum_cut', 'get_node_attributes', 'isomorphism',
    'NetworkXError', 'NetworkXNoCycle', 'NetworkXNoPath', 'NetworkXUnfeasible', 'NodeNotFound', 'set_default_backend',
    'get_backend_name'
]


def DiGraph():
    return resolve.resolve_backend().new_digraph()


def MultiDiGraph():
    return resolve.resolve_backend().new_multidigraph()


def has_path(G, source, target):
    return resolve.backend_for(G).has_path(G, source, target)


def immediate_dominators(G, start):
    return resolve.backend_for(G).immediate_dominators(G, start)


def weakly_connected_components(G):
    return resolve.backend_for(G).weakly_connected_components(G)


def topological_sort(G):
    return resolve.backend_for(G).topological_sort(G)


def simple_cycles(G):
    return resolve.backend_for(G).simple_cycles(G)


def find_cycle(G, source=None):
    return resolve.backend_for(G).find_cycle(G, source)


def descendants(G, source):
    return resolve.backend_for(G).descendants(G, source)


def ancestors(G, source):
    return resolve.backend_for(G).ancestors(G, source)


def all_simple_paths(G, source, target):
    return resolve.backend_for(G).all_simple_paths(G, source, target)


def transitive_closure(G):
    return resolve.backend_for(G).transitive_closure(G)


def transitive_closure_dag(G):
    return resolve.backend_for(G).transitive_closure_dag(G)


def dfs_edges(G, source=None):
    return resolve.backend_for(G).dfs_edges(G, source)


def shortest_path_length(G, source, target):
    return resolve.backend_for(G).shortest_path_length(G, source, target)


def get_node_attributes(G, name, default=None):
    """Plain attribute-dict reader, not an accelerated algorithm (there's no rustworkx-native
    equivalent that would be faster than iterating) -- works directly on either a real networkx
    graph or a RustworkxGraphHandle by using each one's own node-payload access."""
    if isinstance(G, rustworkx_backend.RustworkxGraphHandle):
        result = {}
        payloads = G.node_payloads_by_index()  # one bulk fetch, not a per-node get_node_data round-trip
        for idx, node in G._index.idx_to_obj.items():
            attrs = payloads[idx]
            # networkx omits nodes lacking the attribute only when no default is given; with a
            # default every node appears, carrying that default.
            if name in attrs:
                result[node] = attrs[name]
            elif default is not None:
                result[node] = default
        return result
    import networkx
    return networkx.get_node_attributes(G, name, default)


def minimum_cut(G, s, t, capacity='capacity', flow_func=None, **kwargs):
    """ALWAYS real networkx internally -- rustworkx has no directed s-t max-flow/min-cut. One
    named function, one documented, tested fallback -- not a per-call shim pattern repeated
    everywhere."""
    import networkx
    if isinstance(G, rustworkx_backend.RustworkxGraphHandle):
        G = rustworkx_backend.to_networkx(G)
    return networkx.minimum_cut(G, s, t, capacity=capacity, flow_func=flow_func, **kwargs)


@contextlib.contextmanager
def set_default_backend(name: str):
    """Built on dace.config.set_temporary, plus a matching os.environ override. Config.get()
    checks DACE_graph_backend before its in-memory value (dace/config.py's documented,
    intentional env-var-first precedence), so set_temporary alone is silently a no-op whenever
    that env var is already set in the process -- e.g. CI jobs that export
    DACE_graph_backend=rustworkx for the whole pytest run, which would otherwise defeat every
    test that tries to select a backend explicitly (see tests/graphlib/backend_differential_test.py
    and tests/perf/graph_backend_cloudsc_test.py, both hit this). Affects graphs constructed for
    the duration of the context, not graphs that already exist (see resolve.py: backend is
    bound at construction time)."""
    envvar = 'DACE_graph_backend'
    had_envvar = envvar in os.environ
    old_envvar = os.environ.get(envvar)
    # set_temporary MUST snapshot the old value before the env var is set: it restores what
    # Config.get returned on entry, and Config.get is env-var-first, so setting the env var
    # first makes it snapshot the NEW value and "restore" that on exit -- leaking the
    # temporary backend into the process permanently and contaminating every later test in
    # the same worker.
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
