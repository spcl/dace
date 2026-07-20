"""Picks which GraphBackend implementation backs a given graph object."""
import networkx

import dace.graphlib.networkx_backend as networkx_backend
import dace.graphlib.rustworkx_backend as rustworkx_backend
from dace.config import Config


def resolve_backend():
    """Config.get('graph', 'backend'), read fresh on every call -- see the implementation plan
    (dace.graphlib package docs) for why this is deliberately not cached."""
    name = Config.get('graph', 'backend')
    if name == 'networkx':
        return networkx_backend.INSTANCE
    if name == 'rustworkx':
        return rustworkx_backend.INSTANCE
    raise ValueError(f"Unknown graph backend '{name}', expected 'networkx' or 'rustworkx'")


def backend_for(G):
    """No mixed backends: a real networkx graph (e.g. reached via an SDFG/state's .nx/._nx
    escape hatch) runs on whichever backend is CURRENTLY configured, same as any other graph --
    it is not hardcoded to networkx. Graphlib-native graphs (built via graphlib.DiGraph()) are
    the one exception: they remember their OWN backend from construction time, so an
    in-progress computation never straddles two backends mid-object-lifetime and A/B runs via
    set_default_backend() stay clean (see resolve_backend's docstring).

    This is what lets `graphlib.has_path(state.nx, a, b)` accelerate under backend='rustworkx'
    even though `state.nx` is a plain, unwrapped networkx.DiGraph -- the rustworkx backend
    builds a temporary rustworkx graph on the fly for calls like this (see
    rustworkx_backend.RustworkxBackend's helper). Raw adjacency-protocol access that never goes
    through a graphlib.* function at all (`state.nx.pred[u]`, `state.nx.reverse()` chained into
    a further nx call, mutable `G[u][v]['x']` subscripting on a .nx-derived graph) never reaches
    this dispatch and so is unaffected either way -- it keeps using real networkx directly,
    which is the correct outcome for exactly the operations that need callbacks or the full
    adjacency-view protocol rustworkx doesn't replicate (see the graphlib package docs).
    """
    if isinstance(G, (networkx.Graph, networkx.DiGraph)):
        return resolve_backend()
    return G.graphlib_backend
