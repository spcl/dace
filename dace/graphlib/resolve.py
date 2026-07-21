# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Picks which GraphBackend implementation backs a given graph object."""
from typing import Any

import networkx

import dace.graphlib.networkx_backend as networkx_backend
import dace.graphlib.protocol as protocol
import dace.graphlib.rustworkx_backend as rustworkx_backend
from dace.config import Config


def resolve_backend() -> protocol.GraphBackend:
    """
    Config.get('graph', 'backend'), read fresh on every call -- deliberately not cached.

    :return: The configured backend instance.
    """
    name = Config.get('graph', 'backend')
    if name == 'networkx':
        return networkx_backend.INSTANCE
    if name == 'rustworkx':
        return rustworkx_backend.INSTANCE
    raise ValueError(f"Unknown graph backend '{name}', expected 'networkx' or 'rustworkx'")


def backend_for(G: Any) -> protocol.GraphBackend:
    """
    No mixed backends: a real networkx graph (an SDFG/state's .nx/._nx) runs on whichever backend
    is CURRENTLY configured. Graphlib-native graphs are the exception -- they remember their OWN
    backend from construction time, so one object never straddles two backends mid-lifetime and
    set_default_backend() A/B runs stay clean.

    Raw adjacency access that never goes through a graphlib.* function (`state.nx.pred[u]`,
    mutable `G[u][v]['x']`) never reaches this dispatch and keeps using real networkx directly.

    :param G: The graph to dispatch on.
    :return: The backend that must serve calls against ``G``.
    """
    if isinstance(G, (networkx.Graph, networkx.DiGraph)):
        return resolve_backend()
    return G.graphlib_backend
