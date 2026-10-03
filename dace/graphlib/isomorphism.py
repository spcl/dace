"""DiGraphMatcher-shaped facade -- matches networkx.algorithms.isomorphism.DiGraphMatcher's
constructor and .subgraph_isomorphisms_iter() shape so callers
(dace/transformation/passes/pattern_matching.py) need no changes beyond the import swap
(`from networkx.algorithms import isomorphism as iso` -> `from dace.graphlib import isomorphism as iso`).
Dispatches to whichever backend produced the input graphs -- both are always produced by the
same call site (dace.graphlib.DiGraph()), so they always share one backend.
"""
import dace.graphlib.resolve as resolve


class DiGraphMatcher:

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        self._matcher = resolve.backend_for(G1).isomorphism_matcher(G1, G2, node_match, edge_match)

    def subgraph_isomorphisms_iter(self):
        return self._matcher.subgraph_isomorphisms_iter()
