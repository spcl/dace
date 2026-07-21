# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DiGraphMatcher-shaped facade over the backends, matching
networkx.algorithms.isomorphism.DiGraphMatcher's constructor and .subgraph_isomorphisms_iter()
so pattern_matching.py needs no change beyond the import swap. Both input graphs always come
from the same call site (dace.graphlib.DiGraph()), so they always share one backend.
"""
from typing import Any, Callable, Dict, Iterator, Optional

import dace.graphlib.resolve as resolve


class DiGraphMatcher:

    def __init__(self,
                 G1: Any,
                 G2: Any,
                 node_match: Optional[Callable[..., bool]] = None,
                 edge_match: Optional[Callable[..., bool]] = None) -> None:
        self._matcher = resolve.backend_for(G1).isomorphism_matcher(G1, G2, node_match, edge_match)

    def subgraph_isomorphisms_iter(self) -> Iterator[Dict[Any, Any]]:
        return self._matcher.subgraph_isomorphisms_iter()
