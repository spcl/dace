# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Map-scope membership for passes that must inspect a whole map body.

Deliberately NOT ``SDFGState.all_nodes_between``: that walk abandons the ENTIRE result the moment
it reaches a node with no out-edge (``dace/sdfg/graph.py:435`` returns an empty set), and a
write-only scratch scalar -- a transient ``AccessNode`` with one in-edge and none out -- is exactly
such a node. A gate reading that walk then reports a clean body having inspected nothing, so it
admits whatever it was built to refuse, and a counter built on it reports zero for a body full of
nodes. Scope membership has no such failure mode: ``scope_children`` classifies every node in the
state, and it is already paid for, since ``state.exit_node`` reads the same cached scope dict.

Neutral home on purpose: canonicalization must not import from vectorization, and both trees need
the same answer.
"""
from typing import List

from dace.sdfg import SDFGState, nodes


def map_body_nodes(state: SDFGState, map_entry: nodes.MapEntry) -> List[nodes.Node]:
    """Every node in ``map_entry``'s scope -- entry and exit excluded, inner scopes included.

    :param state: The state holding ``map_entry``.
    :param map_entry: The map whose body is wanted.
    :returns: The scope's nodes, in state order.
    """
    return list(state.scope_subgraph(map_entry, include_entry=False, include_exit=False).nodes())
