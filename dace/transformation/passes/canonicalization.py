# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from dace.transformation import pass_pipeline as ppl
from dace.transformation import helpers as xfh
from dace.memlet import Memlet
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import StateSubgraphView, MultiConnectorEdge

import networkx as nx
from typing import Any, Dict, Optional, Set


class SeparateRefsets(ppl.Pass):
    """
    Moves free (unscoped) Reference sets into a separate, prior state in order
    to avoid code generation scheduling issues (e.g., when the set reference
    is also viewed).
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Nodes

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Set[str]]:
        sets_moved_back = set()

        queue = list(sdfg.states())
        while len(queue) > 0:
            state = queue.pop()
            for anode in state.data_nodes():
                if 'set' in anode.in_connectors and state.out_degree(anode) > 0:
                    if state.entry_node(anode) is not None:
                        continue
                    if 'views' in anode.out_connectors:
                        continue  # Skip refset to a substructure view
                    edge = next(iter(state.in_edges_by_connector(anode, 'set')))

                    # Move reference set and all ancestors to a prior state
                    substate = StateSubgraphView(state, nx.ancestors(state._nx, edge.src) | {edge.src, edge.dst})
                    sets_moved_back.add(edge)
                    newstate = xfh.state_fission(sdfg, substate)

                    # Handle remaining access node
                    anode.remove_in_connector('set')
                    if len([e for e in state.all_edges(anode) if not e.data.is_empty()]) == 0:
                        # Newly-isolated node
                        state.remove_node(anode)

                    # Traverse to newly-created states (e.g., in case a reference
                    # is set by a tasklet that is set by a reference set itself).
                    queue.append(newstate)

        return sets_moved_back or None

    def report(self, pass_retval: Set[MultiConnectorEdge[Memlet]]) -> str:
        return f'Prepended {len(pass_retval)} reference sets.'
