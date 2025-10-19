# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Set, Optional

from dace import SDFG, InterstateEdge, properties
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ReturnBlock
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class EmptyLoopElimination(ppl.Pass):
    """
    Removes all loops with empty bodies.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If connectivity or any edges were changed, some more loops might be dead
        return modified & ppl.Modifies.CFG

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        loops = [(n, parent) for n, parent in sdfg.all_nodes_recursive()
                 if isinstance(n, LoopRegion) and parent.in_degree(n) <= 1 and parent.out_degree(n) <= 1]

        changed = True
        num_removed = 0
        while changed:
            changed = False
            cfgs_to_rm: Set[LoopRegion] = set()

            for node, parent in loops:
                inner_nodes = node.nodes()
                if len(inner_nodes) == 1 and len(
                        inner_nodes[0].nodes()) == 0 and not isinstance(inner_nodes[0], ReturnBlock):
                    cfgs_to_rm.add((node, parent))

            for node, parent_graph in cfgs_to_rm:
                self._remove_node_connect_src_and_dst(node, parent_graph)
                num_removed += 1
                loops.remove((node, parent_graph))
                changed = True

        if num_removed > 0:
            return num_removed
        return None

    def _remove_node_connect_src_and_dst(self, node: LoopRegion, parent_graph: ControlFlowRegion):
        ies = parent_graph.in_edges(node)
        oes = parent_graph.out_edges(node)
        assert len(ies) <= 1 and len(oes) <= 1

        if len(ies) == 0 and len(oes) == 0:
            parent_graph.add_state_before(node)
            parent_graph.remove_node(node)
            return

        if len(ies) == 0:
            parent_graph.add_state_before(node)
        if len(oes) == 0:
            parent_graph.add_state_after(node)

        ies = parent_graph.in_edges(node)
        oes = parent_graph.out_edges(node)
        new_assignments = dict()
        new_assignments.update(ies[0].data.assignments)
        new_assignments.update(oes[0].data.assignments)
        parent_graph.add_edge(ies[0].src, oes[0].dst, InterstateEdge(condition="1", assignments=new_assignments))
        parent_graph.remove_node(node)
