# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict
from dace import SDFG, InterstateEdge, properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveEmptyStates(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, sdfg: SDFG):
        for state in sdfg.all_states():
            # If empty state
            if len(state.nodes()) != 0:
                continue
            g = state.parent_graph
            if len(g.in_edges(state)) > 1:
                continue
            if len(g.out_edges(state)) > 1:
                continue

            if len(g.in_edges(state)) == 0:
                if len(g.out_edges(state)) == 1:
                    out_edge = g.out_edges(state)[0]
                    if out_edge.data.assignments == dict():
                        # Remove state, remove edge too
                        dst_node = out_edge.dst
                        dst_node_out_edges = g.out_edges(dst_node)
                        g.remove_node(state)
                        g.remove_node(dst_node)
                        g.add_node(dst_node, is_start_block=True)
                        for e in dst_node_out_edges:
                            g.add_edge(dst_node, e.dst, copy.deepcopy(e.data))
            else:
                if len(g.in_edges(state)) == 1:
                    if len(g.out_edges(state)) == 1:
                        in_edge = g.in_edge(state)[0]
                        out_edge = g.out_edges(state)[0]
                        joined_assignments = copy.deepcopy(in_edge.data.assignments).update(out_edge.data.assignments)
                        src = in_edge.src
                        dst = in_edge.dst
                        g.remove_node(state)
                        g.add_edge(src, dst, InterstateEdge(assignments=joined_assignments))

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    self._apply(node.sdfg)


    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        self._apply(sdfg)
        sdfg.validate()
