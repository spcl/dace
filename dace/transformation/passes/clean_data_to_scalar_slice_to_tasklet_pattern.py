# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from typing import Dict, Optional, Set, Tuple, List
from dace import SDFG
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl, transformation
from typing import Optional

@transformation.explicit_cf_compatible
class CleanDataToScalarSliceToTaskletPattern(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return (modified & ppl.Modifies.Tasklets) or (modified & ppl.Modifies.AccessNodes) or (modified & ppl.Modifies.Edges)

    def depends_on(self):
        return {}

    def _is_array_scalar_tasklet(
        self,
        middle_node: dace.nodes.AccessNode,
        state: dace.SDFGState
    ) -> Optional[Tuple[MultiConnectorEdge, MultiConnectorEdge]]:
        middle_data = state.sdfg.arrays[middle_node.data]

        if not (isinstance(middle_data, dace.data.Scalar) or (isinstance(middle_data, dace.data.Array) and (middle_data.shape == (1,) or middle_data.shape  == [1]))):
            return None
        if isinstance(middle_data, dace.data.View):
            return None
        if middle_data.transient is False:
            return None

        ies = state.in_edges(middle_node)
        oes = state.out_edges(middle_node)
        if len(ies) != 1:
            return None
        if len(oes) != 1:
            return None

        ie: MultiConnectorEdge = ies[0]
        oe: MultiConnectorEdge = oes[0]
        if not isinstance(ie.src, dace.nodes.AccessNode):
            return None
        if not isinstance(oe.dst, dace.nodes.Tasklet):
            return None

        src_data = state.sdfg.arrays[ie.src.data]
        if not isinstance(src_data, (dace.data.Array, dace.data.Scalar)):
            return None

        if ie.dst_conn is not None:
            return None

        in_memlet: dace.Memlet = ie.data
        if in_memlet.volume != 1:
            return None

        # Expensive check, need to ensure that there are no other writes to this array
        # Expect this edge
        for c_state in state.sdfg.all_states():
            for c_node in c_state.nodes():
                if not isinstance(c_node, dace.nodes.AccessNode):
                    continue
                if c_node == middle_node:
                    continue
                if c_node.data != middle_node.data:
                    continue
                # Check in-edges, if all memlets are None then it is fine
                if c_state.in_degree(c_node) > 0:
                    for ie in c_state.in_edges(c_node):
                        if ie.data is not None:
                            return None

        return ie, oe


    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        patterns_to_rm : List[Tuple[MultiConnectorEdge, MultiConnectorEdge, dace.SDFGState]] = []
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.AccessNode):
                pattern = self._is_array_scalar_tasklet(node, graph)
                if pattern is not None:
                    # pattern: Tuple[MultiConnectorEdge, MultiConnectorEdge]
                    patterns_to_rm.append((pattern[0], pattern[1], graph))

        for arr_scalar_edge, scalar_tasklet_edge, state in patterns_to_rm:
            src = arr_scalar_edge.src
            intermediate = arr_scalar_edge.dst
            dst = scalar_tasklet_edge.dst
            assert state.degree(intermediate) == 2
            assert state.in_degree(intermediate) == 1
            assert state.out_degree(intermediate) == 1
            state.remove_edge(arr_scalar_edge)
            state.remove_edge(scalar_tasklet_edge)
            state.remove_node(intermediate)
            state.add_edge(
                src,
                arr_scalar_edge.src_conn,
                dst,
                scalar_tasklet_edge.dst_conn,
                dace.memlet.Memlet(
                    data=src.data,
                    subset=copy.deepcopy(arr_scalar_edge.data.subset)
                )
            )

        return None
