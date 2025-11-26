# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from typing import Any, Dict, List, Set, Optional, Tuple
from dace import SDFG, InterstateEdge, properties
from dace.memlet import Memlet
from dace.sdfg.graph import Edge
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ReturnBlock
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.eliminate_branches import EliminateBranches


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveAssignmentTasklets(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {EliminateBranches}

    def _detect_access_node_tasklet_access_node(
        self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet
    ) -> Optional[Tuple[dace.nodes.AccessNode, dace.nodes.Tasklet, dace.nodes.AccessNode]]:
        in_edges = state.in_edges(tasklet)
        out_edges = state.out_edges(tasklet)

        if len(in_edges) != 1 or len(out_edges) != 1:
            return None

        in_edge = in_edges[0]
        out_edge = out_edges[0]

        if not isinstance(in_edge.src, dace.nodes.AccessNode):
            return None
        if not isinstance(out_edge.dst, dace.nodes.AccessNode):
            return None

        return in_edge.src, tasklet, out_edge.dst

    def _detect_access_node_tasklet_map_exit(
            self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet
    ) -> Optional[Tuple[dace.nodes.AccessNode, dace.nodes.Tasklet, dace.nodes.MapExit]]:
        in_edges = state.in_edges(tasklet)
        out_edges = state.out_edges(tasklet)

        if len(in_edges) != 1 or len(out_edges) != 1:
            return None

        in_edge = in_edges[0]
        out_edge = out_edges[0]

        if not isinstance(in_edge.src, dace.nodes.AccessNode):
            return None
        if not isinstance(out_edge.dst, dace.nodes.MapExit):
            return None

        if isinstance(state.sdfg.arrays[in_edge.src.data], dace.data.Scalar) is False:
            return None

        return in_edge.src, tasklet, out_edge.dst

    def _is_assignment(self, tasklet: dace.nodes.Tasklet):
        inc = tasklet.in_connectors
        outc = tasklet.out_connectors
        if len(inc) != 1 or len(outc) != 1:
            return False
        in_conn = list(inc)[0]
        out_conn = list(outc)[0]
        code = tasklet.code.as_string.strip()
        if code == f"{out_conn} = {in_conn}" or code == f"{out_conn} = {in_conn};":
            return True
        return False

    def _apply(self, sdfg: SDFG, potential_scalars: Set[str]) -> None:
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    if self._is_assignment(node) is False:
                        continue

                    res = self._detect_access_node_tasklet_access_node(state, node)

                    if res is not None:
                        in_access, tasklet, out_access = res
                        if ((in_access.data in potential_scalars and out_access.data in potential_scalars)
                                or (in_access.data not in potential_scalars and out_access.data in potential_scalars)):
                            oes = state.out_edges(out_access)
                            ie_memlet = state.in_edges(tasklet)[0].data

                            state.remove_node(tasklet)

                            for oe in oes:
                                state.remove_edge(oe)

                            if state.degree(out_access) == 0:
                                state.remove_node(out_access)

                            for oe in oes:
                                state.add_edge(in_access, None, oe.dst, oe.dst_conn, copy.deepcopy(ie_memlet))

                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply(node.sdfg, potential_scalars)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        potential_scalars = pipeline_results[EliminateBranches.__name__][1]  # [0] is num_applied
        self._apply(sdfg, potential_scalars)
        sdfg.validate()
