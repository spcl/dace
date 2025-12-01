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
class RemoveRedundantAssignments(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'
    permissive = dace.properties.Property(dtype=bool, default=False, allow_none=False)

    def __init__(self, permissive: bool = False):
        self.permissive = permissive

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
        i=0
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    if self._is_assignment(node) is False:
                        continue

                    res = self._detect_access_node_tasklet_access_node(state, node)

                    if res is not None:
                        in_access, tasklet, out_access = res
                        print(in_access, tasklet, out_access)
                        if  ((in_access.data in potential_scalars and out_access.data in potential_scalars)
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
                        elif self.permissive:
                            ies = state.in_edges(in_access)
                            oe_memlet = state.out_edges(tasklet)[0].data
                            print("Permissive mode", in_access, tasklet, out_access)
                            oes = state.out_edges(in_access)
                            state.remove_node(in_access)
                            state.remove_node(tasklet)

                            # If number of out edges are more than 1
                            if len(oes) > 1:
                                for oe in oes:
                                    if oe.dst == tasklet:
                                        continue
                                    new_memlet = dace.memlet.Memlet(data=oe_memlet.data, subset=oe_memlet.subset)
                                    state.add_edge(out_access, None, oe.dst, oe.dst_conn, new_memlet)

                            for ie in ies:
                                new_memlet = dace.memlet.Memlet(data=oe_memlet.data, subset=oe_memlet.subset)
                                print(f"Add edge from {ie.src} -> {out_access}")
                                state.add_edge(ie.src, ie.src_conn, out_access, None, new_memlet)

                            i+=1

                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply(node.sdfg, potential_scalars)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        potential_scalars = pipeline_results[EliminateBranches.__name__][1]  # [0] is num_applied
        print(potential_scalars)
        self._apply(sdfg, potential_scalars)
        sdfg.validate()
