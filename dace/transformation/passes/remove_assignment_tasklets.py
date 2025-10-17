import copy
import dace
# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from dace import SDFG, InterstateEdge
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes import analysis as ap
from dace import properties
import dace.sdfg.construction_utils as cutil


@transformation.explicit_cf_compatible
class RemoveAssignmentTasklets(ppl.Pass):
    copy_tasklet_pattern = properties.Property(dtype=str, default='', allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _arr_appears_in_loop(self, arr_name: str, loop: LoopRegion):
        pass

    def _arr_appears_in_conditional(self, arr_name: str, conditional: ConditionalBlock):
        pass

    def _apply(self, sdfg: dace.SDFG):
        for state in sdfg.all_states():
            nodes_to_rm = set()
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    if len(node.in_connectors) == 1 and len(node.out_connectors) == 1:
                        in_conn_name = next(iter(node.in_connectors.keys()))
                        out_conn_name = next(iter(node.out_connectors.keys()))
                        if (node.code.as_string == f"{out_conn_name} = {in_conn_name}" or 
                            node.code.as_string == f"{out_conn_name} = {in_conn_name};" or
                            node.code.as_string == f"vector_copy({out_conn_name}, {in_conn_name});"):
                            # Can rm this node
                            nodes_to_rm.add(node)

            print(nodes_to_rm)
            for node in nodes_to_rm:
                ie = state.in_edges(node)[0]
                oe = state.out_edges(node)[0]

                state.remove_node(node)

                assert isinstance(ie.src, dace.nodes.AccessNode)
                state.add_edge(
                    ie.src, None, oe.dst, oe.dst_conn,
                    dace.memlet.Memlet(
                        data=ie.src.data,
                        subset=copy.deepcopy(ie.data.subset),
                        other_subset=copy.deepcopy(oe.data.subset)
                    )
                )

            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply(node.sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        self._apply(sdfg)

    def report(self, pass_retval: Any) -> Optional[str]:
        return f''
