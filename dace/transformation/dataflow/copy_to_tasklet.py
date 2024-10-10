# Copyright 2022-2024 ETH Zurich and the Daisytuner authors.
import dace
import re

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class CopyToTasklet(transformation.SingleStateTransformation):
    access_node = transformation.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.access_node)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        for edge in state.out_edges(self.access_node):
            if edge.data is None:
                continue
            if edge.data.get_src_subset(edge, state) is None:
                continue
            if edge.data.get_dst_subset(edge, state) is None:
                continue
            if edge.data.num_elements() != 1:
                continue

            return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        for edge in state.out_edges(self.access_node):
            if edge.data is None:
                continue
            if edge.data.get_src_subset(edge, state) is None:
                continue
            if edge.data.get_dst_subset(edge, state) is None:
                continue
            if edge.data.num_elements() != 1:
                continue

            tasklet = state.add_tasklet(
                "copy", set(("_in",)), outputs=set(("_out",)), code="_out = _in"
            )

            memlet_in = dace.Memlet(
                data=self.access_node.data, subset=edge.data.src_subset
            )
            state.add_edge(edge.src, edge.src_conn, tasklet, "_in", memlet_in)

            if isinstance(edge.dst, dace.nodes.AccessNode):
                memlet_out = dace.Memlet(
                    data=edge.dst.data, subset=edge.data.dst_subset
                )
            else:
                memlet_out = dace.Memlet(
                    data=edge.data.data, subset=edge.data.dst_subset
                )

            state.add_edge(tasklet, "_out", edge.dst, edge.dst_conn, memlet_out)

            state.remove_edge(edge)
