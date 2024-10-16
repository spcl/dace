# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy

from dace.sdfg import SDFG, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties
from dace import symbolic, Memlet
from dace import data as dt

@make_properties
class RemoveTrivialStructureView(transformation.SingleStateTransformation):
    view = transformation.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.view)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):                
        desc = self.view.desc(sdfg)

        # Ensure view
        if not isinstance(desc, dt.StructureView):
            return False

        # Get viewed node and non-viewed edges
        view_edge = sdutil.get_view_edge(state, self.view)
        if view_edge is None:
            return False
        if "." in view_edge.data.data:
            return False

        # Gather metadata
        viewed = None
        if view_edge.dst == self.view:
            viewed = view_edge.src
        else:
            viewed = view_edge.dst

        if not isinstance(viewed, nodes.AccessNode):
            return False
        if not isinstance(sdfg.arrays[viewed.data], dt.Structure):
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        view = self.view
        view_edge = sdutil.get_view_edge(state, view)

        if view_edge.dst == view:
            viewed = view_edge.src
            for oedge in state.out_edges(view):
                memlet = oedge.data.data.replace(view.data, viewed.data) + f"[{oedge.data.subset}]"
                state.add_edge(viewed, oedge.src_conn, oedge.dst, oedge.dst_conn, Memlet(memlet))
                state.remove_edge(oedge)
            
            for edge in state.edge_bfs(viewed):
                if edge.data.data == view.data:
                    memlet = edge.data.data.replace(view.data, viewed.data) + f"[{edge.data.subset}]"
                    state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, Memlet(memlet))
                    state.remove_edge(edge)

        else:
            viewed = view_edge.dst
            for iedge in state.in_edges(view):
                memlet = iedge.data.data.replace(view.data, viewed.data) + f"[{iedge.data.subset}]"
                state.add_edge(iedge.src, iedge.src_conn, viewed, iedge.dst_conn, Memlet(memlet))
                state.remove_edge(iedge)
            
            for edge in state.edge_bfs(viewed, reverse=True):
                if edge.data.data == view.data:
                    memlet = edge.data.data.replace(view.data, viewed.data) + f"[{edge.data.subset}]"
                    state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, Memlet(memlet))
                    state.remove_edge(edge)

        state.remove_node(view)
        