# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.data import Structure, View
from dace.sdfg.data_group import DataGroup
import re

def _extract_view_name(view_string: str, struct_name: str) -> str:
    pattern = rf"^v_{re.escape(struct_name)}_(.+)$"
    match = re.match(pattern, view_string)
    if match:
        return match.group(1)
    return None

@make_properties
class StructToDataGroup(transformation.SingleStateTransformation):
    src_access = transformation.PatternNode(nodes.AccessNode)
    dst_access = transformation.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.src_access, cls.dst_access)]

    def can_be_applied(self, state: SDFGState, expr_index, sdfg: SDFG, permissive=False):
        # Pattern1: A -> B, A struct, B pointer/view or whatever or
        # Pattern2: B -> A, B pointer/view, A struct
        # Condition: DataGroups have been generated (sdfg.generate_data_groups_from_structs())
        (struct_to_view_pattern, view_to_struct_pattern)  = self._get_pattern_type(state, sdfg)
        if (not struct_to_view_pattern) and (not view_to_struct_pattern):
            return False
        if struct_to_view_pattern and view_to_struct_pattern:
            raise Exception("A -> B and B -> A found in structure + view access (impossible cycle)")

        (struct_access, view_access, struct_data, view_data) = self._assign_src_dst_to_struct_view(sdfg)
        if struct_access is None or view_access is None:
            return False

        if not (isinstance(struct_data, Structure)):
            return False
        if not (isinstance(view_data, View)):
            return False

        return True

    def _assign_src_dst_to_struct_view(self, sdfg: SDFG):
        struct_access = None
        view_access = None
        struct_data = None
        view_data = None

        src_data = sdfg.arrays[self.src_access.data]
        dst_data = sdfg.arrays[self.dst_access.data]

        if isinstance(src_data, Structure):
            struct_access = self.src_access
            struct_data = src_data
        elif isinstance(dst_data, Structure):
            struct_access = self.dst_access
            struct_data = dst_data

        if isinstance(src_data, View):
            view_access = self.src_access
            view_data = src_data
        elif isinstance(dst_data, View):
            view_access = self.dst_access
            view_data = dst_data

        return (struct_access, view_access, struct_data, view_data)

    def _get_pattern_type(self, state: SDFGState, sdfg: SDFG):
        (struct_access, view_access, struct_data, view_data) = self._assign_src_dst_to_struct_view(sdfg)

        struct_to_view_edges = set([v for _,_,v,_,_ in state.out_edges(struct_access) if v == view_access]) if struct_access else set()
        view_to_struct_edges = set([v for _,_,v,_,_ in state.out_edges(view_access) if v == struct_access]) if view_access else set()

        struct_to_view_pattern = False
        view_to_struct_pattern = False

        if len(struct_to_view_edges) == 0 and len(view_to_struct_edges) == 0:
            return (False, False)
        elif len(struct_to_view_edges) != 0 and len(view_to_struct_edges) != 0:
            raise Exception("A -> B and B -> A found in structure + view access (impossible cycle)")
        elif len(struct_to_view_edges) != 0:
            struct_to_view_pattern = True
        elif len(view_to_struct_edges) != 0:
            view_to_struct_pattern = True

        return (struct_to_view_pattern, view_to_struct_pattern)

    def apply(self, state: SDFGState, sdfg: SDFG):
        struct_to_view, view_to_struct = self._get_pattern_type(state, sdfg)
        if not (struct_to_view or view_to_struct):
            raise Exception("StructToDataGroup not applicable")
        assert not (struct_to_view and view_to_struct)

        if struct_to_view:
            struct_access = self.src_access
            view_access = self.dst_access
        else: # view_to_struct
            view_access = self.src_access
            struct_access = self.dst_access
        view_name = view_access.data
        struct_name = struct_access.data

        extracted_view_name = _extract_view_name(view_name, struct_name)
        demangled_name = sdfg.get_demangled_data_group_member_name([struct_name, extracted_view_name])

        an = nodes.AccessNode(data=demangled_name)

        src, dst = (struct_access, view_access) if struct_to_view else (view_access, struct_access)
        edges = [e for e in state.out_edges(src) if e.dst == dst]
        assert len(edges) == 1

        edge = edges[0]
        u, uc, v, vc = edge.src, edge.src_conn, edge.dst, edge.dst_conn
        mc = copy.deepcopy(edge.data)
        mc.data = demangled_name

        # If Struct -> View -> Dst:
        # Then Struct (uc) -> (None) \ View \ (None) -> (vc) Dst
        # Becomes NewData (None) -> (vc) Dst

        # If View -> Struct -> Dst:
        # Then Src (uc) -> (None) \ View \ (None) -> (vc) Struct
        # Becomes Src (uc) -> (None) NewData
        state.add_node(an)
        if struct_to_view:
            dst_edges = state.out_edges(v)
            assert len(dst_edges) == 1
            dst_edge = dst_edges[0]
            # TODO: Fix memlet calculation in recursive data groups
            mc = copy.deepcopy(dst_edge.data)
            mc.data = demangled_name
            state.add_edge(an, None, dst_edge.dst, dst_edge.dst_conn, mc)
        else: # view_to_struct
            src_edges = state.in_edges(u)
            assert len(src_edges) == 1
            src_edge = src_edges[0]
            # TODO: Fix memlet calculation in recursive data groups
            mc = copy.deepcopy(src_edge.data)
            mc.data = demangled_name
            state.add_edge(src_edge.src, src_edge.src_conn, an, None, mc)

        # Clean-up
        state.remove_edge(edge)
        state.remove_node(view_access)
        state.remove_node(view_access)
        if (len(state.in_edges(struct_access)) == 0) and (len(state.out_edges(struct_access)) == 0):
            state.remove_node(struct_access)

    def annotates_memlets():
        return False
