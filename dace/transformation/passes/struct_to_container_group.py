# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy
from typing import Any, Dict
import dace
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.data import Structure, View
import re
from dace.transformation import pass_pipeline as ppl
from enum import Enum
from dace.sdfg.container_group import ContainerGroupFlatteningMode

def _remove_trailing_number(s):
    # Pattern to match '_<int>' at the end of the string
    return re.sub(r"_\d+$", "", s)


def _has_trailing_number(s):
    # Check if the string ends with '_<int>'
    return bool(re.search(r"_\d+$", s))


@make_properties
class StructToContainerGroups(ppl.Pass):
    def __init__(self, flattening_mode: ContainerGroupFlatteningMode = ContainerGroupFlatteningMode.StructsOfArrays):
        self._access_names_map = dict()
        self._data_connected_to_vsv_struct = dict()
        self._flattening_mode = flattening_mode
        if self._flattening_mode == ContainerGroupFlatteningMode.ArrayOfStructs:
            raise Exception("TODO IMPL")
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> int:
        sdfg.generate_container_groups_from_structs(self._flattening_mode)
        sdfg.register_container_group_members()

        # A -> B both access nodes, this should trigger the further check whether we can apply
        for state in sdfg.nodes():
            nodes = state.nodes()
            removed_nodes = set()
            for node in nodes:
                if node in removed_nodes:
                    continue
                if isinstance(node, dace.nodes.AccessNode):
                    out_edges = state.out_edges(node)
                    for oe in out_edges:
                        if oe.dst in removed_nodes:
                            continue
                        if isinstance(oe.dst, dace.nodes.AccessNode):
                            src_access = node
                            dst_access = oe.dst
                            pattern_found = self._can_be_applied(
                                state, sdfg, src_access, dst_access
                            )
                            if pattern_found:
                                newly_removed_nodes = self._apply(
                                    state, sdfg, src_access, dst_access
                                )
                                removed_nodes = removed_nodes.union(newly_removed_nodes)

        # Clean Mapped Views (Views within data groups)
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    if node.data in self._access_names_map:
                        node.data = self._access_names_map[node.data]
            for edge in state.edges():
                if edge.data.data in self._access_names_map:
                    edge.data.data = self._access_names_map[edge.data.data]

        # View -> Struct -> View patterns result with disconnected compenets reconnect them with saved info
        for state in sdfg.nodes():
            nodes = state.nodes()
            for node in nodes:
                if node not in state.nodes():
                    continue
                for (
                    in_connected_nodes,
                    out_connected_nodes,
                ) in self._data_connected_to_vsv_struct.values():
                    assert (
                        len(in_connected_nodes) <= 1 and len(out_connected_nodes) <= 1
                    )
                    if len(in_connected_nodes) == 1 and len(out_connected_nodes) == 1:
                        if node in in_connected_nodes:
                            src = node
                            dst = out_connected_nodes[0]
                            for oe in state.out_edges(dst):
                                assert oe.src_conn is None
                                state.add_edge(
                                    src,
                                    None,
                                    oe.dst,
                                    oe.dst_conn,
                                    copy.deepcopy(oe.data),
                                )
                            state.remove_node(dst)
                        elif node in out_connected_nodes:
                            continue

    def _can_be_applied(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        # Pattern1: A -> B, A struct, B pointer/view or whatever or
        # Pattern2: B -> A, B pointer/view, A struct
        # Condition: DataGroups have been generated (sdfg.generate_container_groups_from_structs())
        (struct_to_view_pattern, view_to_struct_pattern) = self._get_pattern_type(
            state, sdfg, src_access, dst_access
        )
        if (not struct_to_view_pattern) and (not view_to_struct_pattern):
            return False
        if struct_to_view_pattern and view_to_struct_pattern:
            raise Exception(
                "A -> B and B -> A found in structure + view access (impossible cycle)"
            )

        (struct_access, view_access, struct_data, view_data) = (
            self._assign_src_dst_to_struct_view(sdfg, src_access, dst_access)
        )
        if struct_access is None or view_access is None:
            return False

        if not (isinstance(struct_data, Structure)):
            return False
        if not (isinstance(view_data, View)):
            return False

        return True

    def _assign_src_dst_to_struct_view(
        self, sdfg: SDFG, src_access: nodes.AccessNode, dst_access: nodes.AccessNode
    ):
        struct_access = None
        view_access = None
        struct_data = None
        view_data = None

        src_data = sdfg.arrays[src_access.data]
        dst_data = sdfg.arrays[dst_access.data]

        if isinstance(src_data, Structure):
            struct_access = src_access
            struct_data = src_data
        elif isinstance(dst_data, Structure):
            struct_access = dst_access
            struct_data = dst_data

        if isinstance(src_data, View):
            view_access = src_access
            view_data = src_data
        elif isinstance(dst_data, View):
            view_access = dst_access
            view_data = dst_data

        return (struct_access, view_access, struct_data, view_data)

    def _get_pattern_type(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        (struct_access, view_access, struct_data, view_data) = (
            self._assign_src_dst_to_struct_view(sdfg, src_access, dst_access)
        )

        struct_to_view_edges = (
            set(
                [
                    v
                    for _, _, v, _, _ in state.out_edges(struct_access)
                    if v == view_access
                ]
            )
            if struct_access
            else set()
        )
        view_to_struct_edges = (
            set(
                [
                    v
                    for _, _, v, _, _ in state.out_edges(view_access)
                    if v == struct_access
                ]
            )
            if view_access
            else set()
        )

        struct_to_view_pattern = False
        view_to_struct_pattern = False

        if len(struct_to_view_edges) == 0 and len(view_to_struct_edges) == 0:
            return (False, False)
        elif len(struct_to_view_edges) != 0 and len(view_to_struct_edges) != 0:
            raise Exception(
                "A -> B and B -> A found in structure + view access (impossible cycle)"
            )
        elif len(struct_to_view_edges) != 0:
            struct_to_view_pattern = True
        elif len(view_to_struct_edges) != 0:
            view_to_struct_pattern = True

        return (struct_to_view_pattern, view_to_struct_pattern)

    def _get_view_chain(
        self, state: SDFGState, sdfg: SDFG, first_view_access: nodes.AccessNode
    ):
        view_accesses = [first_view_access]
        current_view_access = first_view_access
        while True:
            out_edges = state.out_edges(current_view_access)
            assert len(out_edges) == 1
            out_edge = out_edges[0]
            u, uc, v, vc, memlet = out_edge
            if isinstance(v, nodes.AccessNode) and isinstance(
                sdfg.arrays[v.data], View
            ):
                current_view_access = v
                view_accesses.append(v)
            else:
                return view_accesses

    def _process_edges(self, edge_list, name_hierarchy, take_last=False):
        assert len(edge_list) == 1
        edge = edge_list[0]
        data = edge.data.data
        tokenized_data = data.split(".")
        assert len(tokenized_data) == 2 or len(tokenized_data) == 1
        if not take_last:
            name_hierarchy += tokenized_data
        else:
            name_hierarchy += [tokenized_data[-1]]

    def _apply(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        removed_nodes = set()

        struct_to_view, view_to_struct = self._get_pattern_type(
            state, sdfg, src_access, dst_access
        )
        if not (struct_to_view or view_to_struct):
            raise Exception("StructToDataGroup not applicable")
        assert not (struct_to_view and view_to_struct)

        if struct_to_view:
            struct_access = src_access
            view_access = dst_access
        else:  # view_to_struct
            view_access = src_access
            struct_access = dst_access

        view_chain = self._get_view_chain(state, sdfg, view_access)
        assert len(view_chain) >= 1
        name_hierarchy = []

        if struct_to_view:
            struct_to_view_edges = [
                e for e in state.out_edges(struct_access) if e.dst == view_chain[0]
            ]
            self._process_edges(
                edge_list=struct_to_view_edges, name_hierarchy=name_hierarchy
            )

        for current_view_access in view_chain[:-1]:
            view_to_next_edges = state.out_edges(current_view_access)
            self._process_edges(
                edge_list=view_to_next_edges,
                name_hierarchy=name_hierarchy,
                take_last=True,
            )

        if view_to_struct:
            view_to_struct_edges = [
                e for e in state.in_edges(struct_access) if e.src == view_chain[-1]
            ]
            self._process_edges(
                edge_list=view_to_struct_edges, name_hierarchy=name_hierarchy
            )

        for i in range(len(name_hierarchy)):
            if _has_trailing_number(name_hierarchy[i]):
                name_hierarchy[i] = _remove_trailing_number(name_hierarchy[i])

        demangled_name = sdfg.get_demangled_container_group_member_name(name_hierarchy)

        an = nodes.AccessNode(data=demangled_name)

        if struct_to_view:
            assert len(state.out_edges(view_chain[0])) == 1
            src_edge = state.out_edges(view_chain[0])[0]
            assert len(state.out_edges(view_chain[-1])) == 1
            dst_edge = state.out_edges(view_chain[-1])[0]
        else:  # view_to_struct
            assert len(state.in_edges(view_chain[0])) == 1
            src_edge = state.in_edges(view_chain[0])[0]
            assert len(state.out_edges(view_chain[-1])) == 1
            dst_edge = state.out_edges(view_chain[-1])[0]

        dst_data = dst_edge.data
        mc = copy.deepcopy(dst_data)
        mc.data = demangled_name

        # If Struct -> View -> Dst:
        # Then Struct (uc) -> (None) \ View \ (None) -> (vc) Dst
        # Becomes NewData (None) -> (vc) Dst

        # If View -> Struct -> Dst:
        # Then Src (uc) -> (None) \ View \ (None) -> (vc) Struct
        # Becomes Src (uc) -> (None) NewData
        state.add_node(an)
        # TODO: Fix memlet calculation in recursive data groups
        if struct_to_view:
            state.add_edge(an, None, dst_edge.dst, dst_edge.dst_conn, mc)

            if struct_access.guid in self._data_connected_to_vsv_struct:
                self._data_connected_to_vsv_struct[struct_access.guid][1].append(an)
            else:
                self._data_connected_to_vsv_struct[struct_access.guid] = ([], [an])

        else:  # view_to_struct
            state.add_edge(src_edge.src, src_edge.src_conn, an, None, mc)

            if struct_access.guid in self._data_connected_to_vsv_struct:
                self._data_connected_to_vsv_struct[struct_access.guid][0].append(an)
            else:
                self._data_connected_to_vsv_struct[struct_access.guid] = ([an], [])

        # Clean-up
        for view_node in view_chain:
            state.remove_node(view_node)
            removed_nodes.add(view_node)
        if (len(state.in_edges(struct_access)) == 0) and (
            len(state.out_edges(struct_access)) == 0
        ):
            state.remove_node(struct_access)
            removed_nodes.add(struct_access)

        self._access_names_map[view_chain[-1 if struct_to_view else 0].data] = (
            demangled_name
        )

        return removed_nodes

    def _get_src_dst(self, state: SDFGState, n1: nodes.Any, n2: nodes.Any):
        n1_to_n2 = [e.dst for e in state.out_edges(n1) if e.dst == n2]
        n2_to_n1 = [e.dst for e in state.out_edges(n2) if e.dst == n1]
        if len(n2_to_n1) == 0 and len(n1_to_n2) == 0:
            raise Exception("E1")
        elif len(n2_to_n1) != 0 and len(n1_to_n2) != 0:
            raise Exception("E2")
        elif len(n2_to_n1) == 0:
            assert len(n1_to_n2) > 0
            return (n1, n2)
        else:
            assert len(n2_to_n1) > 0
            return (n2, n1)

    def annotates_memlets():
        return False
