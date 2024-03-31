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
class ArgumentFlattening(transformation.SingleStateTransformation):
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    def annotates_memlets(self) -> bool:
        return True

    def _candidates(self, nested_sdfg: nodes.NestedSDFG):
        nsdfg = self.nested_sdfg.sdfg

        candidates = set()
        for con in nested_sdfg.in_connectors:
            desc = nested_sdfg.sdfg.arrays[con]
            if isinstance(desc, dt.Structure):
                candidates.add(con)
        
        for con in nested_sdfg.out_connectors:
            desc = nested_sdfg.sdfg.arrays[con]
            if isinstance(desc, dt.Structure):
                candidates.add(con)

        # Remove candidates which are not only viewed for members
        used = set()
        for nstate in nsdfg.states():
            for dnode in nstate.data_nodes():
                if dnode.data not in candidates:
                    continue

                removed = False
                for oedge in nstate.out_edges(dnode):
                    if not oedge.data.data.startswith(dnode.data + "."):
                        candidates.remove(dnode.data)
                        removed = True
                        break
                
                if removed:
                    continue

                for iedge in nstate.in_edges(dnode):
                    if not iedge.data.data.startswith(dnode.data + "."):
                        candidates.remove(dnode.data)
                        break
                
                used.add(dnode.data)

        # Check usages in interstate edges
        for edge in nsdfg.edges():
            for assignment in edge.data.assignments.values():
                for candidate in list(candidates):
                    if candidate in assignment:
                        if (candidate + ".") in assignment or (candidate + "[0].") in assignment:
                            used.add(candidate)
        
        return used & candidates


    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):                
        
        # Potential candidate arguments
        candidates = self._candidates(self.nested_sdfg)
        if candidates:
            return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg_node = self.nested_sdfg
        nsdfg = self.nested_sdfg.sdfg

        candidates = self._candidates(nsdfg_node)
        candidate = candidates.pop()

        #######################################################
        # Find referenced members

        used_members = set()
        new_inputs = set()
        new_outputs = set()

        # Interstate edges
        search_pattern1 = candidate + "."
        search_pattern2 = candidate + "[0]."
        for edge in nsdfg.edges():
            if not edge.data.is_unconditional():
                condition = edge.data.condition.as_string
                if search_pattern1 in condition:
                    occurences = condition.split(search_pattern1)[1:]
                    for occ in occurences:
                        data = ""
                        for k in occ:
                            if k == "." or k == " " or k == "[" or k == ")":
                                break
                            data += k

                        used_members.add(data) 
                        new_inputs.add(data)
                if search_pattern2 in condition:
                    occurences = condition.split(search_pattern2)[1:]
                    for occ in occurences:
                        data = ""
                        for k in occ:
                            if k == "." or k == " " or k == "[" or k == ")":
                                break
                            data += k

                        used_members.add(data) 
                        new_inputs.add(data)

            for assignment in edge.data.assignments.values():
                if search_pattern1 in assignment:
                    occurences = assignment.split(search_pattern1)[1:]
                    for occ in occurences:
                        data = ""
                        for k in occ:
                            if k == "." or k == " " or k == "[" or k == ")":
                                break
                            data += k

                        used_members.add(data) 
                        new_inputs.add(data)

                if search_pattern2 in assignment:
                    occurences = assignment.split(search_pattern2)[1:]
                    for occ in occurences:
                        data = ""
                        for k in occ:
                            if k == "." or k == " " or k == "[" or k == ")":
                                break
                            data += k

                        used_members.add(data) 
                        new_inputs.add(data)

        # Dataflow
        member_views = {}
        for nstate in nsdfg.states():
            for dnode in nstate.data_nodes():
                if dnode.data != candidate:
                    continue

                for oedge in nstate.out_edges(dnode):
                    assert oedge.dst_conn == "views"

                    data = oedge.data.data.split(".")[1]
                    used_members.add(data) 

                    if data not in member_views:
                        member_views[data] = set()
                    member_views[data].add(oedge.dst.data)
                    
                    new_inputs.add(data)

                for iedge in nstate.in_edges(dnode):
                    assert iedge.src_conn == "views"

                    data = iedge.data.data.split(".")[1]
                    used_members.add(data) 

                    if data not in member_views:
                        member_views[data] = set()
                    member_views[data].add(iedge.src.data)

                    new_outputs.add(data)

        #######################################################
        # Update arguments

        input_access_node = None
        if candidate in nsdfg_node.in_connectors:
            input_edge = list(state.in_edges_by_connector(nsdfg_node, connector=candidate))[0]
            input_edge_path = []
            input_node_path = []
            for edge in state.memlet_path(input_edge):
                if input_access_node is None:
                    if isinstance(edge.src, nodes.AccessNode) and edge.src.data == input_edge.data.data:
                        input_access_node = edge.src
                        input_edge_path.append(edge)
                        input_node_path.append(edge.dst)
                    else:
                        continue
                else:
                    input_edge_path.append(edge)
                    input_node_path.append(edge.dst)

        output_access_node = None
        if candidate in nsdfg_node.out_connectors:
            output_edge = list(state.out_edges_by_connector(nsdfg_node, connector=candidate))[0]
            output_edge_path = []
            output_node_path = []
            for edge in state.memlet_path(output_edge):
                output_edge_path.append(edge)
                output_node_path.append(edge.src)

                if output_access_node is None:
                    if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == output_edge.data.data:
                        output_access_node = edge.dst
                        break

        for member in used_members:
            # Add view also to the outer sdfg
            if input_access_node is not None:
                desc = sdfg.arrays[input_access_node.data].members[member]
            else:
                desc = sdfg.arrays[output_access_node.data].members[member]

            # If not there, add it
            if member not in member_views:
                nsdfg.add_datadesc(f"flat_{candidate}_{member}", dt.View.view(desc))
                member_views[member] = set([f"flat_{candidate}_{member}"])

            for view_name in member_views[member]:
                if view_name not in sdfg.arrays:
                    sdfg.add_datadesc(view_name, dt.View.view(desc))
                
                # Convert view to proper data in the inner sdfg
                del nsdfg.arrays[view_name]
                inner_desc = copy.deepcopy(desc)
                inner_desc.transient = False
                nsdfg.add_datadesc(view_name, inner_desc)
                subset = "[" + str(Memlet.from_array(view_name, inner_desc).subset) + "]"

                if member in new_inputs:
                    nsdfg_node.add_in_connector(view_name, force=True)

                    input_member_node = state.add_access(view_name)
                    memlet = Memlet(expr=f"{input_access_node.data}.{member}{subset}")
                    state.add_edge(input_access_node, None, input_member_node, "views", memlet)

                    inner_memlet = Memlet(expr=f"{view_name}{subset}")
                    state.add_memlet_path(
                        input_member_node, *input_node_path,
                        memlet=copy.deepcopy(inner_memlet),
                        src_conn=None,
                        dst_conn=view_name
                    )

                if member in new_outputs:
                    nsdfg_node.add_out_connector(view_name, force=True)

                    output_member_node = state.add_access(view_name)
                    memlet = Memlet(expr=f"{output_access_node.data}.{member}{subset}")
                    state.add_edge(output_member_node, "views", output_access_node, None, copy.deepcopy(memlet))

                    inner_memlet = Memlet(expr=f"{view_name}{subset}")
                    state.add_memlet_path(
                        *output_node_path, output_member_node,
                        memlet=copy.deepcopy(inner_memlet),
                        src_conn=view_name,
                        dst_conn=None
                    )

        if input_access_node is not None:
            nsdfg_node.remove_in_connector(candidate)
            
            for edge in input_edge_path:
                state.remove_edge(edge)
                if edge.src_conn is not None:
                    edge.src.remove_out_connector(edge.src_conn)
                if edge.dst_conn is not None:
                    edge.dst.remove_in_connector(edge.dst_conn)
        
        if output_access_node is not None:
            if candidate in nsdfg_node.out_connectors:
                nsdfg_node.remove_out_connector(candidate)
            
            for edge in output_edge_path:
                state.remove_edge(edge)
                if edge.src_conn is not None:
                    edge.src.remove_out_connector(edge.src_conn)
                if edge.dst_conn is not None:
                    edge.dst.remove_in_connector(edge.dst_conn)

        del nsdfg.arrays[candidate]

        #######################################################
        # Replace uses of members
        
        for nstate in nsdfg.states():
            for dnode in list(nstate.data_nodes()):
                if dnode.data == candidate:
                    for oedge in nstate.out_edges(dnode):
                        oedge.dst.remove_in_connector("views")
                    for iedge in nstate.in_edges(dnode):
                        iedge.src.remove_out_connector("views")
                    
                    nstate.remove_node(dnode)

        replacements = {}
        for member in used_members:
            replacements[f"{candidate}.{member}"] = list(member_views[member])[0]
            replacements[f"{candidate}[0].{member}"] = list(member_views[member])[0]

        for edge in nsdfg.edges():
            if not edge.data.is_unconditional():
                condition = edge.data.condition.as_string
                for key, val in replacements.items():
                    if key in condition:
                        condition = condition.replace(key, val)
                edge.data.condition.as_string = condition

            for sym, assign in list(edge.data.assignments.items()):
                for key, val in replacements.items():
                    if key in assign:
                        edge.data.assignments[sym] = assign.replace(key, val)
