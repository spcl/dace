# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline multi-state SDFGs. """
import ast

from copy import deepcopy as dc
from typing import Dict, List

from dace.data import find_new_name, View, Scalar, Array, ArrayView
from dace import Memlet, subsets
from dace import symbolic, dtypes
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg import InterstateEdge, SDFG, SDFGState
from dace.sdfg import utils as sdutil, infer_types
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation import transformation, helpers
from dace.properties import make_properties
from dace.sdfg.state import StateSubgraphView
from dace.sdfg.utils import get_all_view_nodes, get_view_edge


@make_properties
@transformation.single_level_sdfg_only
class InlineMultistateSDFG(transformation.SingleStateTransformation):
    """
    Inlines a multi-state nested SDFG into a top-level SDFG.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    def can_be_applied(self, state: SDFGState, expr_index, sdfg, permissive=False):
        nested_sdfg = self.nested_sdfg
        if nested_sdfg.no_inline:
            return False
        if nested_sdfg.schedule == dtypes.ScheduleType.FPGA_Device:
            return False
        if state.entry_node(nested_sdfg) is not None:
            return False

        for iedge in state.in_edges(nested_sdfg):
            if iedge.data.data is None:
                return False
            if not isinstance(iedge.src, nodes.AccessNode):
                return False

        for oedge in state.out_edges(nested_sdfg):
            if oedge.data.data is None:
                return False
            if not isinstance(oedge.dst, nodes.AccessNode):
                return False

        return True

    def apply(self, outer_state: SDFGState, sdfg: SDFG):
        nsdfg_node = self.nested_sdfg
        nsdfg: SDFG = nsdfg_node.sdfg

        if nsdfg_node.schedule != dtypes.ScheduleType.Default:
            infer_types.set_default_schedule_and_storage_types(nsdfg, [nsdfg_node.schedule])

        #######################################################
        # Collect and update top-level SDFG metadata

        # Global/init/exit code
        for loc, code in nsdfg.global_code.items():
            sdfg.append_global_code(code.code, loc)
        for loc, code in nsdfg.init_code.items():
            sdfg.append_init_code(code.code, loc)
        for loc, code in nsdfg.exit_code.items():
            sdfg.append_exit_code(code.code, loc)

        # Environments
        for nstate in nsdfg.nodes():
            for node in nstate.nodes():
                if isinstance(node, nodes.CodeNode):
                    node.environments |= nsdfg_node.environments

        #######################################################
        # Rename local variables and symbols to avoid side-effects

        self._rename_nested_symbols(nsdfg_node, sdfg)

        self._rename_nested_constants(nsdfg_node, sdfg)

        self._rename_nested_transients(nsdfg_node, sdfg)

        #######################################################
        # Fission nested SDFG into a separate state to simplify inlining

        nsdfg_state = self._isolate_nsdfg_node(nsdfg_node, sdfg, outer_state)

        #######################################################
        # Each argument of the nested SDFG becomes a view to the outer data

        self._replace_arguments_by_views(nsdfg_node, sdfg, nsdfg_state)

        #######################################################
        # Add symbols, constants and transients to the outer SDFG

        # Symbols
        for nested_symbol in nsdfg.symbols.keys():
            if nested_symbol in sdfg.symbols:
                continue

            sdfg.add_symbol(str(nested_symbol), stype=nsdfg.symbols[nested_symbol])

        # Constants
        for cstname, (csttype, cstval) in nsdfg.constants_prop.items():
            sdfg.constants_prop[cstname] = (csttype, cstval)

        # Transients
        for name, desc in nsdfg.arrays.items():
            if desc.transient and name not in sdfg.arrays:
                sdfg.add_datadesc(name, dc(desc))

        #######################################################
        # Add nested SDFG states into top-level SDFG

        # Make symbol mapping explicit
        statenames = set([s.label for s in nsdfg.all_control_flow_blocks()])
        newname = find_new_name("mapping_state", statenames)
        mapping_state = nsdfg.add_state_before(nsdfg.start_state, label=newname, is_start_state=True)
        mapping_edge = nsdfg.out_edges(mapping_state)[0]
        for inner_sym, expr in nsdfg_node.symbol_mapping.items():
            if str(inner_sym) == str(expr):
                continue
            mapping_edge.data.assignments[inner_sym] = str(expr)

        # Make unique names for states
        statenames = set([s.label for s in sdfg.all_control_flow_blocks()])
        for nstate in list(nsdfg.all_control_flow_blocks()):
            newname = find_new_name(nstate.label, statenames)
            statenames.add(newname)
            nstate.label = newname

        outer_start_state = sdfg.start_state

        sdfg.add_nodes_from(set(nsdfg.nodes()))
        for ise in nsdfg.edges():
            sdfg.add_edge(ise.src, ise.dst, ise.data)

        source = nsdfg.start_state
        sinks = nsdfg.sink_nodes()

        # Reconnect state machine
        for e in sdfg.in_edges(nsdfg_state):
            sdfg.add_edge(e.src, source, e.data)
        for e in sdfg.out_edges(nsdfg_state):
            for sink in sinks:
                sdfg.add_edge(sink, e.dst, dc(e.data))
                # Redirect sink incoming edges with a `False` condition to e.dst (return statements)
                for e2 in sdfg.in_edges(sink):
                    if e2.data.condition_sympy() == False:
                        sdfg.add_edge(e2.src, e.dst, InterstateEdge())

        # Modify start state as necessary
        if outer_start_state is nsdfg_state:
            sdfg.start_state = sdfg.node_id(source)

        # Replace nested SDFG parents with new SDFG
        for nstate in nsdfg.nodes():
            nstate.parent = sdfg
            for node in nstate.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent_sdfg = sdfg
                    node.sdfg.parent_nsdfg_node = node

        #######################################################
        # Remove nested SDFG and state
        nsdfg_state.remove_node(nsdfg_node)
        sdfg.remove_node(nsdfg_state)

        sdfg._cfg_list = sdfg.reset_cfg_list()

        return nsdfg.nodes()


    def _rename_nested_symbols(self, nsdfg_node: nodes.NestedSDFG, sdfg: SDFG) -> None:
        nsdfg = nsdfg_node.sdfg
        
        # Collect symbols that are overwritten
        inner_assignments = set()

        for inner_sym, expr in nsdfg_node.symbol_mapping.items():
            if str(inner_sym) == str(expr):
                continue
            inner_assignments.add(inner_sym)
        
        for e in nsdfg.edges():
            inner_assignments |= set(e.data.assignments.keys())

        outer_symbols = {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.arrays.keys())
        assignments_to_replace = inner_assignments
        
        sym_replacements: Dict[str, str] = {}
        for assign in assignments_to_replace:
            newname = find_new_name(
                assign, outer_symbols | set(sym_replacements.values()) | {str(sym) for sym in nsdfg.symbols.keys()}
            )
            sym_replacements[assign] = newname
        
        nsdfg.replace_dict(sym_replacements)
        for sym, new_sym in sym_replacements.items():
            if sym in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[new_sym] = nsdfg_node.symbol_mapping[sym]
                del nsdfg_node.symbol_mapping[sym]


    def _rename_nested_transients(self, nsdfg_node: nodes.NestedSDFG, sdfg: SDFG):
        nsdfg = nsdfg_node.sdfg

        outer_symbols = set(sdfg.arrays.keys()) | {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.constants.keys())
        
        transient_replacements: Dict[str, str] = {}
        for nstate in nsdfg.nodes():
            for node in nstate.nodes():
                if isinstance(node, nodes.AccessNode):
                    datadesc = nsdfg.arrays[node.data]
                    if isinstance(datadesc, View) or not datadesc.transient:
                        continue

                    if node.data not in transient_replacements:
                        new_name = node.data
                        name = find_new_name(new_name, outer_symbols | set(nsdfg.arrays.keys() | set(transient_replacements.values())))
                        transient_replacements[node.data] = name

            for edge in nstate.edges():
                if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
                    if edge.data.data is not None:
                        datadesc = nsdfg.arrays[edge.data.data]
                        if isinstance(datadesc, View) or not datadesc.transient:
                            continue

                        if edge.data.data not in transient_replacements:
                            new_name = edge.data.data
                            name = find_new_name(new_name, outer_symbols | set(nsdfg.arrays.keys() | set(transient_replacements.values())))
                            transient_replacements[edge.data.data] = name

        replace_datadesc_names(nsdfg, transient_replacements)


    def _rename_nested_constants(self, nsdfg_node: nodes.NestedSDFG, sdfg: SDFG) -> None:
        nsdfg = nsdfg_node.sdfg

        outer_symbols = set(sdfg.arrays.keys()) | {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.constants.keys())

        replacements = {}
        for cstname, (csttype, cstval) in nsdfg.constants_prop.items():
            if cstname in sdfg.constants:
                newname = find_new_name(cstname, outer_symbols | set(nsdfg.constants() | set(replacements.values())))
                replacements[cstname] = newname

        symbolic.safe_replace(replacements, lambda m: replace_datadesc_names(nsdfg, m), value_as_string=True)


    def _isolate_nsdfg_node(self, nsdfg_node: nodes.NestedSDFG, sdfg: SDFG, outer_state: SDFGState) -> SDFGState:
        # Push nsdfg plus childs into new state
        nsdfg_state = helpers.state_fission_after(outer_state, nsdfg_node)

        # Split nsdfg from its childs
        direct_subgraph = set()
        direct_subgraph.add(nsdfg_node)
        direct_subgraph.update(nsdfg_state.predecessors(nsdfg_node))
        direct_subgraph.update(nsdfg_state.successors(nsdfg_node))

        for node in list(direct_subgraph):
            if isinstance(node, nodes.AccessNode) and isinstance(sdfg.arrays[node.data], View):
                for view_node in get_all_view_nodes(nsdfg_state, node):
                    direct_subgraph.add(view_node)

        direct_subgraph = StateSubgraphView(nsdfg_state, direct_subgraph)
        new_state = helpers.state_fission(direct_subgraph)
        return new_state

    def _replace_arguments_by_views(
        self,
        nsdfg_node: nodes.NestedSDFG,
        sdfg: SDFG,
        outer_state: SDFGState
    ) -> None:
        nsdfg = nsdfg_node.sdfg

        used_symbol_names = set(sdfg.arrays.keys()) | {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.constants.keys()) | {str(sym) for sym in nsdfg.symbols.keys()} | set(nsdfg.arrays.keys())

        #######################################################
        # Make argument names unique to avoid naming clashes

        replacements = {}
        for inp in list(nsdfg_node.in_connectors):
            new_name = find_new_name(inp, used_symbol_names)
            used_symbol_names.add(new_name)
            replacements[inp] = new_name

        for outp in list(nsdfg_node.out_connectors):
            if outp in replacements:
                continue

            new_name = find_new_name(outp, used_symbol_names)
            used_symbol_names.add(new_name)
            replacements[outp] = new_name

        replace_datadesc_names(nsdfg, replacements)

        for state in nsdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    for iconn in list(node.in_connectors):
                        if iconn in replacements:
                            iedges = list(state.in_edges_by_connector(node, iconn))
                            node.add_in_connector(replacements[iconn])

                            for iedge in iedges:
                                state.add_edge(iedge.src, iedge.src_conn, iedge.dst, replacements[iconn], dc(iedge.data))
                                state.remove_edge(iedge)
                            
                            node.remove_in_connector(iconn)

        input_memlets: Dict[str, MultiConnectorEdge] = {}
        for e in list(outer_state.in_edges(nsdfg_node)):
            if e.dst_conn not in replacements:
                inner_data = e.dst_conn
                input_memlets[inner_data] = e
                continue

            outer_state.remove_edge_and_connectors(e)
            
            if replacements[e.dst_conn] not in nsdfg_node.in_connectors:
                nsdfg_node.add_in_connector(replacements[e.dst_conn])
            e_ = outer_state.add_edge(e.src, e.src_conn, e.dst, replacements[e.dst_conn], dc(e.data))

            inner_data = e_.dst_conn
            input_memlets[inner_data] = e_

        output_memlets: Dict[str, MultiConnectorEdge] = {}
        for e in list(outer_state.out_edges(nsdfg_node)):
            if e.src_conn not in replacements:
                inner_data = e.src_conn
                input_memlets[inner_data] = e
                continue

            outer_state.remove_edge_and_connectors(e)
            
            if replacements[e.src_conn] not in nsdfg_node.out_connectors:
                nsdfg_node.add_out_connector(replacements[e.src_conn], force=True)
            e_ = outer_state.add_edge(e.src, replacements[e.src_conn], e.dst, e.dst_conn, dc(e.data))

            inner_data = e_.src_conn
            output_memlets[inner_data] = e_

        #######################################################
        # Replace arguments by full outer data and create a view on the inside

        for argument, in_edge in input_memlets.items():
            # Replace argument by view
            del nsdfg.arrays[argument]
            nsdfg.arrays[argument] = View.view(sdfg.arrays[in_edge.src.data])

            # Add outer data to nsdfg
            outer_node: nodes.AccessNode = in_edge.src
            outer_nodes = get_all_view_nodes(outer_state, outer_node)
            for viewed_node in outer_nodes:
                outer_desc = dc(sdfg.arrays[viewed_node.data])
                outer_desc.transient = (viewed_node != outer_nodes[-1])
                if viewed_node.data in nsdfg.arrays:
                    del nsdfg.arrays[viewed_node.data]
                nsdfg.add_datadesc(viewed_node.data, outer_desc)

            # Provide full desc as argument
            outer_node = outer_nodes[-1]
            nsdfg_node.add_in_connector(outer_node.data)
            outer_state.add_edge(
                outer_node,
                None,
                in_edge.dst,
                outer_node.data,
                Memlet.from_array(outer_node.data, sdfg.arrays[outer_node.data])
            )

        # Replace all outputs by the full desc and add a view desc
        for argument, out_edge in output_memlets.items():
            # Replace argument by view
            if argument not in input_memlets:
                del nsdfg.arrays[argument]
                nsdfg.arrays[argument] = View.view(sdfg.arrays[out_edge.dst.data])

            # Add outer data to nsdfg
            outer_node: nodes.AccessNode = out_edge.dst
            outer_nodes = get_all_view_nodes(outer_state, outer_node)
            for viewed_node in outer_nodes:
                outer_desc = dc(sdfg.arrays[viewed_node.data])
                outer_desc.transient = (viewed_node != outer_nodes[-1])
                if viewed_node.data in nsdfg.arrays:
                    del nsdfg.arrays[viewed_node.data]
                nsdfg.add_datadesc(viewed_node.data, outer_desc)

            # Provide full desc as argument
            outer_node = outer_nodes[-1]
            nsdfg_node.add_out_connector(outer_node.data, force=True)
            outer_state.add_edge(
                out_edge.src,
                outer_node.data,
                outer_node,
                None,
                Memlet.from_array(outer_node.data, sdfg.arrays[outer_node.data])
            )

        #######################################################
        # Dataflow: Extend access nodes with towers of views

        for nstate in nsdfg.nodes():
            for node in list(nstate.nodes()):
                if isinstance(node, nodes.AccessNode):
                    if node.data not in input_memlets and node.data not in output_memlets:
                        continue

                    out_edges = nstate.out_edges(node)
                    in_edges = nstate.in_edges(node)
                    
                    # Split node
                    if in_edges and out_edges:
                        out_node = node
                        in_node = nstate.add_access(node.data)
                        for iedge in in_edges:
                            nstate.add_edge(iedge.src, iedge.src_conn, in_node, iedge.dst_conn, dc(iedge.data))
                            nstate.remove_edge(iedge)
                    else:
                        in_node = node
                        out_node = node

                    if out_edges:
                        # Add directly viewed node
                        outer_node = input_memlets[node.data].src
                        viewed_node = nstate.add_access(outer_node.data)
                        nstate.add_edge(
                            viewed_node,
                            None,
                            out_node,
                            "views",
                            dc(input_memlets[node.data].data)
                        )

                        # Add tower of views
                        viewing_node = viewed_node
                        outer_nodes = get_all_view_nodes(outer_state, outer_node)
                        outer_node_ = outer_node
                        for outer_node in outer_nodes[1:]:
                            viewed_node = nstate.add_access(outer_node.data)
                            view_edge = list(outer_state.edges_between(outer_node, outer_node_))[0]
                            nstate.add_edge(
                                viewed_node,
                                view_edge.src_conn,
                                viewing_node,
                                view_edge.dst_conn,
                                dc(view_edge.data)
                            )
                            viewing_node = viewed_node
                            outer_node_ = outer_node

                        last_out_node = viewing_node
                    
                    if in_edges:
                        # Add directly viewed node
                        outer_node = output_memlets[node.data].dst
                        viewed_node = nstate.add_access(outer_node.data)
                        nstate.add_edge(
                            in_node,
                            "views",
                            viewed_node,
                            None,
                            dc(output_memlets[node.data].data)
                        )

                        # Add tower of views
                        viewing_node = viewed_node
                        outer_nodes = get_all_view_nodes(outer_state, outer_node)
                        outer_node_ = outer_node
                        for outer_node in outer_nodes[1:]:
                            viewed_node = nstate.add_access(outer_node.data)
                            view_edge = list(outer_state.edges_between(outer_node_, outer_node))[0]
                            nstate.add_edge(
                                viewing_node,
                                view_edge.src_conn,
                                viewed_node,
                                view_edge.dst_conn,
                                dc(view_edge.data)
                            )
                            viewing_node = viewed_node
                            outer_node_ = outer_node
                        
                        last_in_node = viewed_node

                    # Connect splitted nodes
                    if in_edges and out_edges:
                        for oedge in nstate.out_edges(last_out_node):
                            nstate.add_edge(last_in_node, oedge.src_conn, oedge.dst, oedge.dst_conn, dc(oedge.data))
                            nstate.remove_edge(oedge)

                        nstate.remove_node(last_out_node)

        #######################################################
        # Control-flow: Replace old arguments by the true data
        # Hint: Old arguments are now views and cannot be used here

        # Find arguments to be replaced
        args_used_in_assignments = set()
        for edge in nsdfg.edges():
            args_used_in_assignments |= edge.data.free_symbols
        
        args_used_in_assignments &= input_memlets.keys()

        # Convert argument into path to full data
        for arg in args_used_in_assignments:
            in_edge = input_memlets[arg]
            if not isinstance(nsdfg.arrays[in_edge.data.data], View):
                if isinstance(nsdfg.arrays[in_edge.data.data], Array):
                    first_data = in_edge.data.data + "[" + str(in_edge.data.subset) + "]"
                else:
                    first_data = in_edge.data.data

                data_path = [first_data]
            else:
                view_nodes = get_all_view_nodes(outer_state, in_edge.src)[::-1]
                current_node = view_nodes[0]
                current_desc = sdfg.arrays[current_node.data]
                data_path = [current_node.data]
                for i in range(1, len(view_nodes)):
                    view_node = view_nodes[i]
                    view_edge = get_view_edge(outer_state, view_node)

                    if "." in view_edge.data.data:
                        member_name = view_edge.data.data.split(".")[-1]
                        memlet_part = member_name
                        if i < len(view_nodes) - 1:
                            memlet_part += "[" + view_edge.data.subset.__str__() + "]"
                        
                        data_path.append(memlet_part)

                        current_node = view_node
                        current_desc = current_desc.members[member_name]
                    else:
                        # View on a subset
                        if view_edge.data.subset != Memlet.from_array(current_node.data, current_desc).subset:
                            # TODO: Non-trivial, memlet offsetting required
                            raise NotImplementedError
                        else:
                            # We can simply skip
                            continue

            data_path = ".".join(data_path)
            for edge in nsdfg.edges():
                edge.data.replace(arg, data_path)

        #######################################################
        # Remove old edges to the nested SDFG

        for argument, edge in input_memlets.items():
            outer_state.remove_memlet_path(edge)

        for argument, edge in output_memlets.items():
            outer_state.remove_memlet_path(edge)


