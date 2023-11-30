# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline multi-state SDFGs. """
import ast

from copy import deepcopy as dc
from typing import Dict, List

from dace.data import find_new_name, View
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

        for oedge in state.out_edges(nested_sdfg):
            if oedge.data.data is None:
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

        # Make unique names for states
        statenames = set(s.label for s in sdfg.nodes())
        for nstate in nsdfg.nodes():
            if nstate.label in statenames:
                newname = find_new_name(nstate.label, statenames)
                statenames.add(newname)
                nstate.label = newname

        outer_start_state = sdfg.start_state

        sdfg.add_nodes_from(nsdfg.nodes())
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
        sdfg.remove_node(nsdfg_state)

        sdfg._cfg_list = sdfg.reset_cfg_list()

        return nsdfg.nodes()


    def _rename_nested_symbols(self, nsdfg_node: nodes.NestedSDFG, sdfg: SDFG) -> None:
        nsdfg = nsdfg_node.sdfg

        # Two-step replacement (N -> __dacesym_N --> map[N]) to avoid clashes
        symbolic.safe_replace(nsdfg_node.symbol_mapping, nsdfg.replace_dict)
        
        # Collect symbols that are overwritten
        inner_assignments = set()
        for e in nsdfg.edges():
            inner_assignments |= e.data.assignments.keys()

        # Replace only those symbols
        outer_assignments = set()
        for e in sdfg.edges():
            outer_assignments |= e.data.assignments.keys()
        
        outer_symbols = {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.arrays.keys()) | outer_assignments

        assignments_to_replace = inner_assignments & outer_symbols
        assignments_to_replace -= (set(nsdfg_node.in_connectors.keys()) | set(nsdfg_node.out_connectors.keys()))
        
        sym_replacements: Dict[str, str] = {}
        for assign in assignments_to_replace:
            newname = find_new_name(
                assign, outer_symbols | set(sym_replacements.values()) | {str(sym) for sym in nsdfg.symbols.keys()}
            )
            sym_replacements[assign] = newname
        
        nsdfg.replace_dict(sym_replacements)


    def _rename_nested_transients(self, nsdfg_node: nodes.NestedSDFG, sdfg: SDFG):
        nsdfg = nsdfg_node.sdfg

        outer_symbols = set(sdfg.arrays.keys()) | {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.constants.keys())
        
        replacements: Dict[str, str] = {}
        for nstate in nsdfg.nodes():
            for node in nstate.nodes():
                if isinstance(node, nodes.AccessNode):
                    datadesc = nsdfg.arrays[node.data]
                    if node.data not in replacements and datadesc.transient:
                        new_name = node.data
                        name = find_new_name(new_name, outer_symbols | set(nsdfg.arrays.keys() | set(replacements.values())))
                        replacements[node.data] = name

            for edge in nstate.edges():
                if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
                    if edge.data.data is not None:
                        datadesc = nsdfg.arrays[edge.data.data]
                        if edge.data.data not in replacements and datadesc.transient:
                            new_name = edge.data.data
                            name = find_new_name(new_name, outer_symbols | set(nsdfg.arrays.keys() | set(replacements.values())))
                            replacements[edge.data.data] = name

        symbolic.safe_replace(replacements, lambda m: replace_datadesc_names(nsdfg, m), value_as_string=True)


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
        nsdfg_state = helpers.state_fission_after(sdfg, outer_state, nsdfg_node)
        
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
        return helpers.state_fission(sdfg, direct_subgraph)

    def _replace_arguments_by_views(
        self,
        nsdfg_node: nodes.NestedSDFG,
        sdfg: SDFG,
        outer_state: SDFGState
    ) -> None:
        nsdfg = nsdfg_node.sdfg

        used_symbol_names = set(sdfg.arrays.keys()) | {str(sym) for sym in sdfg.symbols.keys()} | set(sdfg.constants.keys()) | {str(sym) for sym in nsdfg.symbols.keys()} | set(nsdfg.arrays.keys())

        # Make arguments unique in inner and outer sdfg
        replacements = {}
        for inp in list(nsdfg_node.in_connectors):
            # TODO: Integer argument handling? Pass by value/reference
            desc = nsdfg.arrays[inp]
            if desc.dtype in [
                dtypes.bool, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64
            ]:
                continue

            new_name = find_new_name(inp, used_symbol_names)
            used_symbol_names.add(new_name)
            replacements[inp] = new_name

        for outp in list(nsdfg_node.out_connectors):
            if outp in replacements:
                continue

            # TODO: Integer argument handling? Pass by value/reference
            desc = nsdfg.arrays[outp]
            if desc.dtype in [
                dtypes.bool, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64
            ]:
                continue

            new_name = find_new_name(outp, used_symbol_names)
            used_symbol_names.add(new_name)
            replacements[outp] = new_name

        nsdfg.replace_dict(replacements)

        # Re-connect arguments
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

        # Replace all inputs by the full desc and add a view desc
        input_viewed_nodes = {}
        for argument, in_edge in input_memlets.items():
            # Replace argument by view
            inner_desc = nsdfg.arrays[argument]
            if inner_desc.dtype in [
                dtypes.bool, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64
            ]:
                if argument not in sdfg.arrays:
                    desc = dc(inner_desc)
                    desc.transient = True
                    sdfg.add_datadesc(argument, desc)
                continue

            del nsdfg.arrays[argument]
            _ = nsdfg.add_view(argument,
                                inner_desc.shape,
                                inner_desc.dtype,
                                strides=inner_desc.strides,
                                offset=inner_desc.offset,
                                debuginfo=inner_desc.debuginfo,
                                find_new_name=False)

            # Add outer data to nsdfg
            outer_node: nodes.AccessNode = in_edge.src
            input_viewed_nodes[argument] = [(outer_node, in_edge)]

            if outer_node.data in nsdfg.arrays:
                del nsdfg.arrays[outer_node.data]
            
            outer_desc = dc(sdfg.arrays[outer_node.data])
            outer_desc.transient = False
            nsdfg.add_datadesc(outer_node.data, outer_desc)

            # If view, add intermediate views
            if isinstance(sdfg.arrays[outer_node.data], View):
                outer_nodes = get_all_view_nodes(outer_state, outer_node)[1:]
                for viewed_node in outer_nodes:
                    view_edge = get_view_edge(outer_state, viewed_node)
                    input_viewed_nodes[argument].append(
                        (viewed_node, view_edge)
                    )

                    if viewed_node.data in nsdfg.arrays:
                        del nsdfg.arrays[viewed_node.data]

                    outer_desc = dc(sdfg.arrays[viewed_node.data])
                    outer_desc.transient = False
                    nsdfg.add_datadesc(viewed_node.data, outer_desc)

                outer_node = outer_nodes[-1]

            # Provide full desc as argument
            nsdfg_node.add_in_connector(outer_node.data)
            outer_state.add_edge(
                outer_node,
                None,
                in_edge.dst,
                outer_node.data,
                Memlet.from_array(outer_node.data, sdfg.arrays[outer_node.data])
            )

        # Replace all outputs by the full desc and add a view desc
        output_viewed_nodes = {}
        for argument, out_edge in output_memlets.items():
            # Replace argument by view
            inner_desc = nsdfg.arrays[argument]
            if inner_desc.dtype in [
                dtypes.bool, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64
            ]:
                if argument not in input_memlets and argument not in sdfg.arrays:
                    desc = dc(inner_desc)
                    desc.transient = True
                    sdfg.add_datadesc(argument, desc)
                continue

            if argument not in input_memlets:
                del nsdfg.arrays[argument]
                _ = nsdfg.add_view(argument,
                                    inner_desc.shape,
                                    inner_desc.dtype,
                                    strides=inner_desc.strides,
                                    offset=inner_desc.offset,
                                    debuginfo=inner_desc.debuginfo,
                                    find_new_name=False)

            # Add outer data to nsdfg
            outer_node: nodes.AccessNode = out_edge.dst
            output_viewed_nodes[argument] = [(outer_node, out_edge)]
            
            if outer_node.data in nsdfg.arrays:
                del nsdfg.arrays[outer_node.data]

            outer_desc = dc(sdfg.arrays[outer_node.data])
            outer_desc.transient = False
            nsdfg.add_datadesc(outer_node.data, outer_desc)

            # If view, add intermediate views
            if isinstance(sdfg.arrays[outer_node.data], View):
                outer_nodes = get_all_view_nodes(outer_state, outer_node)[1:]
                for viewed_node in outer_nodes:
                    view_edge = get_view_edge(outer_state, viewed_node)
                    output_viewed_nodes[argument].append(
                        (viewed_node, view_edge)
                    )
                
                    if viewed_node.data in nsdfg.arrays:
                        del nsdfg.arrays[viewed_node.data]

                    outer_desc = dc(sdfg.arrays[viewed_node.data])
                    outer_desc.transient = False
                    nsdfg.add_datadesc(viewed_node.data, outer_desc)

                outer_node = outer_nodes[-1]

            # Provide full desc as argument
            nsdfg_node.add_out_connector(outer_node.data, force=True)
            outer_state.add_edge(
                out_edge.src,
                outer_node.data,
                outer_node,
                None,
                Memlet.from_array(outer_node.data, sdfg.arrays[outer_node.data])
            )

        # Extend data accesses with views
        for nstate in nsdfg.nodes():
            for node in list(nstate.nodes()):
                if isinstance(node, nodes.AccessNode):
                    if node.data in input_viewed_nodes:
                        viewed_nodes = input_viewed_nodes[node.data]
                    elif node.data in output_viewed_nodes:
                        viewed_nodes = output_viewed_nodes[node.data]
                    else:
                        continue

                    out_edges = nstate.out_edges(node)
                    in_edges = nstate.in_edges(node)
                    
                    # Split node
                    if in_edges and out_edges:
                        raise NotImplementedError

                    if out_edges and not in_edges:
                        viewing_node = node
                        for viewed_node, viewed_edge in viewed_nodes:
                            viewed_node_ = nstate.add_access(viewed_node.data)
                            viewed_memlet = dc(viewed_edge.data)
                            nstate.add_edge(
                                viewed_node_,
                                None,
                                viewing_node,
                                "views",
                                viewed_memlet
                            )
                            viewing_node = viewed_node_
                    elif in_edges and not out_edges:
                        viewing_node = node
                        for viewed_node, viewed_edge in viewed_nodes:
                            viewed_node_ = nstate.add_access(viewed_node.data)
                            viewed_memlet = dc(viewed_edge.data)
                            nstate.add_edge(
                                viewing_node,
                                "views",
                                viewed_node_,
                                None,
                                viewed_memlet
                            )
                            viewing_node = viewed_node_

        for argument, edge in input_memlets.items():
            outer_state.remove_memlet_path(edge)

        for argument, edge in output_memlets.items():
            outer_state.remove_memlet_path(edge)

        # Replace original arguments on interstate edges
        for arg, views in input_viewed_nodes.items():
            view_edges = list(tuple(zip(*views))[1])
            view_edges = view_edges[::-1]
            memlet_chain = []
            
            view_name = view_edges[0].data.data
            if len(view_edges) > 1:
                view_name += "[" + view_edges[0].data.subset.__str__() + "]"
            memlet_chain.append(view_name)

            for i in range(1, len(view_edges)):
                view_edge = view_edges[i]
                view_name = view_edge.data.data
                view_name = view_name.split(".")[-1]
                if i < len(view_edges) - 1:
                    view_name += "[" + view_edge.data.subset.__str__() + "]"
                memlet_chain.append(view_name)

            viewed_arg = ".".join(memlet_chain)
            for edge in nsdfg.edges():
                edge.data.replace(arg, viewed_arg)

