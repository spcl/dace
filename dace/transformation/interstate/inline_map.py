# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
import sympy

from dace.sdfg import SDFG, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState, SubgraphView
from dace.transformation import transformation
from dace.properties import make_properties
from dace import symbolic, Memlet
from dace.data import View, StructureView, ArrayView
from dace.sdfg.replace import replace_datadesc_names

@make_properties
class InlineMap(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        # xform_single_state = InlineMapSingleState()
        # xform_single_state.setup_match(
        #     sdfg,
        #     sdfg.cfg_id,
        #     sdfg.node_id(state),
        #     {
        #         InlineMapSingleState.nested_sdfg: state.node_id(self.nested_sdfg),
        #         InlineMapSingleState.map_entry: state.node_id(self.map_entry),
        #     },
        #     0
        # )
        # if xform_single_state.can_be_applied(state, 0, sdfg, False):
        #     return True

        xform_assignment = InlineMapByAssignment()
        xform_assignment.setup_match(
            sdfg,
            sdfg.cfg_id,
            sdfg.node_id(state),
            {
                InlineMapByAssignment.nested_sdfg: state.node_id(self.nested_sdfg),
                InlineMapByAssignment.map_entry: state.node_id(self.map_entry),
            },
            0
        )
        if xform_assignment.can_be_applied(state, 0, sdfg, False):
            return True

        # xform_conditions = InlineMapByConditions()
        # xform_conditions.setup_match(
        #     sdfg,
        #     sdfg.cfg_id,
        #     sdfg.node_id(state),
        #     {
        #         InlineMapByConditions.nested_sdfg: state.node_id(self.nested_sdfg),
        #         InlineMapByConditions.map_entry: state.node_id(self.map_entry),
        #     },
        #     0
        # )
        # if xform_conditions.can_be_applied(state, 0, sdfg, False):
        #     return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        # xform_single_state = InlineMapSingleState()
        # xform_single_state.setup_match(
        #     sdfg,
        #     sdfg.cfg_id,
        #     sdfg.node_id(state),
        #     {
        #         InlineMapSingleState.nested_sdfg: state.node_id(self.nested_sdfg),
        #         InlineMapSingleState.map_entry: state.node_id(self.map_entry),
        #     },
        #     0
        # )
        # if xform_single_state.can_be_applied(state, 0, sdfg, False):
        #     return xform_single_state.apply(state, sdfg)

        xform_assignment = InlineMapByAssignment()
        xform_assignment.setup_match(
            sdfg,
            sdfg.cfg_id,
            sdfg.node_id(state),
            {
                InlineMapByAssignment.nested_sdfg: state.node_id(self.nested_sdfg),
                InlineMapByAssignment.map_entry: state.node_id(self.map_entry),
            },
            0
        )
        if xform_assignment.can_be_applied(state, 0, sdfg, False):
            return xform_assignment.apply(state, sdfg)

        # xform_conditions = InlineMapByConditions()
        # xform_conditions.setup_match(
        #     sdfg,
        #     sdfg.cfg_id,
        #     sdfg.node_id(state),
        #     {
        #         InlineMapByConditions.nested_sdfg: state.node_id(self.nested_sdfg),
        #         InlineMapByConditions.map_entry: state.node_id(self.map_entry),
        #     },
        #     0
        # )
        # if xform_conditions.can_be_applied(state, 0, sdfg, False):
        #     return xform_conditions.apply(state, sdfg)

        return None

@make_properties
class InlineMapByAssignment(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        for iedge in state.in_edges(self.map_entry):
            if iedge.data.data is None:
                return False
            if not isinstance(iedge.src, nodes.AccessNode):
                return False

        for oedge in state.out_edges(state.exit_node(self.map_entry)):
            if oedge.data.data is None:
                return False
            if not isinstance(oedge.dst, nodes.AccessNode):
                return False

        nsdfg = self.nested_sdfg.sdfg
        if len(nsdfg.states()) == 1:
            return False

        start_state = nsdfg.start_state
        if len(start_state.nodes()) > 0:
            return False

        if len(nsdfg.out_edges(start_state)) > 1:
            return False

        itervars = set(self.map_entry.map.params)
        candidates = set()
        for oedge in nsdfg.out_edges(start_state):
            if not oedge.data.is_unconditional():
                return False
            
            for symbol, value in oedge.data.assignments.items():
                sympy_value = symbolic.pystr_to_symbolic(value)
                if ({str(sym) for sym in sympy_value.expr_free_symbols} & itervars):
                    return False
                if "_for" in str(symbol):
                    return False

                candidates.add(symbol)

        # Candidates must be nsdfg-local
        for sym in candidates:
            if sym in self.nested_sdfg.symbol_mapping:
                return False

        # Check that symbol is constant
        for edge in nsdfg.edges():
            if edge.src == nsdfg.start_state:
                continue

            for sym in candidates:
                if sym in edge.data.assignments:
                    return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry
        map_exit = state.exit_node(self.map_entry)
        nsdfg_node = self.nested_sdfg
        nsdfg = nsdfg_node.sdfg
        outer_state = state

        ############################################
        # Split nsdfg's first state

        nsdfg_first_state = nsdfg.start_state
        nsdfg_second_state = nsdfg.out_edges(nsdfg_first_state)[0].dst
        to_delete = [s for s in nsdfg.states() if s != nsdfg_first_state]

        # Create new nsdfg without initial first state
        new_nsdfg: SDFG = dace.SDFG(nsdfg.label + "_inlined")
        for name, desc in nsdfg.arrays.items():
            new_nsdfg.add_datadesc(name, copy.deepcopy(desc))
        
        # Define symbols of new_nsdfg
        outer_symbols = set(nsdfg_node.symbol_mapping.keys())
        first_symbols = set()
        for edge in nsdfg.out_edges(nsdfg_first_state):
            first_symbols |= edge.data.assignments.keys()
        
        # for sym in nsdfg.symbols:
        #     if sym not in nsdfg.symbols:
        #         new_nsdfg.add_symbol(sym, nsdfg.symbols[sym])
        
        symbol_mapping = {sym: sym for sym in (first_symbols | outer_symbols)}

        # Add states
        new_states = {}
        for s in to_delete:
            new_state = copy.deepcopy(s)
            for node in new_state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent_sdfg = new_nsdfg
            new_nsdfg.add_node(new_state, is_start_state=(s==nsdfg_second_state))
            new_states[s] = new_state

        for edge in nsdfg.edges():
            if edge.src in to_delete and edge.dst in to_delete:
                new_nsdfg.add_edge(new_states[edge.src], new_states[edge.dst], copy.deepcopy(edge.data))

        new_nsdfg_state = nsdfg.add_state_before(nsdfg_second_state)
        for rem_state in to_delete:
            nsdfg.remove_node(rem_state)

        # Add map around new nsdfg node
        new_map_entry = copy.deepcopy(map_entry)
        new_nsdfg_state.add_node(new_map_entry)
        new_map_exit = copy.deepcopy(map_exit)
        new_nsdfg_state.add_node(new_map_exit)

        new_nsdfg_node = new_nsdfg_state.add_nested_sdfg(
            sdfg=new_nsdfg,
            parent=nsdfg,
            symbol_mapping=symbol_mapping,
            inputs=copy.deepcopy(nsdfg_node.in_connectors),
            outputs=copy.deepcopy(nsdfg_node.out_connectors),
        )

        for iedge in outer_state.in_edges(map_entry):
            if iedge.src.data not in nsdfg_node.in_connectors:
                nsdfg_node.add_in_connector(iedge.src.data)
                
            desc = copy.deepcopy(sdfg.arrays[iedge.src.data])
            if iedge.src.data in nsdfg.arrays:
                del nsdfg.arrays[iedge.src.data]
            
            if isinstance(desc, View):
                if isinstance(desc, StructureView):
                    desc = desc.as_structure()
                elif isinstance(desc, ArrayView):
                    desc = desc.as_array()
                else:
                    raise NotImplementedError
            
            desc.transient = False
            nsdfg.add_datadesc(iedge.src.data, desc)

            outer_state.add_edge(iedge.src, iedge.src_conn, nsdfg_node, iedge.src.data, Memlet.from_array(iedge.src.data, sdfg.arrays[iedge.src.data]))

            inner_access_node = new_nsdfg_state.add_access(iedge.src.data)
            new_nsdfg_state.add_edge(inner_access_node, None, new_map_entry, iedge.dst_conn, copy.deepcopy(iedge.data))

        for oedge in outer_state.out_edges(map_entry):
            new_nsdfg_state.add_edge(new_map_entry, oedge.src_conn, new_nsdfg_node, oedge.dst_conn, copy.deepcopy(oedge.data))

        for oedge in outer_state.out_edges(map_exit):
            if oedge.dst.data not in nsdfg_node.out_connectors:
                nsdfg_node.add_out_connector(oedge.dst.data, force=True)

            desc = copy.deepcopy(sdfg.arrays[oedge.dst.data])
            if oedge.dst.data in nsdfg.arrays:
                del nsdfg.arrays[oedge.dst.data]
            
            if isinstance(desc, View):
                if isinstance(desc, StructureView):
                    desc = desc.as_structure()
                elif isinstance(desc, ArrayView):
                    desc = desc.as_array()
                else:
                    raise NotImplementedError
            
            desc.transient = False
            nsdfg.add_datadesc(oedge.dst.data, desc)

            outer_state.add_edge(nsdfg_node, oedge.dst.data, oedge.dst, oedge.dst_conn, Memlet.from_array(oedge.dst.data, sdfg.arrays[oedge.dst.data]))

            inner_access_node = new_nsdfg_state.add_access(oedge.dst.data)
            new_nsdfg_state.add_edge(new_map_exit, oedge.src_conn, inner_access_node, None, copy.deepcopy(oedge.data))

        for iedge in outer_state.in_edges(map_exit):
            new_nsdfg_state.add_edge(new_nsdfg_node, iedge.src_conn, new_map_exit, iedge.dst_conn, copy.deepcopy(iedge.data))

        ############################################

        outer_state.remove_node(map_entry)
        outer_state.remove_node(map_exit)

        for sym in new_map_entry.map.params:
            if sym in nsdfg_node.symbol_mapping:
                del nsdfg_node.symbol_mapping[sym]
            if sym in nsdfg.symbols:
                nsdfg.remove_symbol(sym)

        for sym in new_map_entry.free_symbols:
            if sym not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[sym] = sym

            if sym in sdfg.symbols:
                if sym not in nsdfg.symbols:
                    nsdfg.add_symbol(sym, sdfg.symbols[sym])

        # Update local symbols of outer nsdfg
        local_symbols = set()
        for edge in nsdfg.edges():
            local_symbols |= edge.data.assignments.keys()

        for sym in set(nsdfg.symbols.keys()):
            if sym not in (local_symbols | nsdfg_node.symbol_mapping.keys()):
                nsdfg.remove_symbol(sym)
                if sym in new_nsdfg_node.symbol_mapping:
                    del new_nsdfg_node.symbol_mapping[sym]

        sdfg._cfg_list = sdfg.reset_cfg_list()

@make_properties
class InlineMapByConditions(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        itervars = self.map_entry.map.params
        if len(itervars) > 1:
            return False

        nsdfg = self.nested_sdfg.sdfg
        if len(nsdfg.states()) == 1:
            return False

        start_state = nsdfg.start_state
        if len(start_state.nodes()) > 0:
            return False

        for oedge in nsdfg.out_edges(start_state):
            if oedge.data.assignments:
                return False

            condition = oedge.data.condition_sympy()
            if condition.__class__ not in [
                sympy.core.relational.StrictLessThan,
                sympy.core.relational.GreaterThan,
                sympy.core.relational.Unequality,
                sympy.core.relational.Equality
            ]:
                return False

            if str(condition.lhs) != itervars[0]:
                return False

            if isinstance(condition, (sympy.core.relational.Unequality,
                sympy.core.relational.Equality)):
                b, e, _ = self.map_entry.map.range[0]
                if not (str(condition.rhs) == str(b) or str(condition.rhs) == str(e)):
                    return False

            print(oedge.data.condition.as_string)
            print()
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry
        map_exit = state.exit_node(self.map_entry)
        nsdfg_node = self.nested_sdfg
        outer_state = state

        ############################################
        # Fission maps by conditions

        nsdfg = nsdfg_node.sdfg
        start_state = nsdfg.start_state
        for oedge in nsdfg.out_edges(start_state):
            branch_nsdfg_node = copy.deepcopy(nsdfg_node)
            branch_nsdfg = branch_nsdfg_node.sdfg
            
            old_start_state = branch_nsdfg.start_state

            matching_edge = None
            for branch in branch_nsdfg.out_edges(old_start_state):
                if branch.data.condition.as_string == oedge.data.condition.as_string:
                    matching_edge = branch
                    break

            new_start_state = matching_edge.dst
            
            # Remove unreachable states
            branch_subgraph = set([e.dst for e in branch_nsdfg.edge_bfs(new_start_state)])
            branch_subgraph.add(new_start_state)
            states_to_remove = set(branch_nsdfg.states()) - branch_subgraph
            branch_nsdfg.remove_nodes_from(states_to_remove)

            branch_nsdfg.start_state = branch_nsdfg.node_id(new_start_state)
            outer_state.add_node(branch_nsdfg_node)
            
            # Add branch nsdfg to outer state
            branch_map_entry = copy.deepcopy(map_entry)
            outer_state.add_node(branch_map_entry)
            branch_map_exit = copy.deepcopy(map_exit)
            outer_state.add_node(branch_map_exit)

            for iedge in outer_state.in_edges(map_entry):
                outer_state.add_edge(iedge.src, iedge.src_conn, branch_map_entry, iedge.dst_conn, copy.deepcopy(iedge.data))

            for oedge in outer_state.out_edges(map_exit):
                outer_state.add_edge(branch_map_exit, oedge.src_conn, oedge.dst, oedge.dst_conn, copy.deepcopy(oedge.data))

            for oedge in outer_state.out_edges(map_entry):
                outer_state.add_edge(branch_map_entry, oedge.src_conn, branch_nsdfg_node, oedge.dst_conn, copy.deepcopy(oedge.data))

            for iedge in outer_state.in_edges(map_exit):
                outer_state.add_edge(branch_nsdfg_node, iedge.src_conn, branch_map_exit, iedge.dst_conn, copy.deepcopy(iedge.data))

            # Add condition to map definition
            condition = matching_edge.data.condition_sympy()
            b, e, s = branch_map_entry.map.range[0]
            if isinstance(condition, sympy.core.relational.StrictLessThan):
                e = min(e, condition.rhs - 1)
            elif isinstance(condition, sympy.core.relational.GreaterThan):
                b = max(b, condition.rhs)
            elif isinstance(condition, sympy.core.relational.Equality):
                b = condition.rhs
                e = condition.rhs
                s = 1
            elif isinstance(condition, sympy.core.relational.Unequality):
                if str(condition.rhs) == str(b):
                    b = condition.rhs + 1
                else:
                    e = condition.rhs - 1
            else:
                raise NotImplementedError
        
            branch_map_entry.map.range[0] = (b, e, s)

        ############################################
        # Clean up

        state.remove_node(map_entry)
        state.remove_node(nsdfg_node)
        state.remove_node(map_exit)

        sdfg._cfg_list = sdfg.reset_cfg_list()

@make_properties
class InlineMapSingleState(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        nsdfg = self.nested_sdfg.sdfg
        if len(nsdfg.states()) > 1:
            return False

        for inp in state.in_edges(self.map_entry):
            if not isinstance(inp.src, nodes.AccessNode):
                return False

        for outp in state.out_edges(state.exit_node(self.map_entry)):
            if not isinstance(outp.dst, nodes.AccessNode):
                return False

        nested_state = nsdfg.start_state
        subgraph = SubgraphView(nested_state, nested_state.nodes())

        for source_node in subgraph.source_nodes():
            if not isinstance(source_node, nodes.AccessNode):
                continue

            for oedge in subgraph.out_edges(source_node):
                if oedge.dst_conn == "views":
                    return False
                if oedge.data.subset != Memlet.from_array(oedge.data.data, nsdfg.arrays[oedge.data.data]).subset:
                    return False

        for sink_node in subgraph.sink_nodes():
            if not isinstance(sink_node, nodes.AccessNode):
                continue

            for iedge in subgraph.in_edges(sink_node):
                if iedge.src_conn == "views":
                    return False
                if iedge.data.subset != Memlet.from_array(iedge.data.data, nsdfg.arrays[iedge.data.data]).subset:
                    return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry
        map_exit = state.exit_node(self.map_entry)
        nsdfg_node = self.nested_sdfg
        nsdfg = self.nested_sdfg.sdfg
        nested_state: SDFGState = nsdfg.start_state
        outer_state: SDFGState = state

        ############################################
        # Make original arguments transients

        for array, desc in nsdfg.arrays.items():
            if array in nsdfg_node.in_connectors or array in nsdfg_node.out_connectors:
                desc.transient = True

        ############################################
        # Inline Map

        # Map subgraph
        subgraph = SubgraphView(nested_state, nested_state.nodes())

        # Collect map inputs/outputs
        input_memlets = set()
        for iedge in outer_state.in_edges(map_entry):
            input_memlets.add(iedge)

        input_inner_memlets = {}
        for oedge in outer_state.out_edges(map_entry):
            input_inner_memlets[oedge.dst_conn] = oedge
            nsdfg_node.remove_in_connector(oedge.dst_conn)

        output_memlets = set()
        for oedge in outer_state.out_edges(map_exit):
            output_memlets.add(oedge)

        output_inner_memlets = {}
        for oedge in outer_state.in_edges(map_exit):
            output_inner_memlets[oedge.src_conn] = oedge
            nsdfg_node.remove_out_connector(oedge.src_conn)

        # Add new inputs to nested SDFG
        for inp in input_memlets:
            nsdfg_node.add_in_connector(inp.src.data)
            outer_state.add_edge(inp.src, inp.src_conn, nsdfg_node, inp.src.data, Memlet.from_array(inp.src.data, sdfg.arrays[inp.src.data]))

            desc = copy.deepcopy(sdfg.arrays[inp.src.data])
            if inp.src.data in nsdfg.arrays:
                del nsdfg.arrays[inp.src.data]
            
            if isinstance(desc, View):
                if isinstance(desc, StructureView):
                    desc = desc.as_structure()
                elif isinstance(desc, ArrayView):
                    desc = desc.as_array()
                else:
                    raise NotImplementedError
            
            desc.transient = False
            nsdfg.add_datadesc(inp.src.data, desc)

        for outp in output_memlets:
            nsdfg_node.add_out_connector(outp.dst.data, force=True)
            outer_state.add_edge(nsdfg_node, outp.dst.data, outp.dst, outp.dst_conn, Memlet.from_array(outp.dst.data, sdfg.arrays[outp.dst.data]))

            desc = copy.deepcopy(sdfg.arrays[outp.dst.data])
            if outp.dst.data in nsdfg.arrays:
                del nsdfg.arrays[outp.dst.data]
            
            if isinstance(desc, View):
                if isinstance(desc, StructureView):
                    desc = desc.as_structure()
                elif isinstance(desc, ArrayView):
                    desc = desc.as_array()
                else:
                    raise NotImplementedError
            
            desc.transient = False
            nsdfg.add_datadesc(outp.dst.data, desc)

        # Add map to nested_sdfg
        nested_map_entry = copy.deepcopy(map_entry)
        nested_state.add_node(nested_map_entry)
        nested_map_exit = copy.deepcopy(map_exit)
        nested_state.add_node(nested_map_exit)

        # Reconnect map to arguments inside nested SDFG
        for inp in input_memlets:
            nested_inp = nested_state.add_access(inp.src.data)

            nested_state.add_edge(nested_inp, inp.src_conn, nested_map_entry, inp.dst_conn, copy.deepcopy(inp.data))

        for outp in output_memlets:
            nested_outp = nested_state.add_access(outp.dst.data)

            nested_state.add_edge(nested_map_exit, outp.src_conn, nested_outp, outp.dst_conn, copy.deepcopy(outp.data))

        # Connect map to subgraph
        source_nodes = list(subgraph.source_nodes())
        sink_nodes = list(subgraph.sink_nodes())
        for source_node in source_nodes:
            if not isinstance(source_node, nodes.AccessNode):
                continue

            if source_node.data in input_inner_memlets:
                edge = input_inner_memlets[source_node.data]

                for oedge in subgraph.out_edges(source_node):
                    nested_state.add_edge(nested_map_entry, edge.src_conn, oedge.dst, oedge.dst_conn, copy.deepcopy(edge.data))
                
                nested_state.remove_node(source_node)

        for sink_node in sink_nodes:
            if not isinstance(sink_node, nodes.AccessNode):
                continue

            if sink_node.data in output_inner_memlets:
                edge = output_inner_memlets[sink_node.data]

                for iedge in subgraph.in_edges(sink_node):
                    nested_state.add_edge(iedge.src, iedge.src_conn, nested_map_exit, edge.dst_conn, copy.deepcopy(edge.data))

                nested_state.remove_node(sink_node)

        ############################################
        # Clean up
        for sym in map_entry.map.params:
            nsdfg.remove_symbol(sym)

        for sym in map_entry.free_symbols:
            if sym not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[sym] = sym

            if sym in sdfg.symbols:
                if sym not in nsdfg.symbols:
                    nsdfg.add_symbol(sym, sdfg.symbols[sym])

        # Remove outer parts
        outer_state.remove_node(map_entry)
        outer_state.remove_node(map_exit)

        sdfg._cfg_list = sdfg.reset_cfg_list()
