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
from dace.sdfg.replace import replace_datadesc_names, replace_properties_dict
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

            if edge.data.subset != subsets.Range.from_array(sdfg.arrays[edge.data.data]):
                return False

            outer_desc = sdfg.arrays[edge.data.data]
            if isinstance(outer_desc, data.View):
                return False

            # We can not compare shapes directly, we have to consider the symbol map
            #  for that. Clone the descriptor because the operation is inplace.
            inner_desc = nested_sdfg.sdfg.arrays[edge.dst_conn].clone()
            symbolic.safe_replace(nested_sdfg.symbol_mapping, lambda m: replace_properties_dict(inner_desc, m))
            if (outer_desc.shape != inner_desc.shape or outer_desc.strides != inner_desc.strides):
                return False

        for edge in state.out_edges(nested_sdfg):
            if edge.data.data is None:
                return False

            if not isinstance(edge.dst, nodes.AccessNode):
                return False

            if edge.data.subset != subsets.Range.from_array(sdfg.arrays[edge.data.data]):
                return False

            outer_desc = sdfg.arrays[edge.data.data]
            if isinstance(outer_desc, data.View):
                return False

            inner_desc = nested_sdfg.sdfg.arrays[edge.src_conn].clone()
            symbolic.safe_replace(nested_sdfg.symbol_mapping, lambda m: replace_properties_dict(inner_desc, m))
            if (outer_desc.shape != inner_desc.shape or outer_desc.strides != inner_desc.strides):
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

        # Find original source/destination edges (there is only one edge per
        # connector, according to match)
        inputs: Dict[str, MultiConnectorEdge] = {}
        outputs: Dict[str, MultiConnectorEdge] = {}
        input_set: Dict[str, str] = {}
        output_set: Dict[str, str] = {}
        for e in nsdfg_state.in_edges(nsdfg_node):
            inputs[e.dst_conn] = e
            input_set[e.data.data] = e.dst_conn
        for e in nsdfg_state.out_edges(nsdfg_node):
            outputs[e.src_conn] = e
            output_set[e.data.data] = e.src_conn

        # Replace symbols using invocation symbol mapping
        # Two-step replacement (N -> __dacesym_N --> map[N]) to avoid clashes
        symbolic.safe_replace(nsdfg_node.symbol_mapping, nsdfg.replace_dict)

        #######################################################
        # Collect and modify interstate edges as necessary

        outer_assignments = set()
        for e in sdfg.edges():
            outer_assignments |= e.data.assignments.keys()

        inner_assignments = set()
        for e in nsdfg.edges():
            inner_assignments |= e.data.assignments.keys()

        allnames = set(outer_symbols.keys()) | set(sdfg.arrays.keys())
        assignments_to_replace = inner_assignments & (outer_assignments | allnames)
        sym_replacements: Dict[str, str] = {}
        for assign in assignments_to_replace:
            newname = data.find_new_name(assign, allnames)
            allnames.add(newname)
            outer_symbols[newname] = nsdfg.symbols.get(assign, None)
            sym_replacements[assign] = newname
        nsdfg.replace_dict(sym_replacements)

        #######################################################
        # Collect and modify access nodes as necessary

        # Mapping from nested transient name to top-level name
        transients: Dict[str, str] = {}

        # All transients become transients of the parent (if data already
        # exists, find new name)
        for nstate in nsdfg.nodes():
            for node in nstate.nodes():
                if isinstance(node, nodes.AccessNode):
                    datadesc = nsdfg.arrays[node.data]
                    if node.data not in transients and datadesc.transient:
                        new_name = node.data
                        if (new_name in sdfg.arrays or new_name in outer_symbols or new_name in sdfg.constants):
                            new_name = f'{nsdfg.label}_{node.data}'

                        name = sdfg.add_datadesc(new_name, datadesc, find_new_name=True)
                        transients[node.data] = name

            # All transients of edges between code nodes are also added to parent
            for edge in nstate.edges():
                if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
                    if edge.data.data is not None:
                        datadesc = nsdfg.arrays[edge.data.data]
                        if edge.data.data not in transients and datadesc.transient:
                            new_name = edge.data.data
                            if (new_name in sdfg.arrays or new_name in outer_symbols or new_name in sdfg.constants):
                                new_name = f'{nsdfg.label}_{edge.data.data}'

                            name = sdfg.add_datadesc(new_name, datadesc, find_new_name=True)
                            transients[edge.data.data] = name

        # All constants (and associated transients) become constants of the parent
        for cstname, (csttype, cstval) in nsdfg.constants_prop.items():
            sdfg.constants_prop[cstname] = (csttype, cstval)

        #######################################################
        # Replace data on inlined SDFG nodes/edges

        # Replace data names with their top-level counterparts
        repldict = {}
        repldict.update(transients)
        repldict.update({k: v.data.data for k, v in itertools.chain(inputs.items(), outputs.items())})

        symbolic.safe_replace(repldict, lambda m: replace_datadesc_names(nsdfg, m), value_as_string=True)

        # Make unique names for states
        statenames = set(s.label for s in sdfg.nodes())
        for nstate in nsdfg.nodes():
            if nstate.label in statenames:
                newname = data.find_new_name(nstate.label, statenames)
                statenames.add(newname)
                nstate.label = newname

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

        # TODO: Modify memlets by offsetting

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
