# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline multi-state SDFGs. """

from copy import deepcopy as dc
import itertools
from typing import Dict, List

from dace import Memlet, symbolic, dtypes, subsets
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg import InterstateEdge, SDFG, SDFGState
from dace.sdfg import utils as sdutil, infer_types
from dace.sdfg.replace import replace_datadesc_names, replace_properties_dict
from dace.transformation import transformation, helpers
from dace.properties import make_properties
from dace import data
from dace.sdfg.state import LoopRegion, ReturnBlock, StateSubgraphView


@make_properties
@transformation.explicit_cf_compatible
class InlineMultistateSDFG(transformation.SingleStateTransformation):
    """
    Inlines a multi-state nested SDFG into a top-level SDFG. This only happens
    if the state has the nested SDFG node isolated (i.e., only containing it
    and input/output access nodes), and thus the state machines can be combined.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    @staticmethod
    def _check_strides(inner_strides: List[symbolic.SymbolicType], outer_strides: List[symbolic.SymbolicType],
                       memlet: Memlet, nested_sdfg: nodes.NestedSDFG) -> bool:
        """
        Returns True if the strides of the inner array can be matched
        to the strides of the outer array upon inlining. Takes into
        consideration memlet (un)squeeze and nested SDFG symbol mapping.

        :param inner_strides: The strides of the array inside the nested SDFG.
        :param outer_strides: The strides of the array in the external SDFG.
        :param nested_sdfg: Nested SDFG node with symbol mapping.
        :return: True if all strides match, False otherwise.
        """
        # Replace all inner symbols based on symbol mapping
        istrides = list(inner_strides)

        def replfunc(mapping):
            for i, s in enumerate(istrides):
                if symbolic.issymbolic(s):
                    istrides[i] = s.subs(mapping)

        symbolic.safe_replace(nested_sdfg.symbol_mapping, replfunc)

        if istrides == list(outer_strides):
            return True

        # Take unsqueezing into account
        dims_to_ignore = [i for i, s in enumerate(memlet.subset.size()) if s == 1]
        ostrides = [os for i, os in enumerate(outer_strides) if i not in dims_to_ignore]

        if len(ostrides) == 0:
            ostrides = [1]

        if len(ostrides) != len(istrides):
            return False

        return all(istr == ostr for istr, ostr in zip(istrides, ostrides))

    def can_be_applied(self, state: SDFGState, expr_index, sdfg: SDFG, permissive=False):
        nested_sdfg = self.nested_sdfg
        if nested_sdfg.no_inline:
            return False
        if nested_sdfg.schedule == dtypes.ScheduleType.FPGA_Device:
            return False

        # Not nested in scope
        if state.entry_node(nested_sdfg) is not None:
            return False

        # Must be
        # - connected to access nodes only
        # - read full subsets
        # - not use views inside
        for edge in state.in_edges(nested_sdfg):
            if edge.data.data is None:
                return False

            if not isinstance(edge.src, nodes.AccessNode):
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

        if not helpers.isolate_nested_sdfg(state, nsdfg_node=nested_sdfg, test_if_applicable=True):
            return False

        return True

    def apply(self, outer_state: SDFGState, sdfg: SDFG):
        nsdfg_node = self.nested_sdfg
        nsdfg: SDFG = nsdfg_node.sdfg

        # If the nested SDFG contains returns, ensure they are inlined first.
        has_return = False
        for blk in nsdfg.all_control_flow_blocks():
            if isinstance(blk, ReturnBlock):
                has_return = True
        if has_return:
            sdutil.inline_control_flow_regions(nsdfg, lower_returns=True)

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

        # Callbacks and other types
        sdfg._callback_mapping.update(nsdfg.callback_mapping)

        # Environments
        for nstate in nsdfg.states():
            for node in nstate.nodes():
                if isinstance(node, nodes.CodeNode):
                    node.environments |= nsdfg_node.environments

        # Symbols
        outer_symbols = {str(k): v for k, v in sdfg.symbols.items()}
        for ise in sdfg.all_interstate_edges():
            outer_symbols.update(ise.data.new_symbols(sdfg, outer_symbols))

        # Isolate the nested SDFG in a separate state.
        predecessor_state, nsdfg_state, successor_state = helpers.isolate_nested_sdfg(state=outer_state,
                                                                                      nsdfg_node=nsdfg_node)

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
        for e in sdfg.all_interstate_edges():
            outer_assignments |= e.data.assignments.keys()
        for b in sdfg.all_control_flow_blocks():
            if isinstance(b, LoopRegion):
                if b.loop_variable is not None:
                    outer_assignments.add(b.loop_variable)

        inner_assignments = set()
        for e in nsdfg.all_interstate_edges():
            inner_assignments |= e.data.assignments.keys()
        for b in nsdfg.all_control_flow_blocks():
            if isinstance(b, LoopRegion):
                if b.loop_variable is not None:
                    inner_assignments.add(b.loop_variable)

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
        for nstate in nsdfg.states():
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
            if cstname in sdfg.constants:
                if cstname in transients:
                    newname = transients[cstname]
                else:
                    newname = sdfg.find_new_constant(cstname)
                    transients[cstname] = newname
                sdfg.constants_prop[newname] = (csttype, cstval)
            else:
                sdfg.constants_prop[cstname] = (csttype, cstval)

        #######################################################
        # Replace data on inlined SDFG nodes/edges

        # Replace data names with their top-level counterparts
        repldict = {}
        repldict.update(transients)
        repldict.update({k: v.data.data for k, v in itertools.chain(inputs.items(), outputs.items())})

        symbolic.safe_replace(repldict, lambda m: replace_datadesc_names(nsdfg, m), value_as_string=True)

        # Make unique names for states
        statenames = set(s.label for s in sdfg.states())
        for nstate in nsdfg.states():
            if nstate.label in statenames:
                newname = data.find_new_name(nstate.label, statenames)
                nstate.label = newname
            statenames.add(nstate.label)

        #######################################################
        # Add nested SDFG states into top-level SDFG

        outer_start_state = outer_state.parent_graph.start_block

        outer_state.parent_graph.add_nodes_from(nsdfg.nodes())
        for ise in nsdfg.edges():
            outer_state.parent_graph.add_edge(ise.src, ise.dst, ise.data)

        #######################################################
        # Reconnect inlined SDFG

        source = nsdfg.start_state
        sinks = nsdfg.sink_nodes()

        # Reconnect state machine
        for e in outer_state.parent_graph.in_edges(nsdfg_state):
            outer_state.parent_graph.add_edge(e.src, source, e.data)
        for e in outer_state.parent_graph.out_edges(nsdfg_state):
            for sink in sinks:
                outer_state.parent_graph.add_edge(sink, e.dst, dc(e.data))
                # Redirect sink incoming edges with a `False` condition to e.dst (return statements)
                for e2 in outer_state.parent_graph.in_edges(sink):
                    if e2.data.condition_sympy() == False:
                        outer_state.parent_graph.add_edge(e2.src, e.dst, InterstateEdge())

        # Modify start state as necessary
        if outer_start_state is nsdfg_state:
            outer_state.parent_graph.start_block = outer_state.parent_graph.node_id(source)

        # TODO: Modify memlets by offsetting

        # Replace nested SDFG parents with new SDFG
        for nstate in nsdfg.states():
            nstate.sdfg = sdfg
            for node in nstate.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent_sdfg = sdfg
                    node.sdfg.parent_nsdfg_node = node

        #######################################################
        # Remove nested SDFG and state
        outer_state.parent_graph.remove_node(nsdfg_state)

        sdfg.reset_cfg_list()

        return nsdfg.nodes()
