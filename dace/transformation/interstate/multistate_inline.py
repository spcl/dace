# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline multi-state SDFGs. """

import ast
from collections import defaultdict
from copy import deepcopy as dc
from dace.frontend.python.ndloop import ndrange
import itertools
import networkx as nx
from typing import Callable, Dict, Iterable, List, Set, Optional, Tuple, Union
import warnings

from dace import memlet, registry, sdfg as sd, Memlet, symbolic, dtypes, subsets
from dace.frontend.python import astutils
from dace.sdfg import nodes, propagation
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
from dace.sdfg import InterstateEdge, SDFG, SDFGState
from dace.sdfg import utils as sdutil, infer_types, propagation
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation import transformation, helpers
from dace.properties import make_properties, Property
from dace import data
from dace.sdfg.state import StateSubgraphView


@make_properties
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

    def can_be_applied(self, state: SDFGState, expr_index, sdfg, permissive=False):
        nested_sdfg = self.nested_sdfg
        if nested_sdfg.no_inline:
            return False
        if nested_sdfg.schedule == dtypes.ScheduleType.FPGA_Device:
            return False

        # Not nested in scope
        if state.entry_node(nested_sdfg) is not None:
            return False

        # Must be connected to access nodes only
        needs_view = set()
        for edge in state.in_edges(nested_sdfg):
            if edge.data.data is None:
                return False
            
            # Is regular access node to an array
            if not isinstance(edge.src, nodes.AccessNode):
                return False
            outer_desc = sdfg.arrays[edge.data.data]
            if isinstance(outer_desc, data.View):
                return False

            # Is not full subset?
            if edge.data.subset != subsets.Range.from_array(sdfg.arrays[edge.data.data]):
                needs_view.add(edge.dst_conn)
                continue

            inner_desc = nested_sdfg.sdfg.arrays[edge.dst_conn]
            if outer_desc.strides != inner_desc.strides:
                needs_view.add(edge.dst_conn)

        for edge in state.out_edges(nested_sdfg):
            if edge.data.data is None:
                return False

            # Is regular access node to an array
            if not isinstance(edge.dst, nodes.AccessNode):
                return False
            outer_desc = sdfg.arrays[edge.data.data]
            if isinstance(outer_desc, data.View):
                return False

            # Is not full subset?
            if edge.data.subset != subsets.Range.from_array(sdfg.arrays[edge.data.data]):
                needs_view.add(edge.src_conn)
                continue

            inner_desc = nested_sdfg.sdfg.arrays[edge.src_conn]
            if outer_desc.strides != inner_desc.strides:
                needs_view.add(edge.src_conn)

        # View replacements may not be used in interstate edges
        for view in needs_view:
            for edge in nested_sdfg.sdfg.edges():
                if view in edge.data.free_symbols:
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

        # Symbols
        outer_symbols = {str(k): v for k, v in sdfg.symbols.items()}
        for ise in sdfg.edges():
            outer_symbols.update(ise.data.new_symbols(sdfg, outer_symbols))

        # Isolate nsdfg in a separate state
        # 1. Push nsdfg node plus dependencies down into new state
        nsdfg_state = helpers.state_fission_after(sdfg, outer_state, nsdfg_node)
        # 2. Push successors of nsdfg node into a later state
        direct_subgraph = set()
        direct_subgraph.add(nsdfg_node)
        direct_subgraph.update(nsdfg_state.predecessors(nsdfg_node))
        direct_subgraph.update(nsdfg_state.successors(nsdfg_node))
        direct_subgraph = StateSubgraphView(nsdfg_state, direct_subgraph)
        nsdfg_state = helpers.state_fission(sdfg, direct_subgraph)

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
        # Add views for each input and output

        views = {}
        for outer_array, inner_array in input_set.items():
            outer_desc = sdfg.arrays[outer_array]
            inner_desc = nsdfg.arrays[inner_array]
            outer_edge = inputs[inner_array]

            # Needs view?
            if outer_edge.data.subset == subsets.Range.from_array(sdfg.arrays[outer_array]) and outer_desc.strides == inner_desc.strides:
                continue

            # Provide full array as input
            nsdfg_state.remove_edge_and_connectors(outer_edge)
            nsdfg_node.add_in_connector(outer_array)
            nsdfg_state.add_edge(outer_edge.src, outer_edge.src_conn, outer_edge.dst, outer_array, dc(outer_edge.data))

            # Add full array to nsdfg if necessary
            if outer_array in nsdfg.arrays:
                del nsdfg[outer_array]
            nsdfg.add_datadesc(outer_array, dc(outer_desc))

            # Add view node
            del nsdfg.arrays[inner_array]
            newname, _ = nsdfg.add_view(inner_array,
                                       inner_desc.shape,
                                       inner_desc.dtype,
                                       storage=inner_desc.storage,
                                       strides=inner_desc.strides,
                                       offset=inner_desc.offset,
                                       debuginfo=inner_desc.debuginfo,
                                       allow_conflicts=inner_desc.allow_conflicts,
                                       total_size=inner_desc.total_size,
                                       alignment=inner_desc.alignment,
                                       may_alias=inner_desc.may_alias,
                                       find_new_name=True)

            views[inner_array] = outer_array

        for outer_array, inner_array in output_set.items():
            outer_desc = sdfg.arrays[outer_array]
            inner_desc = nsdfg.arrays[inner_array]
            outer_edge = outputs[inner_array]

            # Needs view?
            if outer_edge.data.subset == subsets.Range.from_array(sdfg.arrays[outer_array]) and outer_desc.strides == inner_desc.strides:
                continue

            # Provide full array as output
            outer_edge = outputs[inner_array]
            nsdfg_state.remove_edge_and_connectors(outer_edge)
            nsdfg_node.add_out_connector(outer_array)
            nsdfg_state.add_edge(outer_edge.src, outer_array, outer_edge.dst, outer_edge.dst_conn, dc(outer_edge.data))

            # Add full array to nsdfg if necessary
            if outer_array in nsdfg.arrays:
                del nsdfg.arrays[outer_array]
            nsdfg.add_datadesc(outer_array, dc(outer_desc))

            # Add view node
            del nsdfg.arrays[inner_array]
            newname, _ = nsdfg.add_view(inner_array,
                                       inner_desc.shape,
                                       inner_desc.dtype,
                                       storage=inner_desc.storage,
                                       strides=inner_desc.strides,
                                       offset=inner_desc.offset,
                                       debuginfo=inner_desc.debuginfo,
                                       allow_conflicts=inner_desc.allow_conflicts,
                                       total_size=inner_desc.total_size,
                                       alignment=inner_desc.alignment,
                                       may_alias=inner_desc.may_alias,
                                       find_new_name=True)

            views[inner_array] = outer_array

        for nstate in nsdfg.nodes():
            for node in list(nstate.nodes()):
                if isinstance(node, nodes.AccessNode):
                    if node.data in views:
                        outer_array = views[node.data]

                        in_edges = nstate.in_edges(node)
                        out_edges = nstate.out_edges(node)
                        if out_edges and not in_edges:
                            array_node = nstate.add_access(outer_array)
                            outer_memlet = dc(inputs[node.data].data) if node.data in inputs else dc(outputs[node.data].data)
                            nstate.add_edge(array_node, None, node, "views", outer_memlet)
                        elif in_edges and not out_edges:
                            array_node = nstate.add_access(outer_array)
                            outer_memlet = dc(inputs[node.data].data) if node.data in inputs else dc(outputs[node.data].data)
                            nstate.add_edge(node, "views", array_node, None, outer_memlet)
                        else:
                            array_node = nstate.add_access(outer_array)
                            outer_memlet = dc(inputs[node.data].data) if node.data in inputs else dc(outputs[node.data].data)
                            nstate.add_edge(node, "views", array_node, None, outer_memlet)

                            view_node_out = nstate.add_access(node.data)
                            nstate.add_edge(array_node, None, view_node_out, "views", dc(outer_memlet))

                            for edge in out_edges:
                                nstate.add_edge(view_node_out, edge.src_conn, edge.dst, edge.dst_conn, dc(edge.data))
                                nstate.remove_edge(edge)

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
        repldict.update({k: v.data.data for k, v in itertools.chain(inputs.items(), outputs.items()) if k not in views})
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

        outer_start_state = sdfg.start_state

        sdfg.add_nodes_from(nsdfg.nodes())
        for ise in nsdfg.edges():
            sdfg.add_edge(ise.src, ise.dst, ise.data)

        #######################################################
        # Reconnect inlined SDFG

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

        sdfg._sdfg_list = sdfg.reset_sdfg_list()

        return nsdfg.nodes()
