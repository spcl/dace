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

    def can_be_applied(self, state: SDFGState, expr_index, sdfg, permissive=False):
        nested_sdfg = self.nested_sdfg
        if nested_sdfg.no_inline:
            return False
        if nested_sdfg.schedule == dtypes.ScheduleType.FPGA_Device:
            return False

        # Ensure the state only contains a nested SDFG and input/output access
        # nodes
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                if node is not nested_sdfg:
                    return False
            elif isinstance(node, nodes.AccessNode):
                # Must be connected to nested SDFG
                # if nested_sdfg in state.predecessors(nested_sdfg):
                #     if state.in_degree(node) > 0:
                #         return False
                found = False
                for e in state.out_edges(node):
                    if e.dst is not nested_sdfg:
                        return False
                    if state.in_degree(node) > 0:
                        return False
                    # Only accept full ranges for now. TODO(later): Improve
                    if e.data.subset != subsets.Range.from_array(sdfg.arrays[node.data]):
                        return False
                    if e.dst_conn in nested_sdfg.sdfg.arrays:
                        # Do not accept views. TODO(later): Improve
                        outer_desc = sdfg.arrays[node.data]
                        inner_desc = nested_sdfg.sdfg.arrays[e.dst_conn]
                        if (outer_desc.shape != inner_desc.shape or outer_desc.strides != inner_desc.strides):
                            return False
                    found = True

                for e in state.in_edges(node):
                    if e.src is not nested_sdfg:
                        return False
                    if state.out_degree(node) > 0:
                        return False
                    # Only accept full ranges for now. TODO(later): Improve
                    if e.data.subset != subsets.Range.from_array(sdfg.arrays[node.data]):
                        return False
                    if e.src_conn in nested_sdfg.sdfg.arrays:
                        # Do not accept views. TODO(later): Improve
                        outer_desc = sdfg.arrays[node.data]
                        inner_desc = nested_sdfg.sdfg.arrays[e.src_conn]
                        if (outer_desc.shape != inner_desc.shape or outer_desc.strides != inner_desc.strides):
                            return False
                    found = True

                # elif nested_sdfg in state.successors(nested_sdfg):
                #     if state.out_degree(node) > 0:
                #         return False
                if not found:
                    return False
            else:
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

        # Find original source/destination edges (there is only one edge per
        # connector, according to match)
        inputs: Dict[str, MultiConnectorEdge] = {}
        outputs: Dict[str, MultiConnectorEdge] = {}
        input_set: Dict[str, str] = {}
        output_set: Dict[str, str] = {}
        for e in outer_state.in_edges(nsdfg_node):
            inputs[e.dst_conn] = e
            input_set[e.data.data] = e.dst_conn
        for e in outer_state.out_edges(nsdfg_node):
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

        # Access nodes that need to be reshaped
        # reshapes: Set(str) = set()
        # for aname, array in nsdfg.arrays.items():
        #     if array.transient:
        #         continue
        #     edge = None
        #     if aname in inputs:
        #         edge = inputs[aname]
        #         if len(array.shape) > len(edge.data.subset):
        #             reshapes.add(aname)
        #             continue
        #     if aname in outputs:
        #         edge = outputs[aname]
        #         if len(array.shape) > len(edge.data.subset):
        #             reshapes.add(aname)
        #             continue
        #     if edge is not None and not InlineMultistateSDFG._check_strides(
        #             array.strides, sdfg.arrays[edge.data.data].strides,
        #             edge.data, nsdfg_node):
        #         reshapes.add(aname)

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
        repldict.update({k: v.data.data for k, v in itertools.chain(inputs.items(), outputs.items())})

        symbolic.safe_replace(repldict, lambda m: replace_datadesc_names(nsdfg, m), value_as_string=True)

        # Add views whenever reshapes are necessary
        # for dname in reshapes:
        #     desc = nsdfg.arrays[dname]
        #     # To avoid potential confusion, rename protected __return keyword
        #     if dname.startswith('__return'):
        #         newname = f'{nsdfg.name}_ret{dname[8:]}'
        #     else:
        #         newname = dname
        #     newname, _ = sdfg.add_view(newname,
        #                                desc.shape,
        #                                desc.dtype,
        #                                storage=desc.storage,
        #                                strides=desc.strides,
        #                                offset=desc.offset,
        #                                debuginfo=desc.debuginfo,
        #                                allow_conflicts=desc.allow_conflicts,
        #                                total_size=desc.total_size,
        #                                alignment=desc.alignment,
        #                                may_alias=desc.may_alias,
        #                                find_new_name=True)
        #     repldict[dname] = newname

        # Add extra access nodes for out/in view nodes
        # inv_reshapes = {repldict[r]: r for r in reshapes}
        # for nstate in nsdfg.nodes():
        #     for node in nstate.nodes():
        #         if isinstance(node,
        #                       nodes.AccessNode) and node.data in inv_reshapes:
        #             if nstate.in_degree(node) > 0 and nstate.out_degree(
        #                     node) > 0:
        #                 # Such a node has to be in the output set
        #                 edge = outputs[inv_reshapes[node.data]]

        #                 # Redirect outgoing edges through access node
        #                 out_edges = list(nstate.out_edges(node))
        #                 anode = nstate.add_access(edge.data.data)
        #                 vnode = nstate.add_access(node.data)
        #                 nstate.add_nedge(node, anode, edge.data)
        #                 nstate.add_nedge(anode, vnode, edge.data)
        #                 for e in out_edges:
        #                     nstate.remove_edge(e)
        #                     nstate.add_edge(vnode, e.src_conn, e.dst,
        #                                     e.dst_conn, e.data)

        # Make unique names for states
        statenames = set(s.label for s in sdfg.nodes())
        for nstate in nsdfg.nodes():
            if nstate.label in statenames:
                newname = data.find_new_name(nstate.label, statenames)
                statenames.add(newname)
                nstate.set_label(newname)

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
        for e in sdfg.in_edges(outer_state):
            sdfg.add_edge(e.src, source, e.data)
        for e in sdfg.out_edges(outer_state):
            for sink in sinks:
                sdfg.add_edge(sink, e.dst, dc(e.data))
                # Redirect sink incoming edges with a `False` condition to e.dst (return statements)
                for e2 in sdfg.in_edges(sink):
                    if e2.data.condition_sympy() == False:
                        sdfg.add_edge(e2.src, e.dst, InterstateEdge())

        # Modify start state as necessary
        if outer_start_state is outer_state:
            sdfg.start_state = sdfg.node_id(source)

        # TODO: Modify memlets by offsetting
        # If both source and sink nodes are inputs/outputs, reconnect once
        # edges_to_ignore = self._modify_access_to_access(new_incoming_edges,
        #                                                 nsdfg, nstate, state,
        #                                                 orig_data)

        # source_to_outer = {n: e.src for n, e in new_incoming_edges.items()}
        # sink_to_outer = {n: e.dst for n, e in new_outgoing_edges.items()}
        # # If a source/sink node is one of the inputs/outputs, reconnect it,
        # # replacing memlets in outgoing/incoming paths
        # modified_edges = set()
        # modified_edges |= self._modify_memlet_path(new_incoming_edges, nstate,
        #                                            state, sink_to_outer, True,
        #                                            edges_to_ignore)
        # modified_edges |= self._modify_memlet_path(new_outgoing_edges, nstate,
        #                                            state, source_to_outer,
        #                                            False, edges_to_ignore)

        # # Reshape: add connections to viewed data
        # self._modify_reshape_data(reshapes, repldict, inputs, nstate, state,
        #                           True)
        # self._modify_reshape_data(reshapes, repldict, outputs, nstate, state,
        #                           False)

        # Modify all other internal edges pertaining to input/output nodes
        # for nstate in nsdfg.nodes():
        #     for node in nstate.nodes():
        #         if isinstance(node, nodes.AccessNode):
        #             if node.data in input_set or node.data in output_set:
        #                 if node.data in input_set:
        #                     outer_edge = inputs[input_set[node.data]]
        #                 else:
        #                     outer_edge = outputs[output_set[node.data]]

        #                 for edge in state.all_edges(node):
        #                     if (edge not in modified_edges
        #                             and edge.data.data == node.data):
        #                         for e in state.memlet_tree(edge):
        #                             if e.data.data == node.data:
        #                                 e._data = helpers.unsqueeze_memlet(
        #                                     e.data, outer_edge.data)

        # Replace nested SDFG parents with new SDFG
        for nstate in nsdfg.nodes():
            nstate.parent = sdfg
            for node in nstate.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent_sdfg = sdfg
                    node.sdfg.parent_nsdfg_node = node

        #######################################################
        # Remove nested SDFG and state
        sdfg.remove_node(outer_state)

        sdfg._sdfg_list = sdfg.reset_sdfg_list()

        return nsdfg.nodes()

    # def _modify_access_to_access(
    #     self,
    #     input_edges: Dict[nodes.Node, MultiConnectorEdge],
    #     nsdfg: SDFG,
    #     nstate: SDFGState,
    #     state: SDFGState,
    #     orig_data: Dict[Union[nodes.AccessNode, MultiConnectorEdge], str],
    # ) -> Set[MultiConnectorEdge]:
    #     """
    #     Deals with access->access edges where both sides are non-transient.
    #     """
    #     result = set()
    #     for node, top_edge in input_edges.items():
    #         for inner_edge in nstate.out_edges(node):
    #             if inner_edge.dst not in orig_data:
    #                 continue
    #             inner_data = orig_data[inner_edge.dst]
    #             if (isinstance(inner_edge.dst, nodes.AccessNode)
    #                     and not nsdfg.arrays[inner_data].transient):
    #                 matching_edge: MultiConnectorEdge = next(
    #                     state.out_edges_by_connector(top_edge.dst, inner_data))
    #                 # Create memlet by unsqueezing both w.r.t. src and dst
    #                 # subsets
    #                 in_memlet = helpers.unsqueeze_memlet(
    #                     inner_edge.data, top_edge.data)
    #                 out_memlet = helpers.unsqueeze_memlet(
    #                     inner_edge.data, matching_edge.data)
    #                 new_memlet = in_memlet
    #                 new_memlet.other_subset = out_memlet.subset

    #                 # Connect with new edge
    #                 state.add_edge(top_edge.src, top_edge.src_conn,
    #                                matching_edge.dst, matching_edge.dst_conn,
    #                                new_memlet)
    #                 result.add(inner_edge)

    #     return result

    # def _modify_memlet_path(
    #     self,
    #     new_edges: Dict[nodes.Node, MultiConnectorEdge],
    #     nstate: SDFGState,
    #     state: SDFGState,
    #     inner_to_outer: Dict[nodes.Node, MultiConnectorEdge],
    #     inputs: bool,
    #     edges_to_ignore: Set[MultiConnectorEdge],
    # ) -> Set[MultiConnectorEdge]:
    #     """ Modifies memlet paths in an inlined SDFG. Returns set of modified
    #         edges.
    #     """
    #     result = set()
    #     for node, top_edge in new_edges.items():
    #         inner_edges = (nstate.out_edges(node)
    #                        if inputs else nstate.in_edges(node))
    #         for inner_edge in inner_edges:
    #             if inner_edge in edges_to_ignore:
    #                 continue
    #             new_memlet = helpers.unsqueeze_memlet(inner_edge.data,
    #                                                   top_edge.data)
    #             if inputs:
    #                 if inner_edge.dst in inner_to_outer:
    #                     dst = inner_to_outer[inner_edge.dst]
    #                 else:
    #                     dst = inner_edge.dst

    #                 new_edge = state.add_edge(top_edge.src, top_edge.src_conn,
    #                                           dst, inner_edge.dst_conn,
    #                                           new_memlet)
    #                 mtree = state.memlet_tree(new_edge)
    #             else:
    #                 if inner_edge.src in inner_to_outer:
    #                     # don't add edges twice
    #                     continue

    #                 new_edge = state.add_edge(inner_edge.src,
    #                                           inner_edge.src_conn, top_edge.dst,
    #                                           top_edge.dst_conn, new_memlet)
    #                 mtree = state.memlet_tree(new_edge)

    #             # Modify all memlets going forward/backward
    #             def traverse(mtree_node):
    #                 result.add(mtree_node.edge)
    #                 mtree_node.edge._data = helpers.unsqueeze_memlet(
    #                     mtree_node.edge.data, top_edge.data)
    #                 for child in mtree_node.children:
    #                     traverse(child)

    #             for child in mtree.children:
    #                 traverse(child)

    #     return result

    # def _modify_reshape_data(self, reshapes: Set[str], repldict: Dict[str, str],
    #                          new_edges: Dict[str, MultiConnectorEdge],
    #                          nstate: SDFGState, state: SDFGState, inputs: bool):
    #     anodes = nstate.source_nodes() if inputs else nstate.sink_nodes()
    #     reshp = {repldict[r]: r for r in reshapes}
    #     for node in anodes:
    #         if not isinstance(node, nodes.AccessNode):
    #             continue
    #         if node.data not in reshp:
    #             continue
    #         edge = new_edges[reshp[node.data]]
    #         if inputs:
    #             state.add_edge(edge.src, edge.src_conn, node, None, edge.data)
    #         else:
    #             state.add_edge(node, None, edge.dst, edge.dst_conn, edge.data)
