# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG nesting transformation. """

import ast
from collections import defaultdict
from copy import deepcopy as dc
from dace.frontend.python.ndloop import ndrange
import itertools
import networkx as nx
from typing import Callable, Dict, Iterable, List, Set, Optional, Tuple, Union
import warnings
from functools import reduce
import operator
import copy

from dace import memlet, registry, sdfg as sd, Memlet, symbolic, dtypes, subsets
from dace.frontend.python import astutils
from dace.sdfg import nodes, propagation
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil, infer_types, propagation
from dace.transformation import transformation, helpers
from dace.properties import make_properties, Property
from dace import data


@registry.autoregister_params(singlestate=True, strict=True)
@make_properties
class InlineSDFG(transformation.Transformation):
    """ Inlines a single-state nested SDFG into a top-level SDFG.

        In particular, the steps taken are:

        1. All transient arrays become transients of the parent
        2. If a source/sink node is one of the inputs/outputs:
          a. Remove it
          b. Reconnect through external edges (map/accessnode)
          c. Replace and reoffset memlets with external data descriptor
        3. If other nodes carry the names of inputs/outputs:
          a. Replace data with external data descriptor
          b. Replace and reoffset memlets with external data descriptor
        4. If source/sink node is not connected to a source/destination, and
           the nested SDFG is in a scope, connect to scope with empty memlets
        5. Remove all unused external inputs/output memlet paths
        6. Remove isolated nodes resulting from previous step

    """

    _nested_sdfg = nodes.NestedSDFG('_', sd.SDFG('_'), {}, {})

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(InlineSDFG._nested_sdfg)]

    @staticmethod
    def _check_strides(inner_strides: List[symbolic.SymbolicType],
                       outer_strides: List[symbolic.SymbolicType],
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
        repldict = {
            symbolic.pystr_to_symbolic(k):
            symbolic.pystr_to_symbolic('__dacesym_' + str(v))
            for k, v in nested_sdfg.symbol_mapping.items()
        }
        # need two dicts to avoid clashes
        repldict_inv = {
            symbolic.pystr_to_symbolic('__dacesym_' + str(v)):
            symbolic.pystr_to_symbolic(v)
            for v in nested_sdfg.symbol_mapping.values()
        }

        istrides = [
            istr.subs(repldict).subs(repldict_inv)
            if symbolic.issymbolic(istr) else istr for istr in inner_strides
        ]

        if istrides == list(outer_strides):
            return True

        # Take unsqueezing into account
        dims_to_ignore = [
            i for i, s in enumerate(memlet.subset.size()) if s == 1
        ]
        ostrides = [
            os for i, os in enumerate(outer_strides) if i not in dims_to_ignore
        ]

        if len(ostrides) == 0:
            ostrides = [1]

        if len(ostrides) != len(istrides):
            return False

        return all(istr == ostr for istr, ostr in zip(istrides, ostrides))

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate,
                       expr_index,
                       sdfg,
                       strict=False):
        nested_sdfg = graph.nodes()[candidate[InlineSDFG._nested_sdfg]]
        if nested_sdfg.no_inline:
            return False
        if len(nested_sdfg.sdfg.nodes()) != 1:
            return False

        # Ensure every connector has one incoming/outgoing edge and that it
        # is not empty
        in_connectors = set()
        out_connectors = set()
        for edge in graph.in_edges(nested_sdfg):
            if edge.dst_conn in in_connectors:
                return False
            if (edge.data.is_empty()
                    and not isinstance(edge.src, nodes.EntryNode)):
                return False
            in_connectors.add(edge.dst_conn)
        for edge in graph.out_edges(nested_sdfg):
            if edge.src_conn in out_connectors:
                return False
            if (edge.data.is_empty()
                    and not isinstance(edge.dst, nodes.ExitNode)):
                return False
            out_connectors.add(edge.src_conn)

        # Ensure output connectors have no additional outputs (if in a scope),
        # and ensure no two connectors are directly connected to each other
        if graph.entry_node(nested_sdfg) is not None:
            all_connectors = in_connectors | out_connectors
            nstate = nested_sdfg.sdfg.node(0)
            for node in nstate.nodes():
                if isinstance(node, nodes.AccessNode):
                    if (node.data in out_connectors
                            and nstate.out_degree(node) > 0
                            and (node.data not in in_connectors
                                 or nstate.in_degree(node) > 0)):
                        return False
                    if (node.data in in_connectors
                            and any(e.dst.data in all_connectors
                                    for e in nstate.out_edges(node)
                                    if isinstance(e.dst, nodes.AccessNode))):
                        return False

        # Ensure that every connector has at least one corresponding access
        # node in the (nested) SDFG. Otherwise, inlining is not possible.
        # NOTE: FPGA-compatible SDFGs can have input connectors for data that
        # are only written.
        inp_data = {conn: set() for conn in in_connectors}
        for e in graph.in_edges(nested_sdfg):
            src = graph.memlet_path(e)[0].src
            if isinstance(src, nodes.AccessNode):
                inp_data[e.dst_conn].add(src.data)
        out_data = dict()
        for e in graph.out_edges(nested_sdfg):
            dst = graph.memlet_path(e)[-1].dst
            if isinstance(dst, nodes.AccessNode):
                out_data[dst.data] = e.src_conn
        rem_inpconns = dc(in_connectors)
        rem_outconns = dc(out_connectors)
        nstate = nested_sdfg.sdfg.node(0)
        for node in nstate.nodes():
            if isinstance(node, nodes.AccessNode):
                if node.data in rem_inpconns:
                    rem_inpconns.remove(node.data)
                if node.data in rem_outconns:
                    rem_outconns.remove(node.data)
        if len(rem_outconns) > 0:
            return False
        if len(rem_inpconns) > 0:
            for inpconn in list(rem_inpconns):
                for access in inp_data[inpconn]:
                    if access in out_data.keys():
                        rem_inpconns.remove(inpconn)
                        break
        if len(rem_inpconns) > 0:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def _remove_edge_path(self,
                          state: SDFGState,
                          edge_map: Dict[str, MultiConnectorEdge],
                          unused: Set[str],
                          reverse: bool = False) -> List[MultiConnectorEdge]:
        """ Remove all edges along a path, until memlet tree contains siblings
            that should not be removed. Removes resulting isolated nodes as
            well. Operates in place.
            :param state: The state in which to remove edges.
            :param edge_map: Mapping from identifier to edge, used as a
                             predicate for removal.
            :param unused: Set of edge identifiers to remove.
            :param reverse: If False, removes forward in path, otherwise
                            backward.
            :return: List of edges from removed nodes at the path's end.
        """

        if reverse:
            edge_func = lambda e: state.out_edges(e.src)
            edge_pred = lambda pedge, e: e.src_conn == pedge.src_conn
        else:
            edge_func = lambda e: state.in_edges(e.dst)
            edge_pred = lambda pedge, e: e.dst_conn == pedge.dst_conn

        result = []

        for identifier, edge in edge_map.items():
            if identifier in unused:
                path = state.memlet_path(edge)
                pedge = None
                for pedge in (reversed(path) if reverse else path):
                    # If there are no other edges, it is safe to remove
                    if len([e for e in edge_func(pedge)
                            if edge_pred(pedge, e)]) == 1:
                        # Remove connectors as well
                        state.remove_edge_and_connectors(pedge)
                    else:
                        break
                else:  # Reached terminus without breaking, remove external node
                    if pedge is not None:
                        node = pedge.src if reverse else pedge.dst

                        # Keep track of edges on the other end of these nodes,
                        # they will be used to reconnect to first/last
                        # occurrence of access nodes in the inlined subgraph.
                        if reverse:
                            result.extend(state.in_edges(node))
                        else:
                            result.extend(state.out_edges(node))

                        state.remove_node(node)

        return result

    def apply(self, sdfg: SDFG):
        state: SDFGState = sdfg.nodes()[self.state_id]
        nsdfg_node = state.nodes()[self.subgraph[InlineSDFG._nested_sdfg]]
        nsdfg: SDFG = nsdfg_node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]

        if nsdfg_node.schedule is not dtypes.ScheduleType.Default:
            infer_types.set_default_schedule_and_storage_types(
                nsdfg, nsdfg_node.schedule)

        nsdfg_scope_entry = state.entry_node(nsdfg_node)
        nsdfg_scope_exit = (state.exit_node(nsdfg_scope_entry)
                            if nsdfg_scope_entry is not None else None)

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
        for node in nstate.nodes():
            if isinstance(node, nodes.CodeNode):
                node.environments |= nsdfg_node.environments

        # Constants
        for cstname, cstval in nsdfg.constants.items():
            if cstname in sdfg.constants:
                if cstval != sdfg.constants[cstname]:
                    warnings.warn('Constant value mismatch for "%s" while '
                                  'inlining SDFG. Inner = %s != %s = outer' %
                                  (cstname, cstval, sdfg.constants[cstname]))
            else:
                sdfg.add_constant(cstname, cstval)

        # Find original source/destination edges (there is only one edge per
        # connector, according to match)
        inputs: Dict[str, MultiConnectorEdge] = {}
        outputs: Dict[str, MultiConnectorEdge] = {}
        input_set: Dict[str, str] = {}
        output_set: Dict[str, str] = {}
        for e in state.in_edges(nsdfg_node):
            inputs[e.dst_conn] = e
            input_set[e.data.data] = e.dst_conn
        for e in state.out_edges(nsdfg_node):
            outputs[e.src_conn] = e
            output_set[e.data.data] = e.src_conn

        # Access nodes that need to be reshaped
        reshapes: Set(str) = set()
        for aname, array in nsdfg.arrays.items():
            if array.transient:
                continue
            edge = None
            if aname in inputs:
                edge = inputs[aname]
                if len(array.shape) > len(edge.data.subset):
                    reshapes.add(aname)
                    continue
            if aname in outputs:
                edge = outputs[aname]
                if len(array.shape) > len(edge.data.subset):
                    reshapes.add(aname)
                    continue
            if edge is not None and not InlineSDFG._check_strides(
                    array.strides, sdfg.arrays[edge.data.data].strides,
                    edge.data, nsdfg_node):
                reshapes.add(aname)

        # Replace symbols using invocation symbol mapping
        # Two-step replacement (N -> __dacesym_N --> map[N]) to avoid clashes
        for symname, symvalue in nsdfg_node.symbol_mapping.items():
            if str(symname) != str(symvalue):
                nsdfg.replace(symname, '__dacesym_' + symname)
        for symname, symvalue in nsdfg_node.symbol_mapping.items():
            if str(symname) != str(symvalue):
                nsdfg.replace('__dacesym_' + symname, symvalue)

        # All transients become transients of the parent (if data already
        # exists, find new name)
        # Mapping from nested transient name to top-level name
        transients: Dict[str, str] = {}
        for node in nstate.nodes():
            if isinstance(node, nodes.AccessNode):
                datadesc = nsdfg.arrays[node.data]
                if node.data not in transients and datadesc.transient:
                    name = sdfg.add_datadesc('%s_%s' % (nsdfg.label, node.data),
                                             datadesc,
                                             find_new_name=True)
                    transients[node.data] = name

        # All transients of edges between code nodes are also added to parent
        for edge in nstate.edges():
            if (isinstance(edge.src, nodes.CodeNode)
                    and isinstance(edge.dst, nodes.CodeNode)):
                if edge.data.data is not None:
                    datadesc = nsdfg.arrays[edge.data.data]
                    if edge.data.data not in transients and datadesc.transient:
                        name = sdfg.add_datadesc('%s_%s' %
                                                 (nsdfg.label, edge.data.data),
                                                 datadesc,
                                                 find_new_name=True)
                        transients[edge.data.data] = name

        # Collect nodes to add to top-level graph
        new_incoming_edges: Dict[nodes.Node, MultiConnectorEdge] = {}
        new_outgoing_edges: Dict[nodes.Node, MultiConnectorEdge] = {}

        source_accesses = set()
        sink_accesses = set()
        for node in nstate.source_nodes():
            if (isinstance(node, nodes.AccessNode)
                    and node.data not in transients
                    and node.data not in reshapes):
                try:
                    new_incoming_edges[node] = inputs[node.data]
                    source_accesses.add(node)
                except KeyError:
                    pass
        for node in nstate.sink_nodes():
            if (isinstance(node, nodes.AccessNode)
                    and node.data not in transients
                    and node.data not in reshapes):
                new_outgoing_edges[node] = outputs[node.data]
                sink_accesses.add(node)

        #######################################################
        # Replace data on inlined SDFG nodes/edges

        # Replace data names with their top-level counterparts
        repldict = {}
        repldict.update(transients)
        repldict.update({
            k: v.data.data
            for k, v in itertools.chain(inputs.items(), outputs.items())
        })

        # Add views whenever reshapes are necessary
        for dname in reshapes:
            desc = nsdfg.arrays[dname]
            # To avoid potential confusion, rename protected __return keyword
            if dname.startswith('__return'):
                newname = f'{nsdfg.name}_ret{dname[8:]}'
            else:
                newname = dname
            newname, _ = sdfg.add_view(newname,
                                       desc.shape,
                                       desc.dtype,
                                       storage=desc.storage,
                                       strides=desc.strides,
                                       offset=desc.offset,
                                       debuginfo=desc.debuginfo,
                                       allow_conflicts=desc.allow_conflicts,
                                       total_size=desc.total_size,
                                       alignment=desc.alignment,
                                       may_alias=desc.may_alias,
                                       find_new_name=True)
            repldict[dname] = newname

        orig_data: Dict[Union[nodes.AccessNode, MultiConnectorEdge], str] = {}
        for node in nstate.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in repldict:
                orig_data[node] = node.data
                node.data = repldict[node.data]
        for edge in nstate.edges():
            if edge.data.data in repldict:
                orig_data[edge] = edge.data.data
                edge.data.data = repldict[edge.data.data]

        # Add extra access nodes for out/in view nodes
        for node in nstate.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in reshapes:
                if nstate.in_degree(node) > 0 and nstate.out_degree(node) > 0:
                    # Such a node has to be in the output set
                    edge = outputs[node.data]

                    # Redirect outgoing edges through access node
                    out_edges = list(nstate.out_edges(node))
                    anode = nstate.add_access(edge.data.data)
                    vnode = nstate.add_access(node.data)
                    nstate.add_nedge(node, anode, edge.data)
                    nstate.add_nedge(anode, vnode, edge.data)
                    for e in out_edges:
                        nstate.remove_edge(e)
                        nstate.add_edge(vnode, e.src_conn, e.dst, e.dst_conn,
                                        e.data)

        #######################################################
        # Add nested SDFG into top-level SDFG

        # Add nested nodes into original state
        subgraph = SubgraphView(nstate, [
            n for n in nstate.nodes()
            if n not in (source_accesses | sink_accesses)
        ])
        state.add_nodes_from(subgraph.nodes())
        for edge in subgraph.edges():
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn,
                           edge.data)

        #######################################################
        # Reconnect inlined SDFG

        # If both source and sink nodes are inputs/outputs, reconnect once
        edges_to_ignore = self._modify_access_to_access(new_incoming_edges,
                                                        nsdfg, nstate, state,
                                                        orig_data)

        source_to_outer = {n: e.src for n, e in new_incoming_edges.items()}
        sink_to_outer = {n: e.dst for n, e in new_outgoing_edges.items()}
        # If a source/sink node is one of the inputs/outputs, reconnect it,
        # replacing memlets in outgoing/incoming paths
        modified_edges = set()
        modified_edges |= self._modify_memlet_path(new_incoming_edges, nstate,
                                                   state, sink_to_outer, True,
                                                   edges_to_ignore)
        modified_edges |= self._modify_memlet_path(new_outgoing_edges, nstate,
                                                   state, source_to_outer,
                                                   False, edges_to_ignore)

        # Reshape: add connections to viewed data
        self._modify_reshape_data(reshapes, repldict, inputs, nstate, state,
                                  True)
        self._modify_reshape_data(reshapes, repldict, outputs, nstate, state,
                                  False)

        # Modify all other internal edges pertaining to input/output nodes
        for node in subgraph.nodes():
            if isinstance(node, nodes.AccessNode):
                if node.data in input_set or node.data in output_set:
                    if node.data in input_set:
                        outer_edge = inputs[input_set[node.data]]
                    else:
                        outer_edge = outputs[output_set[node.data]]

                    for edge in state.all_edges(node):
                        if (edge not in modified_edges
                                and edge.data.data == node.data):
                            for e in state.memlet_tree(edge):
                                if e.data.data == node.data:
                                    e._data = helpers.unsqueeze_memlet(
                                        e.data, outer_edge.data)

        # If source/sink node is not connected to a source/destination access
        # node, and the nested SDFG is in a scope, connect to scope with empty
        # memlets
        if nsdfg_scope_entry is not None:
            for node in subgraph.nodes():
                if state.in_degree(node) == 0:
                    state.add_edge(nsdfg_scope_entry, None, node, None,
                                   Memlet())
                if state.out_degree(node) == 0:
                    state.add_edge(node, None, nsdfg_scope_exit, None, Memlet())

        # Replace nested SDFG parents with new SDFG
        for node in nstate.nodes():
            if isinstance(node, nodes.NestedSDFG):
                node.sdfg.parent = state
                node.sdfg.parent_sdfg = sdfg
                node.sdfg.parent_nsdfg_node = node

        # Remove all unused external inputs/output memlet paths, as well as
        # resulting isolated nodes
        inverse_repldict = {v: k for k, v in repldict.items()}
        removed_in_edges = self._remove_edge_path(
            state,
            inputs,
            set(inputs.keys()) -
            {inverse_repldict[n.data]
             for n in source_accesses},
            reverse=True)
        removed_out_edges = self._remove_edge_path(
            state,
            outputs,
            set(outputs.keys()) -
            {inverse_repldict[n.data]
             for n in sink_accesses},
            reverse=False)

        # Re-add in/out edges to first/last nodes in subgraph
        order = [
            x for x in nx.topological_sort(nstate._nx)
            if isinstance(x, nodes.AccessNode)
        ]
        for edge in removed_in_edges:
            # Find first access node that refers to this edge
            try:
                node = next(n for n in order if n.data == edge.data.data)
            except StopIteration:
                raise NameError(
                    f'Access node with data "{edge.data.data}" not found in'
                    f' nested SDFG "{nsdfg.name}" while inlining '
                    '(reconnecting inputs)')
            state.add_edge(edge.src, edge.src_conn, node, edge.dst_conn,
                           edge.data)
        for edge in removed_out_edges:
            # Find last access node that refers to this edge
            try:
                node = next(n for n in reversed(order)
                            if n.data == edge.data.data)
            except StopIteration:
                raise NameError(
                    f'Access node with data "{edge.data.data}" not found in'
                    f' nested SDFG "{nsdfg.name}" while inlining '
                    '(reconnecting outputs)')
            state.add_edge(node, edge.src_conn, edge.dst, edge.dst_conn,
                           edge.data)

        #######################################################
        # Remove nested SDFG node
        state.remove_node(nsdfg_node)

    def _modify_access_to_access(
        self,
        input_edges: Dict[nodes.Node, MultiConnectorEdge],
        nsdfg: SDFG,
        nstate: SDFGState,
        state: SDFGState,
        orig_data: Dict[Union[nodes.AccessNode, MultiConnectorEdge], str],
    ) -> Set[MultiConnectorEdge]:
        """
        Deals with access->access edges where both sides are non-transient.
        """
        result = set()
        for node, top_edge in input_edges.items():
            for inner_edge in nstate.out_edges(node):
                if inner_edge.dst not in orig_data:
                    continue
                inner_data = orig_data[inner_edge.dst]
                if (isinstance(inner_edge.dst, nodes.AccessNode)
                        and not nsdfg.arrays[inner_data].transient):
                    matching_edge: MultiConnectorEdge = next(
                        state.out_edges_by_connector(top_edge.dst, inner_data))
                    # Create memlet by unsqueezing both w.r.t. src and dst
                    # subsets
                    in_memlet = helpers.unsqueeze_memlet(
                        inner_edge.data, top_edge.data)
                    out_memlet = helpers.unsqueeze_memlet(
                        inner_edge.data, matching_edge.data)
                    new_memlet = in_memlet
                    new_memlet.other_subset = out_memlet.subset

                    # Connect with new edge
                    state.add_edge(top_edge.src, top_edge.src_conn,
                                   matching_edge.dst, matching_edge.dst_conn,
                                   new_memlet)
                    result.add(inner_edge)

        return result

    def _modify_memlet_path(
        self,
        new_edges: Dict[nodes.Node, MultiConnectorEdge],
        nstate: SDFGState,
        state: SDFGState,
        inner_to_outer: Dict[nodes.Node, MultiConnectorEdge],
        inputs: bool,
        edges_to_ignore: Set[MultiConnectorEdge],
    ) -> Set[MultiConnectorEdge]:
        """ Modifies memlet paths in an inlined SDFG. Returns set of modified
            edges.
        """
        result = set()
        for node, top_edge in new_edges.items():
            inner_edges = (nstate.out_edges(node)
                           if inputs else nstate.in_edges(node))
            for inner_edge in inner_edges:
                if inner_edge in edges_to_ignore:
                    continue
                new_memlet = helpers.unsqueeze_memlet(inner_edge.data,
                                                      top_edge.data)
                if inputs:
                    if inner_edge.dst in inner_to_outer:
                        dst = inner_to_outer[inner_edge.dst]
                    else:
                        dst = inner_edge.dst

                    new_edge = state.add_edge(top_edge.src, top_edge.src_conn,
                                              dst, inner_edge.dst_conn,
                                              new_memlet)
                    mtree = state.memlet_tree(new_edge)
                else:
                    if inner_edge.src in inner_to_outer:
                        # don't add edges twice
                        continue

                    new_edge = state.add_edge(inner_edge.src,
                                              inner_edge.src_conn, top_edge.dst,
                                              top_edge.dst_conn, new_memlet)
                    mtree = state.memlet_tree(new_edge)

                # Modify all memlets going forward/backward
                def traverse(mtree_node):
                    result.add(mtree_node.edge)
                    mtree_node.edge._data = helpers.unsqueeze_memlet(
                        mtree_node.edge.data, top_edge.data)
                    for child in mtree_node.children:
                        traverse(child)

                for child in mtree.children:
                    traverse(child)

        return result

    def _modify_reshape_data(self, reshapes: Set[str], repldict: Dict[str, str],
                             new_edges: Dict[str, MultiConnectorEdge],
                             nstate: SDFGState, state: SDFGState, inputs: bool):
        anodes = nstate.source_nodes() if inputs else nstate.sink_nodes()
        reshp = {repldict[r]: r for r in reshapes}
        for node in anodes:
            if not isinstance(node, nodes.AccessNode):
                continue
            if node.data not in reshp:
                continue
            edge = new_edges[reshp[node.data]]
            if inputs:
                state.add_edge(edge.src, edge.src_conn, node, None, edge.data)
            else:
                state.add_edge(node, None, edge.dst, edge.dst_conn, edge.data)


@registry.autoregister_params(singlestate=True)
@make_properties
class InlineTransients(transformation.Transformation):
    """
    Inlines all transient arrays that are not used anywhere else into a
    nested SDFG.
    """

    nsdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(InlineTransients.nsdfg)]

    @staticmethod
    def _candidates(sdfg: SDFG, graph: SDFGState,
                    nsdfg: nodes.NestedSDFG) -> Dict[str, str]:
        candidates = {}
        for e in graph.all_edges(nsdfg):
            if e.data.is_empty():
                continue
            conn = (e.src_conn if e.src is nsdfg else e.dst_conn)
            desc = sdfg.arrays[e.data.data]
            # Needs to be transient
            if not desc.transient:
                continue
            # Needs to be allocated in "Scope" lifetime
            if desc.lifetime is not dtypes.AllocationLifetime.Scope:
                continue
            # If same transient is connected with multiple connectors, bail
            # for now
            if e.data.data in candidates and candidates[e.data.data] != conn:
                del candidates[e.data.data]
                continue
            # (for now) needs to use entire data descriptor (skipped due to
            # above check for multiple connectors)
            # if desc.shape != e.data.subset.size():
            #     continue
            candidates[e.data.data] = conn

        if not candidates:
            return candidates

        # Check for uses in other states
        for state in sdfg.nodes():
            if state is graph:
                continue
            for node in state.data_nodes():
                if node.data in candidates:
                    del candidates[node.data]

        if not candidates:
            return candidates

        # Check for uses in state
        access_nodes = set()
        for e in graph.in_edges(nsdfg):
            src = graph.memlet_path(e)[0].src
            if isinstance(src, nodes.AccessNode) and graph.in_degree(src) == 0:
                access_nodes.add(src)
        for e in graph.out_edges(nsdfg):
            dst = graph.memlet_path(e)[-1].dst
            if isinstance(dst, nodes.AccessNode) and graph.out_degree(dst) == 0:
                access_nodes.add(dst)
        for node in graph.data_nodes():
            if node.data in candidates and node not in access_nodes:
                del candidates[node.data]

        return candidates

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate: Dict[transformation.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False):
        nsdfg = graph.node(candidate[InlineTransients.nsdfg])

        # Not every schedule is supported
        if strict:
            if nsdfg.schedule not in (dtypes.ScheduleType.Default,
                                      dtypes.ScheduleType.Sequential,
                                      dtypes.ScheduleType.CPU_Multicore,
                                      dtypes.ScheduleType.GPU_Device):
                return False

        candidates = InlineTransients._candidates(sdfg, graph, nsdfg)
        return len(candidates) > 0

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg):
        state: SDFGState = sdfg.nodes()[self.state_id]
        nsdfg_node: nodes.NestedSDFG = self.nsdfg(sdfg)
        nsdfg: SDFG = nsdfg_node.sdfg
        toremove = InlineTransients._candidates(sdfg, state, nsdfg_node)

        for dname, cname in toremove.items():
            # Make nested SDFG data descriptors transient
            nsdfg.arrays[cname].transient = True

            # Remove connectors from node
            nsdfg_node.remove_in_connector(cname)
            nsdfg_node.remove_out_connector(cname)

            # Remove data descriptor from outer SDFG
            del sdfg.arrays[dname]

        # Remove edges from outer SDFG
        for e in state.in_edges(nsdfg_node):
            if e.data.data not in toremove:
                continue
            tree = state.memlet_tree(e)
            for te in tree:
                state.remove_edge_and_connectors(te)
            # Remove newly isolated node
            state.remove_node(tree.root().edge.src)

        for e in state.out_edges(nsdfg_node):
            if e.data.data not in toremove:
                continue
            tree = state.memlet_tree(e)
            for te in tree:
                state.remove_edge_and_connectors(te)
            # Remove newly isolated node
            state.remove_node(tree.root().edge.dst)


class ASTRefiner(ast.NodeTransformer):
    """
    Python AST transformer used in ``RefineNestedAccess`` to reduce (refine) the
    subscript ranges based on the specification given in the transformation.
    """
    def __init__(self,
                 to_refine: str,
                 refine_subset: subsets.Subset,
                 sdfg: SDFG,
                 indices: Set[int] = None) -> None:
        self.to_refine = to_refine
        self.subset = refine_subset
        self.sdfg = sdfg
        self.indices = indices

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        if astutils.rname(node.value) == self.to_refine:
            rng = subsets.Range(
                astutils.subscript_to_slice(node,
                                            self.sdfg.arrays,
                                            without_array=True))
            rng.offset(self.subset, True, self.indices)
            return ast.copy_location(
                astutils.slice_to_subscript(self.to_refine, rng), node)

        return self.generic_visit(node)


@registry.autoregister_params(singlestate=True)
@make_properties
class RefineNestedAccess(transformation.Transformation):
    """
    Reduces memlet shape when a memlet is connected to a nested SDFG, but not
    using all of the contents. Makes the outer memlet smaller in shape and
    ensures that the offsets in the nested SDFG start with zero.
    This helps with subsequent transformations on the outer SDFGs.

    For example, in the following program::

        @dace.program
        def func_a(y):
            return y[1:5] + 1

        @dace.program
        def main(x: dace.float32[N]):
            return func_a(x)

    The memlet pointing to ``func_a`` will contain all of ``x`` (``x[0:N]``),
    and it is offset to ``y[1:5]`` in the function, with ``y``'s size being
    ``N``. After the transformation, the memlet connected to the nested SDFG of
    ``func_a`` would contain ``x[1:5]`` directly and the internal ``y`` array
    would have a size of 4, accessed as ``y[0:4]``.
    """

    nsdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(RefineNestedAccess.nsdfg)]

    @staticmethod
    def _candidates(
        state: SDFGState, nsdfg: nodes.NestedSDFG
    ) -> Tuple[Dict[str, Tuple[Memlet, Set[int]]], Dict[str, Tuple[Memlet,
                                                                   Set[int]]]]:
        in_candidates: Dict[str, Tuple[Memlet, SDFGState, Set[int]]] = {}
        out_candidates: Dict[str, Tuple[Memlet, SDFGState, Set[int]]] = {}
        ignore = set()
        for nstate in nsdfg.sdfg.nodes():
            for dnode in nstate.data_nodes():
                if nsdfg.sdfg.arrays[dnode.data].transient:
                    continue

                # For now we only detect one element
                for e in nstate.in_edges(dnode):
                    # If more than one unique element detected, remove from
                    # candidates
                    if e.data.data in out_candidates:
                        memlet, ns, indices = out_candidates[e.data.data]
                        # Try to find dimensions in which there is a mismatch
                        # and remove them from list
                        for i, (s1, s2) in enumerate(
                                zip(e.data.subset, memlet.subset)):
                            if s1 != s2 and i in indices:
                                indices.remove(i)
                        if len(indices) == 0:
                            ignore.add(e.data.data)
                        out_candidates[e.data.data] = (memlet, ns, indices)
                        continue
                    out_candidates[e.data.data] = (e.data, nstate,
                                                   set(range(len(
                                                       e.data.subset))))
                for e in nstate.out_edges(dnode):
                    # If more than one unique element detected, remove from
                    # candidates
                    if e.data.data in in_candidates:
                        memlet, ns, indices = in_candidates[e.data.data]
                        # Try to find dimensions in which there is a mismatch
                        # and remove them from list
                        for i, (s1, s2) in enumerate(
                                zip(e.data.subset, memlet.subset)):
                            if s1 != s2 and i in indices:
                                indices.remove(i)
                        if len(indices) == 0:
                            ignore.add(e.data.data)
                        in_candidates[e.data.data] = (memlet, ns, indices)
                        continue
                    in_candidates[e.data.data] = (e.data, nstate,
                                                  set(range(len(
                                                      e.data.subset))))

        # TODO: Check in_candidates in interstate edges as well

        # Check in/out candidates
        for cand in in_candidates.keys() & out_candidates.keys():
            s1, nstate1, ind1 = in_candidates[cand]
            s2, nstate2, ind2 = out_candidates[cand]
            indices = ind1 & ind2
            if any(s1.subset[ind] != s2.subset[ind] for ind in indices):
                ignore.add(cand)
            in_candidates[cand] = (s1, nstate1, indices)
            out_candidates[cand] = (s2, nstate2, indices)

        # Ensure minimum elements of candidates do not begin with zero
        def _check_cand(candidates, outer_edges):
            for cname, (cand, nstate, indices) in candidates.items():
                if all(me == 0 for i, me in enumerate(cand.subset.min_element())
                       if i in indices):
                    ignore.add(cname)
                    continue

                # Ensure outer memlets begin with 0
                outer_edge = next(iter(outer_edges(nsdfg, cname)))
                if any(me != 0 for i, me in enumerate(
                        outer_edge.data.subset.min_element()) if i in indices):
                    ignore.add(cname)
                    continue

                # Check w.r.t. loops
                if len(nstate.ranges) > 0:
                    # Re-annotate loop ranges, in case someone changed them
                    # TODO: Move out of here!
                    nstate.ranges = {}
                    from dace.sdfg.propagation import _annotate_loop_ranges
                    _annotate_loop_ranges(nsdfg.sdfg, [])

                    memlet = propagation.propagate_subset(
                        [cand], nsdfg.sdfg.arrays[cname],
                        sorted(nstate.ranges.keys()),
                        subsets.Range([
                            v.ndrange()[0]
                            for _, v in sorted(nstate.ranges.items())
                        ]))
                    if all(me == 0
                           for i, me in enumerate(memlet.subset.min_element())
                           if i in indices):
                        ignore.add(cname)
                        continue

                    # Modify memlet to propagated one
                    candidates[cname] = (memlet, nstate, indices)
                else:
                    memlet = cand

                # If there are any symbols here that are not defined
                # in "defined_symbols"
                missing_symbols = (memlet.free_symbols -
                                   set(nsdfg.symbol_mapping.keys()))
                if missing_symbols:
                    ignore.add(cname)
                    continue

        _check_cand(in_candidates, state.in_edges_by_connector)
        _check_cand(out_candidates, state.out_edges_by_connector)

        # Return result, filtering out the states
        return ({
            k: (dc(v), ind)
            for k, (v, _, ind) in in_candidates.items() if k not in ignore
        }, {
            k: (dc(v), ind)
            for k, (v, _, ind) in out_candidates.items() if k not in ignore
        })

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate: Dict[transformation.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False):
        nsdfg = graph.node(candidate[RefineNestedAccess.nsdfg])
        ic, oc = RefineNestedAccess._candidates(graph, nsdfg)
        return (len(ic) + len(oc)) > 0

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg):
        state: SDFGState = sdfg.nodes()[self.state_id]
        nsdfg_node: nodes.NestedSDFG = self.nsdfg(sdfg)
        nsdfg: SDFG = nsdfg_node.sdfg
        torefine_in, torefine_out = RefineNestedAccess._candidates(
            state, nsdfg_node)

        refined = set()

        def _offset_refine(
            torefine: Dict[str, Tuple[Memlet, Set[int]]],
            outer_edges: Callable[[nodes.NestedSDFG, str],
                                  Iterable[MultiConnectorEdge[Memlet]]]):
            # Offset memlets inside negatively by "refine", modify outer
            # memlets to be "refine"
            for aname, (refine, indices) in torefine.items():
                outer_edge = next(iter(outer_edges(nsdfg_node, aname)))
                new_memlet = helpers.unsqueeze_memlet(refine, outer_edge.data)
                outer_edge.data.subset = subsets.Range([
                    ns if i in indices else os for i, (os, ns) in enumerate(
                        zip(outer_edge.data.subset, new_memlet.subset))
                ])
                if aname in refined:
                    continue
                # Refine internal memlets
                for nstate in nsdfg.nodes():
                    for e in nstate.edges():
                        if e.data.data == aname:
                            e.data.subset.offset(refine.subset, True, indices)
                # Refine accesses in interstate edges
                refiner = ASTRefiner(aname, refine.subset, nsdfg, indices)
                for isedge in nsdfg.edges():
                    for k, v in isedge.data.assignments.items():
                        vast = ast.parse(v)
                        refiner.visit(vast)
                        isedge.data.assignments[k] = astutils.unparse(vast)
                    if isedge.data.condition.language is dtypes.Language.Python:
                        for i, stmt in enumerate(isedge.data.condition.code):
                            isedge.data.condition.code[i] = refiner.visit(stmt)
                    else:
                        raise NotImplementedError
                refined.add(aname)

        # Proceed symmetrically on incoming and outgoing edges
        _offset_refine(torefine_in, state.in_edges_by_connector)
        _offset_refine(torefine_out, state.out_edges_by_connector)


@registry.autoregister
@make_properties
class NestSDFG(transformation.Transformation):
    """ Implements SDFG Nesting, taking an SDFG as an input and creating a
        nested SDFG node from it. """

    promote_global_trans = Property(
        dtype=bool,
        default=False,
        desc="Promotes transients to be allocated once")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [nx.DiGraph()]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg):

        outer_sdfg = sdfg
        nested_sdfg = dc(sdfg)

        outer_sdfg.arrays.clear()
        outer_sdfg.remove_nodes_from(outer_sdfg.nodes())

        inputs = {}
        outputs = {}
        transients = {}

        for state in nested_sdfg.nodes():
            #  Input and output nodes are added as input and output nodes of the nested SDFG
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode)
                        and not node.desc(nested_sdfg).transient):
                    if (state.out_degree(node) > 0):  # input node
                        arrname = node.data
                        arrname_nested = f"__{arrname}_in"
                        if arrname not in inputs:
                            arrobj = nested_sdfg.arrays[arrname]
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                            nested_sdfg.arrays[arrname_nested] = arrobj
                            inputs[arrname] = arrname_nested
                        node_data_name = arrname_nested
                    if (state.in_degree(node) > 0):  # output node
                        arrname = node.data
                        arrname_nested = f"__{arrname}_out"
                        if arrname not in outputs:
                            arrobj = nested_sdfg.arrays[arrname]
                            if arrname not in inputs:
                                outer_sdfg.arrays[arrname] = dc(arrobj)
                            nested_sdfg.arrays[arrname_nested] = arrobj
                            outputs[arrname] = arrname_nested
                        node_data_name = arrname_nested
                    node.data = node_data_name

            if self.promote_global_trans:
                scope_dict = state.scope_dict()
                for node in state.nodes():
                    if (isinstance(node, nodes.AccessNode)
                            and node.desc(nested_sdfg).transient and
                            not isinstance(node.desc(nested_sdfg), data.View)):
                        nodedesc = node.desc(nested_sdfg)

                        # If this transient has a symbolic shape, and if any symbol is in in the "ranges"
                        # of the state then substitute it with its max value (if it can be inferred).
                        # This is useful for the cases where the transient comes from a slice operation
                        # (e.g. array[:i] or array[i:]), and we are on devices such as FPGAs that do not
                        # support dynamic memory allocation.

                        propagation.propagate_states(nested_sdfg)
                        overapprox_shape = []
                        if not isinstance(nodedesc,
                                          data.Scalar) and state.ranges:
                            for sz in nodedesc.shape:
                                newsz = sz

                                if symbolic.issymbolic(sz):
                                    for s in newsz.free_symbols:

                                        replacement_limit_value = None
                                        to_solve_limit_value = copy.deepcopy(s)
                                        replacement_initial_value = None

                                        to_solve_initial_value = copy.deepcopy(
                                            s)

                                        # We should detect the maximal size, therefore we consider the
                                        # state ranges, by looking both at the initial and the final value

                                        # Range Limit value
                                        while str(to_solve_limit_value
                                                  ) in state.ranges.keys():
                                            replacement_limit_value = state.ranges[
                                                str(to_solve_limit_value
                                                    )][0][1] + 1
                                            to_solve_limit_value = replacement_limit_value

                                        # Range Initial value
                                        while str(to_solve_initial_value
                                                  ) in state.ranges.keys():
                                            replacement_initial_value = state.ranges[
                                                str(to_solve_initial_value
                                                    )][0][0]
                                            to_solve_initial_value = replacement_initial_value

                                        if replacement_initial_value is not None and replacement_limit_value is not None:
                                            # We compute the shape by considering both the range initial and limit value

                                            # Note: here we are lenient. We can't evaluate the maximum of the two,
                                            # since we don't know the value of symbols, therefore we only take the one
                                            # that is not negative

                                            newsz_limit = newsz.subs(
                                                {s: replacement_limit_value})
                                            newsz_initial = newsz.subs(
                                                {s: replacement_initial_value})
                                            if newsz_limit.is_negative:
                                                if newsz_initial.is_negative:
                                                    raise ValueError(
                                                        f"Can not over-approximate shape for transient{node.data}"
                                                    )
                                                newsz = newsz_initial
                                            else:
                                                newsz = newsz_limit
                                overapprox_shape.append(newsz)
                            nodedesc.shape = overapprox_shape
                            nodedesc.total_size = reduce(
                                operator.mul, nodedesc.shape, 1)

                        arrname = node.data
                        if not scope_dict[node]:
                            arrname_nested = f"__{arrname}_out"
                            node.data = arrname_nested
                            if arrname not in transients:
                                arrobj = nested_sdfg.arrays[arrname]
                                outer_sdfg.arrays[arrname] = dc(arrobj)
                                nested_sdfg.arrays[arrname_nested] = arrobj
                                transients[arrname] = arrname_nested
                                if state.out_degree(node) > 0:
                                    inputs[arrname] = arrname_nested
                                if state.in_degree(node) > 0:
                                    outputs[arrname] = arrname_nested

        # Catch data containers that we didn't find on any access nodes, and add
        # them as inputs. This can happen when a scalar input is used on an
        # interstate edge, and thus doesn't appear in the dataflow.
        nested_data = set(
            itertools.chain(inputs.values(), outputs.values(),
                            transients.values()))
        for arrname, desc in list(nested_sdfg.arrays.items()):
            if not desc.transient and arrname not in nested_data:
                arrname_nested = f"__{arrname}_in"
                outer_sdfg.arrays[arrname] = dc(desc)
                nested_sdfg.arrays[arrname_nested] = desc
                inputs[arrname] = arrname_nested

        # Purge the old descriptors
        for name in set(itertools.chain(inputs, outputs, transients)):
            del nested_sdfg.arrays[name]

        for newarrname in transients.values():
            nested_sdfg.arrays[newarrname].transient = False

        # Update memlets
        for state in nested_sdfg.nodes():
            for _, edge in enumerate(state.edges()):
                _, _, _, _, mem = edge
                src = state.memlet_path(edge)[0].src
                dst = state.memlet_path(edge)[-1].dst
                if isinstance(src, nodes.AccessNode):
                    if (mem.data in inputs.keys()
                            and src.data == inputs[mem.data]):
                        mem.data = inputs[mem.data]
                    elif (mem.data in outputs.keys()
                          and (src.data == outputs[mem.data] or
                               (isinstance(dst, nodes.AccessNode)
                                and dst.data == outputs[mem.data]))):
                        mem.data = outputs[mem.data]
                elif (isinstance(dst, nodes.AccessNode)
                      and mem.data in outputs.keys()
                      and dst.data == outputs[mem.data]):
                    mem.data = outputs[mem.data]
        outer_state = outer_sdfg.add_state(outer_sdfg.label)

        # Clean up any remaining mentions of input nodes in the nested SDFG
        for before, after in inputs.items():
            nested_sdfg.replace(before, after)

        # Remove from the parent SDFG the symbols that are defined in the nested one
        defined_syms = set()

        for name, desc in nested_sdfg.arrays.items():
            defined_syms.add(name)

        for e in nested_sdfg.edges():
            defined_syms |= set(e.data.new_symbols({}).keys())

        defined_syms |= set(nested_sdfg.constants.keys())

        for s in defined_syms:
            type = outer_sdfg.symbols.pop(s, None)
            if type is not None:
                # update or add the symbol in the nested sdfg
                nested_sdfg.symbols[s] = type

        # Add the nested SDFG to the parent state and connect it
        nested_node = outer_state.add_nested_sdfg(nested_sdfg, outer_sdfg,
                                                  set(inputs.values()),
                                                  set(outputs.values()))

        for key, val in inputs.items():
            arrnode = outer_state.add_read(key)
            outer_state.add_edge(
                arrnode, None, nested_node, val,
                memlet.Memlet.from_array(key, arrnode.desc(outer_sdfg)))
        for key, val in outputs.items():
            arrnode = outer_state.add_write(key)
            outer_state.add_edge(
                nested_node, val, arrnode, None,
                memlet.Memlet.from_array(key, arrnode.desc(outer_sdfg)))
