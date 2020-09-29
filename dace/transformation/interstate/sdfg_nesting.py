# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG nesting transformation. """

from copy import deepcopy as dc
import itertools
import networkx as nx
from typing import Dict, List, Set, Optional
import warnings

from dace import memlet, registry, sdfg as sd, Memlet, symbolic
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.transformation import transformation, helpers
from dace.properties import make_properties, Property


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
        # Matches anything
        return [sdutil.node_path_graph(InlineSDFG._nested_sdfg)]

    @staticmethod
    def _find_edge(state: SDFGState, node: nodes.Node,
                   connector: str) -> Optional[MultiConnectorEdge]:
        for edge in state.in_edges(node):
            if edge.dst_conn == connector:
                return edge
        for edge in state.out_edges(node):
            if edge.src_conn == connector:
                return edge
        raise NameError('Edge with connector %s not found on node %s' %
                        (connector, node))

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
        # Take unsqueezing into account
        dims_to_ignore = [
            i for i, s in enumerate(memlet.subset.size()) if s == 1
        ]
        ostrides = [
            os for i, os in enumerate(outer_strides) if i not in dims_to_ignore
        ]
        if len(ostrides) == 0:
            ostrides = [1]
        if len(ostrides) != len(inner_strides):
            return False

        # Replace all inner symbols based on symbol mapping
        repldict = {
            symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v)
            for k, v in nested_sdfg.symbol_mapping.items()
        }
        istrides = [
            istr.subs(repldict) if symbolic.issymbolic(istr) else istr
            for istr in inner_strides
        ]

        return all(istr == ostr for istr, ostr in zip(istrides, ostrides))

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        nested_sdfg = graph.nodes()[candidate[InlineSDFG._nested_sdfg]]
        if len(nested_sdfg.sdfg.nodes()) != 1:
            return False

        # Ensure every connector has one incoming/outgoing edge
        in_connectors = set()
        out_connectors = set()
        for edge in graph.in_edges(nested_sdfg):
            if edge.dst_conn in in_connectors:
                return False
            in_connectors.add(edge.dst_conn)
        for edge in graph.out_edges(nested_sdfg):
            if edge.src_conn in out_connectors:
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

        # If some reshaping that cannot be inlined / unsqueezed is happening,
        # do not match transformation in strict mode.
        if strict:
            for aname, array in nested_sdfg.sdfg.arrays.items():
                if array.transient:
                    continue
                edge = InlineSDFG._find_edge(graph, nested_sdfg, aname)
                if len(array.shape) > len(edge.data.subset):
                    return False
                if not InlineSDFG._check_strides(
                        array.strides, sdfg.arrays[edge.data.data].strides,
                        edge.data, nested_sdfg):
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

    def apply(self, sdfg):
        state: SDFGState = sdfg.nodes()[self.state_id]
        nsdfg_node = state.nodes()[self.subgraph[InlineSDFG._nested_sdfg]]
        nsdfg: SDFG = nsdfg_node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]

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
                    and node.data not in transients):
                new_incoming_edges[node] = inputs[node.data]
                source_accesses.add(node)
        for node in nstate.sink_nodes():
            if (isinstance(node, nodes.AccessNode)
                    and node.data not in transients):
                new_outgoing_edges[node] = outputs[node.data]
                sink_accesses.add(node)

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
        # Replace data on inlined SDFG nodes/edges

        # Replace symbols using invocation symbol mapping
        # Two-step replacement (N -> __dacesym_N --> map[N]) to avoid clashes
        for symname, symvalue in nsdfg_node.symbol_mapping.items():
            if str(symname) != str(symvalue):
                nsdfg.replace(symname, '__dacesym_' + symname)
        for symname, symvalue in nsdfg_node.symbol_mapping.items():
            if str(symname) != str(symvalue):
                nsdfg.replace('__dacesym_' + symname, symvalue)

        # Replace data names with their top-level counterparts
        repldict = {}
        repldict.update(transients)
        repldict.update({
            k: v.data.data
            for k, v in itertools.chain(inputs.items(), outputs.items())
        })
        for node in subgraph.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in repldict:
                node.data = repldict[node.data]
        for edge in subgraph.edges():
            if edge.data.data in repldict:
                edge.data.data = repldict[edge.data.data]

        #######################################################
        # Reconnect inlined SDFG

        # If a source/sink node is one of the inputs/outputs, reconnect it,
        # replacing memlets in outgoing/incoming paths
        modified_edges = set()
        modified_edges |= self._modify_memlet_path(new_incoming_edges, nstate,
                                                   state, True)
        modified_edges |= self._modify_memlet_path(new_outgoing_edges, nstate,
                                                   state, False)

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
        removed_in_edges = self._remove_edge_path(state,
                                                  inputs,
                                                  set(inputs.keys()) -
                                                  source_accesses,
                                                  reverse=True)
        removed_out_edges = self._remove_edge_path(state,
                                                   outputs,
                                                   set(outputs.keys()) -
                                                   sink_accesses,
                                                   reverse=False)

        # Re-add in/out edges to first/last nodes in subgraph
        order = [
            x for x in nx.topological_sort(nstate._nx)
            if isinstance(x, nodes.AccessNode)
        ]
        for edge in removed_in_edges:
            # Find first access node that refers to this edge
            node = next(n for n in order if n.data == edge.data.data)
            state.add_edge(edge.src, edge.src_conn, node, edge.dst_conn,
                           edge.data)
        for edge in removed_out_edges:
            # Find last access node that refers to this edge
            node = next(n for n in reversed(order) if n.data == edge.data.data)
            state.add_edge(node, edge.src_conn, edge.dst, edge.dst_conn,
                           edge.data)

        #######################################################
        # Remove nested SDFG node
        state.remove_node(nsdfg_node)

    def _modify_memlet_path(self, new_edges: Dict[nodes.Node,
                                                  MultiConnectorEdge],
                            nstate: SDFGState, state: SDFGState,
                            inputs: bool) -> Set[MultiConnectorEdge]:
        """ Modifies memlet paths in an inlined SDFG. Returns set of modified
            edges.
        """
        result = set()
        for node, top_edge in new_edges.items():
            inner_edges = (nstate.out_edges(node)
                           if inputs else nstate.in_edges(node))
            for inner_edge in inner_edges:
                new_memlet = helpers.unsqueeze_memlet(inner_edge.data,
                                                      top_edge.data)
                if inputs:
                    new_edge = state.add_edge(top_edge.src, top_edge.src_conn,
                                              inner_edge.dst,
                                              inner_edge.dst_conn, new_memlet)
                    mtree = state.memlet_tree(new_edge)
                else:
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
                        if arrname not in inputs:
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays['__' + arrname + '_in'] = arrobj
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                            inputs[arrname] = '__' + arrname + '_in'
                        node_data_name = '__' + arrname + '_in'
                    if (state.in_degree(node) > 0):  # output node
                        arrname = node.data
                        if arrname not in outputs:
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays['__' + arrname + '_out'] = arrobj
                            if arrname not in inputs:
                                outer_sdfg.arrays[arrname] = dc(arrobj)
                            outputs[arrname] = '__' + arrname + '_out'
                        node_data_name = '__' + arrname + '_out'
                    node.data = node_data_name

            if self.promote_global_trans:
                scope_dict = state.scope_dict()
                for node in state.nodes():
                    if (isinstance(node, nodes.AccessNode)
                            and node.desc(nested_sdfg).transient):

                        arrname = node.data
                        if arrname not in transients and not scope_dict[node]:
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays['__' + arrname + '_out'] = arrobj
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                            transients[arrname] = '__' + arrname + '_out'
                        node.data = '__' + arrname + '_out'

        for arrname in inputs.keys():
            nested_sdfg.arrays.pop(arrname)
        for arrname in outputs.keys():
            nested_sdfg.arrays.pop(arrname, None)
        for oldarrname, newarrname in transients.items():
            nested_sdfg.arrays.pop(oldarrname)
            nested_sdfg.arrays[newarrname].transient = False
        outputs.update(transients)

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
                          and src.data == outputs[mem.data]):
                        mem.data = outputs[mem.data]
                elif (isinstance(dst, nodes.AccessNode)
                      and mem.data in outputs.keys()
                      and dst.data == outputs[mem.data]):
                    mem.data = outputs[mem.data]

        outer_state = outer_sdfg.add_state(outer_sdfg.label)

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
