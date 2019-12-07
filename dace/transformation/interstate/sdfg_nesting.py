""" SDFG nesting transformation. """

from copy import deepcopy as dc
import networkx as nx

import dace
from dace import data as dt, memlet, sdfg as sd, subsets, Memlet, EmptyMemlet
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property


@make_properties
class NestSDFG(pattern_matching.Transformation):
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
                            nested_sdfg.arrays[arrname + '_in'] = arrobj
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                            inputs[arrname] = arrname + '_in'
                        node_data_name = arrname + '_in'
                    if (state.in_degree(node) > 0):  # output node
                        arrname = node.data
                        if arrname not in outputs:
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays[arrname + '_out'] = arrobj
                            if arrname not in inputs:
                                outer_sdfg.arrays[arrname] = dc(arrobj)
                            outputs[arrname] = arrname + '_out'
                        node_data_name = arrname + '_out'
                    node.data = node_data_name

            if self.promote_global_trans:
                scope_dict = state.scope_dict()
                for node in state.nodes():
                    if (isinstance(node, nodes.AccessNode)
                            and node.desc(nested_sdfg).transient):

                        arrname = node.data
                        if arrname not in transients and not scope_dict[node]:
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays[arrname + '_out'] = arrobj
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                            transients[arrname] = arrname + '_out'
                        node.data = arrname + '_out'

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
                                                  inputs.values(),
                                                  outputs.values())
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


@make_properties
class InlineSDFG(pattern_matching.Transformation):
    """ Inlines a single-state nested SDFG into a top-level SDFG """

    _nested_sdfg = nodes.NestedSDFG('_', sd.SDFG('_'), set(), set())

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [nxutil.node_path_graph(InlineSDFG._nested_sdfg)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        nested_sdfg = graph.nodes()[candidate[InlineSDFG._nested_sdfg]]
        if len(nested_sdfg.sdfg.nodes()) != 1:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def _modify_memlet(self, internal_memlet: Memlet, external_memlet: Memlet):
        """ Unsqueezes and offsets a memlet, as per the semantics of nested
            SDFGs.
            :param internal_memlet: The internal memlet (inside nested SDFG)
                                    before modification.
            :param internal_memlet: The external memlet before modification.
            :return: Offset Memlet to set on the resulting graph.
        """
        result = dc(internal_memlet)
        result.data = external_memlet.data

        shape = external_memlet.subset.size()
        if len(internal_memlet.subset) < len(external_memlet.subset):
            ones = [i for i, d in enumerate(shape) if d == 1]

            # Special case: If internal memlet is a range of size 1 with (0,0,1),
            #               ignore it when unsqueezing
            if (len(internal_memlet.subset) == 1
                    and (internal_memlet.subset[0] == (0, 0, 1)
                         or internal_memlet.subset[0] == 0)):
                to_unsqueeze = ones[1:]
            else:
                to_unsqueeze = ones

            result.subset.unsqueeze(to_unsqueeze)
        elif len(internal_memlet.subset) > len(external_memlet.subset):
            raise ValueError(
                'Unexpected extra dimensions in internal memlet '
                'while inlining SDFG.\nExternal memlet: %s\n'
                'Internal memlet: %s' % (external_memlet, internal_memlet))

        result.subset.offset(external_memlet.subset, False)

        # TODO: Offset rest of memlet according to other_subset
        if external_memlet.other_subset is not None:
            raise NotImplementedError

        return result

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        nsdfg_node = graph.nodes()[self.subgraph[InlineSDFG._nested_sdfg]]
        nsdfg = nsdfg_node.sdfg

        # Find original source/destination nodes
        inputs = {}
        outputs = {}
        for e in graph.in_edges(nsdfg_node):
            inputs[e.dst_conn] = (e.src, e.src_conn, e.data)
        for e in graph.out_edges(nsdfg_node):
            outputs[e.src_conn] = (e.dst, e.dst_conn, e.data)

        to_reconnect = set()

        torename = {}
        torename.update({k: v[2].data for k, v in inputs.items()})
        torename.update({k: v[2].data for k, v in outputs.items()})

        # Add SDFG nodes to top-level SDFG
        state = nsdfg.nodes()[0]
        # Keep a backup of the topological sorted order of the access nodes,
        order = [
            x for x in reversed(list(nx.topological_sort(state._nx)))
            if isinstance(x, nodes.AccessNode)
        ]
        for node in state.nodes():
            # Data access nodes
            if isinstance(node, nodes.AccessNode):
                # External node
                if node.data in inputs or node.data in outputs:
                    continue
                # Internal node (e.g., transient)
                if node.data not in torename:
                    name = node.data
                    # Name already exists
                    if name in sdfg.arrays:
                        name = '%s_%s' % (nsdfg.label, node.data)
                        i = 0
                        while name in sdfg.arrays:
                            name = '%s_%s_%d' % (nsdfg.label, node.data, i)
                            i += 1
                    # Add transient
                    sdfg.arrays[name] = nsdfg.arrays[node.data]
                    # Rename all internal uses
                    torename[node.data] = name
            # Set all parents of nested SDFG nodes in the inlined SDFG to their
            # new parent
            elif isinstance(node, nodes.NestedSDFG):
                node.sdfg.parent = graph
                node.sdfg.parent_sdfg = sdfg

            graph.add_node(node)
            to_reconnect.add(node)

        # TODO: Confirm that the following is always correct
        # Add Scalars of the nested SDFG to the parent
        for name, arr in nsdfg.arrays.items():
            if isinstance(arr, dt.Scalar) and name not in sdfg.arrays:
                sdfg.arrays[name] = arr

        # Reconnect edges to their original source
        for e in state.edges():
            if isinstance(e.src, nodes.AccessNode) and e.src.data in inputs:
                cnode, cconn, cmemlet = inputs[e.src.data]
                # Connect to source node instead
                newmemlet = self._modify_memlet(e.data, cmemlet)
                graph.add_edge(cnode, cconn, e.dst, e.dst_conn, newmemlet)
                to_reconnect.remove(e.dst)
            elif isinstance(e.dst, nodes.AccessNode) and e.dst.data in outputs:
                cnode, cconn, cmemlet = outputs[e.dst.data]
                newmemlet = self._modify_memlet(e.data, cmemlet)
                if state.out_edges(e.dst):
                    # Connector is written in a non-sink access node
                    graph.add_edge(e.src, e.src_conn, e.dst, e.dst_conn,
                                   newmemlet)
                    # Check if there is another sink-node for the connector.
                    n = next((x for x in order if x.label == e.dst.label),
                             None)
                    if not state.out_edges(n):
                        continue
                    else:
                        # Connector is ONLY written in a non-sink access node,
                        # through the exit node to the true output access node.
                        e._src = e._dst
                        e._src_conn = e._dst_conn
                        # Remove wcr
                        newmemlet = dc(newmemlet)
                        newmemlet.wcr = None
                        newmemlet.other_subset = dc(newmemlet.subset)
                        for _, _, dst, _, memlet in graph.out_edges(cnode):
                            if isinstance(dst, nodes.AccessNode
                                          ) and memlet.data == cmemlet.data:
                                memlet.wcr = None
                # Connect to destination node instead
                graph.add_edge(e.src, e.src_conn, cnode, cconn, newmemlet)
                to_reconnect.remove(e.src)
            elif e.data.data in torename:
                if e.data.data in inputs:
                    newmemlet = self._modify_memlet(e.data,
                                                    inputs[e.data.data][2])
                elif e.data.data in outputs:
                    newmemlet = self._modify_memlet(e.data,
                                                    outputs[e.data.data][2])
                else:
                    # Rename data
                    cdata = torename[e.data.data]
                    newmemlet = dc(e.data)
                    newmemlet.data = cdata

                graph.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, newmemlet)
            else:
                # Do nothing
                graph.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, e.data)

        # Rename all access nodes
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in torename:
                node.data = torename[node.data]

        # If in scope, reconnect all source and sink nodes with empty memlets
        scope_node = graph.scope_dict()[nsdfg_node]
        if scope_node is not None:
            scope_exit = graph.exit_nodes(scope_node)[0]
            for node in state.source_nodes():
                if node in to_reconnect:
                    graph.add_edge(scope_node, None, node, None, EmptyMemlet())
                    to_reconnect.remove(node)
            for node in state.sink_nodes():
                if node in to_reconnect:
                    graph.add_edge(node, None, scope_exit, None, EmptyMemlet())
                    to_reconnect.remove(node)

        # Remove the nested SDFG node
        graph.remove_node(nsdfg_node)

        # Remove input/output nodes from top-level graph if not connected to
        # any internal node
        for node, _, _ in list(inputs.values()) + list(outputs.values()):
            if len(graph.all_edges(node)) == 0:
                graph.remove_node(node)


pattern_matching.Transformation.register_stateflow_pattern(NestSDFG)
pattern_matching.Transformation.register_pattern(InlineSDFG)
