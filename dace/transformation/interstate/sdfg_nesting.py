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

            for node in nxutil.find_source_nodes(state):
                if (isinstance(node, nodes.AccessNode) and
                        not node.desc(nested_sdfg).transient):
                    arrname = node.data
                    if arrname not in inputs:
                        arrobj = nested_sdfg.arrays[arrname]
                        nested_sdfg.arrays[arrname + '_in'] = arrobj
                        outer_sdfg.arrays[arrname] = dc(arrobj)
                        inputs[arrname] = arrname + '_in'
                    node.data = arrname + '_in'

            for node in nxutil.find_sink_nodes(state):
                if (isinstance(node, nodes.AccessNode) and
                        not node.desc(nested_sdfg).transient):
                    arrname = node.data
                    if arrname not in outputs:
                        arrobj = nested_sdfg.arrays[arrname]
                        nested_sdfg.arrays[arrname + '_out'] = arrobj
                        if arrname not in inputs:
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                        outputs[arrname] = arrname + '_out'

                        # TODO: Is this needed any longer ?
                        # # WCR Fix
                        # if self.promote_global_trans:
                        #     for edge in state.in_edges(node):
                        #         if state.memlet_path(edge)[0].data.wcr:
                        #             if node.data not in input_data:
                        #                 input_orig.update({
                        #                     node.data + '_in':
                        #                     node.data
                        #                 })
                        #                 input_nodes.update({
                        #                     node.data + '_in':
                        #                     dc(node)
                        #                 })
                        #                 new_data = dc(node.desc(sdfg))
                        #                 sdfg.arrays.update({
                        #                     node.data + '_in':
                        #                     new_data
                        #                 })
                        #                 input_data.add(node.data + '_in')
                        #             break

                    node.data = arrname + '_out'

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
            outer_sdfg.arrays[oldarrname].transient = True  # TO BE CHECKED: stop promoting?
        outputs.update(transients)

        for state in nested_sdfg.nodes():
            for _, edge in enumerate(state.edges()):
                _, _, _, _, mem = edge
                src = state.memlet_path(edge)[0].src
                dst = state.memlet_path(edge)[-1].dst
                if isinstance(src, nodes.AccessNode):
                    if (mem.data in inputs.keys() and
                            src.data == inputs[mem.data]):
                        mem.data = inputs[mem.data]
                    elif (mem.data in outputs.keys() and
                            src.data == outputs[mem.data]):
                        mem.data = outputs[mem.data]
                elif (isinstance(dst, nodes.AccessNode) and
                        mem.data in outputs.keys() and
                        dst.data == outputs[mem.data]):
                    mem.data = outputs[mem.data]

        outer_state = outer_sdfg.add_state(outer_sdfg.label)

        nested_node = outer_state.add_nested_sdfg(
            nested_sdfg, outer_sdfg, inputs.values(), outputs.values()
        )
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
            @param internal_memlet: The internal memlet (inside nested SDFG)
                                    before modification.
            @param internal_memlet: The external memlet before modification.
            @return: Offset Memlet to set on the resulting graph.
        """
        result = dc(internal_memlet)
        result.data = external_memlet.data

        shape = external_memlet.subset.size()
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

        torename = {}
        torename.update({k: v[2].data for k, v in inputs.items()})
        torename.update({k: v[2].data for k, v in outputs.items()})
        entry_connectors = set()

        # Add SDFG nodes to top-level SDFG
        state = nsdfg.nodes()[0]
        for node in state.nodes():
            # Data access nodes
            if isinstance(node, nodes.AccessNode):
                # External node
                if node.data in inputs or node.data in outputs:
                    for _, _, dst, dst_conn, _ in state.out_edges(node):
                        # Custom entry connector case
                        if (isinstance(dst, nodes.EntryNode)
                                and dst_conn[0:3] != 'IN_'):
                            entry_connectors.add(node.data)
                            sdfg.arrays[node.data] = nsdfg.arrays[node.data]
                            sdfg.arrays[node.data].transient = True
                            graph.add_node(node)
                            torename.pop(node.data)
                            break
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

            graph.add_node(node)

        # TODO: Confirm that the following is always correct
        # Add Scalars of the nested SDFG to the parent
        for name, arr in nsdfg.arrays.items():
            if isinstance(arr, dt.Scalar) and name not in sdfg.arrays:
                sdfg.arrays[name] = arr

        # Reconnect edges to their original source
        for e in state.edges():
            if isinstance(e.src, nodes.AccessNode) and e.src.data in inputs:
                cnode, cconn, cmemlet = inputs[e.src.data]
                if e.src.data in entry_connectors:
                    graph.add_edge(cnode, cconn, e.src, None, cmemlet)
                    graph.add_edge(e.src, None, e.dst, e.dst_conn, e.data)
                else:
                    # Connect to source node instead
                    newmemlet = self._modify_memlet(e.data, cmemlet)
                    graph.add_edge(cnode, cconn, e.dst, e.dst_conn, newmemlet)
            elif isinstance(e.dst, nodes.AccessNode) and e.dst.data in outputs:
                cnode, cconn, cmemlet = outputs[e.dst.data]
                # Connect to destination node instead
                newmemlet = self._modify_memlet(e.data, cmemlet)
                graph.add_edge(e.src, e.src_conn, cnode, cconn, newmemlet)
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

        # If an empty memlet was connected to the nested SDFG, reconnect
        # all source nodes with empty memlets
        if None in inputs:
            cnode, cconn, cmemlet = inputs[None]
            for node in state.source_nodes():
                graph.add_edge(cnode, cconn, node, None, EmptyMemlet())

        # Remove the nested SDFG node
        graph.remove_node(nsdfg_node)

        # Remove input/output nodes from top-level graph if not connected to
        # any internal node
        for node, _, _ in list(inputs.values()) + list(outputs.values()):
            if len(graph.all_edges(node)) == 0:
                graph.remove_node(node)

        # TODO: We may want to re-propagate memlets here


pattern_matching.Transformation.register_stateflow_pattern(NestSDFG)
pattern_matching.Transformation.register_pattern(InlineSDFG)
