""" SDFG nesting transformation. """

from copy import deepcopy as dc
import networkx as nx

import dace
from dace import data as dt, memlet, sdfg as sd, subsets, symbolic, Memlet
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
        # Copy SDFG to nested SDFG
        nested_sdfg = dace.SDFG('nested_' + sdfg.label)
        nested_sdfg.add_nodes_from(sdfg.nodes())
        for src, dst, data in sdfg.edges():
            nested_sdfg.add_edge(src, dst, data)

        input_orig = {}
        input_data = set()
        input_nodes = {}
        output_orig = {}
        output_data = set()
        output_nodes = {}
        for state in sdfg.nodes():
            for node in nxutil.find_source_nodes(state):
                if isinstance(
                        node,
                        nodes.AccessNode) and not node.desc(sdfg).transient:
                    if node.data not in input_data:
                        input_orig.update({node.data + '_in': node.data})
                        input_nodes.update({node.data + '_in': dc(node)})
                        new_data = dc(node.desc(sdfg))
                        input_data.add(node.data)
                        sdfg.arrays.update({node.data + '_in': new_data})
                    node.data = node.data + '_in'
            for node in nxutil.find_sink_nodes(state):
                if isinstance(
                        node,
                        nodes.AccessNode) and not node.desc(sdfg).transient:
                    if node.data not in output_data:
                        output_orig.update({node.data + '_out': node.data})
                        output_nodes.update({node.data + '_out': dc(node)})
                        new_data = dc(node.desc(sdfg))
                        output_data.add(node.data)
                        sdfg.arrays.update({node.data + '_out': new_data})

                        # WCR Fix
                        if self.promote_global_trans:
                            for edge in state.in_edges(node):
                                if sd._memlet_path(state, edge)[0].data.wcr:
                                    if node.data not in input_data:
                                        input_orig.update({
                                            node.data + '_in':
                                            node.data
                                        })
                                        input_nodes.update({
                                            node.data + '_in':
                                            dc(node)
                                        })
                                        new_data = dc(node.desc(sdfg))
                                        sdfg.arrays.update({
                                            node.data + '_in':
                                            new_data
                                        })
                                        input_data.add(node.data + '_in')
                                    break

                    node.data = node.data + '_out'
            if self.promote_global_trans:
                scope_dict = state.scope_dict()
                for node in state.nodes():
                    if (isinstance(node, nodes.AccessNode)
                            and node.desc(sdfg).transient
                            and not scope_dict[node]):
                        if node.data not in output_data:
                            output_orig.update({node.data + '_out': node.data})
                            output_nodes.update({node.data + '_out': dc(node)})
                            new_data = dc(node.desc(sdfg))
                            output_data.add(node.data + '_out')
                            sdfg.arrays.update({node.data + '_out': new_data})
                        node.data = node.data + '_out'
                        node.desc(sdfg).transient = False
            for _, edge in enumerate(state.edges()):
                _, _, _, _, mem = edge
                src = sd._memlet_path(state, edge)[0].src
                dst = sd._memlet_path(state, edge)[-1].dst
                if isinstance(src,
                              nodes.AccessNode) and src.data in input_data:
                    mem.data = src.data
                if isinstance(src,
                              nodes.AccessNode) and src.data in output_data:
                    mem.data = src.data
                if isinstance(dst,
                              nodes.AccessNode) and dst.data in output_data:
                    mem.data = dst.data

        sdfg.remove_nodes_from(sdfg.nodes())

        state = sdfg.add_state(sdfg.label)
        state.add_nodes_from(input_nodes.values())
        state.add_nodes_from(output_nodes.values())

        nested_node = state.add_nested_sdfg(nested_sdfg, sdfg,
                                            input_data.keys(),
                                            output_data.keys())
        for key, val in input_nodes.items():
            state.add_edge(
                val, None, nested_node, key,
                memlet.Memlet.simple(
                    val, str(subsets.Range.from_array(val.desc(sdfg)))))
        for key, val in output_nodes.items():
            state.add_edge(
                nested_node, key, val, None,
                memlet.Memlet.simple(
                    val, str(subsets.Range.from_array(val.desc(sdfg)))))


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

        # TODO: Remove this loop when subsets are applied below
        if strict:
            for e in graph.all_edges(nested_sdfg):
                if e.data.other_subset is not None:
                    return False
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        nsdfg_node = graph.nodes()[self.subgraph[InlineSDFG._nested_sdfg]]
        nsdfg = nsdfg_node.sdfg

        # Find original source/destination nodes
        inputs = {}
        outputs = {}
        for e in graph.in_edges(nsdfg_node):
            inputs[e.dst_conn] = (e.src, e.src_conn, e.data.data)
        for e in graph.out_edges(nsdfg_node):
            outputs[e.src_conn] = (e.dst, e.dst_conn, e.data.data)

        torename = {}
        torename.update({k: v[2] for k, v in inputs.items()})
        torename.update({k: v[2] for k, v in outputs.items()})

        # Add SDFG nodes to top-level SDFG
        state = nsdfg.nodes()[0]
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

            graph.add_node(node)

        # Reconnect edges to their original source
        # TODO: When copying memlets, apply subset on them first (if memlet to/from nested SDFG is a subset)
        for e in state.edges():
            if isinstance(e.src, nodes.AccessNode) and e.src.data in inputs:
                cnode, cconn, cdata = inputs[e.src.data]
                # Connect to source node instead
                newmemlet = dc(e.data)
                newmemlet.data = cdata
                graph.add_edge(cnode, cconn, e.dst, e.dst_conn, newmemlet)
            elif isinstance(e.dst, nodes.AccessNode) and e.dst.data in outputs:
                cnode, cconn, cdata = outputs[e.dst.data]
                # Connect to destination node instead
                newmemlet = dc(e.data)
                newmemlet.data = cdata
                graph.add_edge(e.src, e.src_conn, cnode, cconn, newmemlet)
            elif e.data.data in torename:
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
