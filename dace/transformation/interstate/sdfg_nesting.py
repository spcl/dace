""" SDFG nesting transformation. """

from copy import deepcopy as dc
import networkx as nx

import dace
from dace import data as dt, memlet, sdfg as sd, subsets, symbolic
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


pattern_matching.Transformation.register_stateflow_pattern(NestSDFG)
