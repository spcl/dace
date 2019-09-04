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
            outer_sdfg.arrays[oldarrname].transient = False
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


pattern_matching.Transformation.register_stateflow_pattern(NestSDFG)
