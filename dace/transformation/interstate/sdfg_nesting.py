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

            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode)
                        and not node.desc(nested_sdfg).transient):
                    if (state.out_degree(node) > 0):      # input node
                        print("Node: " +str(node) + " is input node")
                        arrname = node.data
                        if arrname not in inputs:
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays[arrname + '_in'] = arrobj
                            outer_sdfg.arrays[arrname] = dc(arrobj)
                            inputs[arrname] = arrname + '_in'
                        node_data_name = arrname + '_in'
                        print("State: " + state.name + " Added node " + arrname + "_in")
                    if (state.in_degree(node) > 0): # output node
                        arrname = node.data
                        if arrname not in outputs:
                            print("Node: " + str(node) + " is output node")
                            arrobj = nested_sdfg.arrays[arrname]
                            nested_sdfg.arrays[arrname + '_out'] = arrobj
                            if arrname not in inputs:
                                outer_sdfg.arrays[arrname] = dc(arrobj)
                            outputs[arrname] = arrname + '_out'
                            print("State: " + state.name + " Added node " + arrname + "_out")
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
            outer_sdfg.arrays[oldarrname].transient = True
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

            # for _, edge in enumerate(state.edges()):
            #     _, _, _, _, mem = edge
            #     src = state.memlet_path(edge)[0].src
            #     dst = state.memlet_path(edge)[-1].dst
            #     if isinstance(src, nodes.AccessNode):
            #         if (mem.data in inputs.keys() and
            #                 src.data == inputs[mem.data]):
            #             mem.data = inputs[mem.data]
            #         elif mem.data in outputs.keys():
            #             src.data = outputs[mem.data]
            #             mem.data = outputs[mem.data]
            #     elif isinstance(dst, nodes.AccessNode):
            #         if mem.data in outputs.keys():
            #             if dst.data != outputs[mem.data]:
            #                 dst.data = outputs[mem.data]
            #             mem.data = outputs[mem.data]
            #         elif mem.data in inputs.keys():
            #             dst.data = inputs[mem.data]
            #             mem.data = inputs[mem.data]



            for node in nxutil.find_source_nodes(state):
                print("Source node: " + str(node))
            for node in nxutil.find_sink_nodes(state):
                print("Sink node: " + str(node))

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
