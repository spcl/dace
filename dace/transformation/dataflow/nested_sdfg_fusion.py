# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from dace import registry
from dace.properties import make_properties
from dace.transformation.transformation import Transformation
from dace.sdfg.nodes import NestedSDFG, Tasklet, Node, AccessNode
from dace.sdfg.graph import OrderedDiGraph, MultiConnectorEdge
from dace.sdfg.utils import node_path_graph
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.sdfg import InterstateEdge
from dace import Memlet
from dace.subsets import Range

from typing import Dict
import itertools

def add_nsdfg_array_prefix(parent_sdfg_state: SDFGState, nested_sdfg: NestedSDFG, prefix: str):
    """ Prepends prefix to all array, connector, and memlet names of given nested_sdfg. Operates inplace. """

    def process_connectors_and_arrays(connectors):
        new_in_connectors = {}
        for name, dtype in connectors.items():
            new_name = prefix + name
            nested_sdfg.sdfg.replace(name, prefix + name)
            new_in_connectors[new_name] = connectors[name]
        connectors = new_in_connectors

    process_connectors_and_arrays(nested_sdfg.in_connectors)
    process_connectors_and_arrays(nested_sdfg.out_connectors)

    for edge in parent_sdfg_state.in_edges(nested_sdfg):
        edge.dst_conn = prefix + edge.dst_conn

    for edge in parent_sdfg_state.out_edges(nested_sdfg):
        edge.src_conn = prefix + edge.src_conn



@registry.autoregister_params(singlestate=True)
@make_properties
class NestedSDFGFusion(Transformation):
    """ Fused a pair of nested SDFGs that are connected with a single AccessNode: NestedSDFG->AccessNode->NestedSDFG.
    """


    _nested_sdfg1 = NestedSDFG(label="", sdfg=OrderedDiGraph(), inputs=set(), outputs=set())
    _access_node = Tasklet('_')
    _nested_sdfg2 = NestedSDFG(label="", sdfg=OrderedDiGraph(), inputs=set(), outputs=set())

    @staticmethod
    def expressions():
        return [
            node_path_graph(
                NestedSDFGFusion._nested_sdfg1,
                NestedSDFGFusion._access_node,
                NestedSDFGFusion._nested_sdfg2
            )
        ]

    @staticmethod
    def can_be_applied(sdfg_state: SDFGState, candidate: Dict[Node, int], expr_index: int, sdfg: SDFG, strict=False):
        nested_sdfg1 = sdfg_state.nodes()[candidate[NestedSDFGFusion._nested_sdfg1]]
        access_node = sdfg_state.nodes()[candidate[NestedSDFGFusion._access_node]]
        nested_sdfg2 = sdfg_state.nodes()[candidate[NestedSDFGFusion._nested_sdfg2]]

        # TODO: check that between nested_sdfg1 and nested_sdfg2 only transient accessnodes are present

        return True

    @staticmethod
    def match_to_str(sdfg_state: SDFGState, candidate: Dict[Node, int]):

        nested_sdfg1 = sdfg_state.nodes()[candidate[NestedSDFGFusion._nested_sdfg1]]
        access_node = sdfg_state.nodes()[candidate[NestedSDFGFusion._access_node]]
        nested_sdfg2 = sdfg_state.nodes()[candidate[NestedSDFGFusion._nested_sdfg2]]

        return nested_sdfg1 + '->' + access_node + '->' + nested_sdfg2

    def apply(self, sdfg: SDFG):
        sdfg_state = sdfg.nodes()[self.state_id]

        nested_sdfg1: NestedSDFG = sdfg_state.nodes()[self.subgraph[NestedSDFGFusion._nested_sdfg1]]
        access_node = sdfg_state.nodes()[self.subgraph[NestedSDFGFusion._access_node]]
        nested_sdfg2: NestedSDFG = sdfg_state.nodes()[self.subgraph[NestedSDFGFusion._nested_sdfg2]]

        # make names of arrays and connectors unique for both nested sdfg

        add_nsdfg_array_prefix(sdfg_state, nested_sdfg1, 'n1_')
        add_nsdfg_array_prefix(sdfg_state, nested_sdfg2, 'n2_')

        # for each connector collect list of memlets

        in_edges1 = sdfg_state.in_edges(nested_sdfg1)
        in_edges2 = sdfg_state.in_edges(nested_sdfg2)
        out_edges1 = sdfg_state.out_edges(nested_sdfg1)
        out_edges2 = sdfg_state.out_edges(nested_sdfg2)

        new_out_edges = out_edges1 + out_edges2

        # separate in_edges2 into two groups
        # - edges that go from outside should be inputs to new nested sdfg
        # - all other edges should be removed

        new_in_edges = []

        # if some edge is input to the second nested sdfg, then it should be input to the new sdfg only
        # when it is not output of the first nested sdfg
        for in_edge2 in in_edges2:
            if not isinstance(in_edge2.src, AccessNode):
                new_in_edges.append(in_edge2)
                continue

            temp_node: AccessNode = in_edge2.src

            node_in_edges = sdfg_state.in_edges(temp_node)
            if len(node_in_edges) != 1:
                raise Exception("Not implemented")
            node_in_edge: MultiConnectorEdge = node_in_edges[0]

            if node_in_edge not in out_edges1:

                new_in_edges.append(in_edge2)
                continue

            # If we are here, then in_edge2 originates from output of the first SDFG.
            # We need to change array name that is used in the second SDFG to refer to the array from the first SDFG.

            nested_sdfg2.sdfg.replace(in_edge2.dst_conn, node_in_edge.src_conn)

        new_in_edges += in_edges1

        # create new sdfg that replaces existing ones

        new_nested_sdfg = SDFG(name='name')

        # move existing states with state transition edge into the new sdfg

        for node in itertools.chain(nested_sdfg1.sdfg.nodes(), nested_sdfg2.sdfg.nodes()):
            if node in nested_sdfg2.sdfg.nodes():
                prefix = 'nested1_'
            else:
                prefix = 'nested2_'
            node._label = prefix + node._label # is there a better way?
            node.parent = new_nested_sdfg
            new_nested_sdfg.add_node(node)

        for edge in itertools.chain(nested_sdfg1.sdfg.edges(), nested_sdfg2.sdfg.edges()):
            new_nested_sdfg.add_edge(edge.src, edge.dst, edge.data)

        # copy arrays
        for name, array in nested_sdfg1.sdfg.arrays.items():
            new_nested_sdfg.add_datadesc(name, array)

        for name, array in nested_sdfg2.sdfg.arrays.items():
            # it is possible that we reuse array name from the first nested sdfg, skip them since they are already added
            if name in new_nested_sdfg.arrays:
                continue

            new_nested_sdfg.add_datadesc(name, array)

        # detect sdfg1 exit states (states without outgoing edges)
        sdfg1_exit_states = [n for n in nested_sdfg1.sdfg.nodes() if not nested_sdfg1.sdfg.out_edges(n)]

        # create edges from sdfg1 exit states to sdfg2 initial state
        start2_node = nested_sdfg2.sdfg.start_state
        for exit1_node in sdfg1_exit_states:
            new_nested_sdfg.add_edge(exit1_node, start2_node, InterstateEdge())

        # create new NestedSDFG that will replace existing states
        new_nested_sdfg_node: NestedSDFG = sdfg_state.add_nested_sdfg(
            sdfg=new_nested_sdfg,
            parent="unused",
            inputs=set([e.dst_conn for e in new_in_edges]),
            outputs=set([e.src_conn for e in new_out_edges]),
        )

        # reconnect memlets to new nested sdfg

        for edge in itertools.chain(new_in_edges, new_out_edges):
            if edge in new_in_edges:
                src = edge.src
                dst = new_nested_sdfg_node
            else:
                src = new_nested_sdfg_node
                dst = edge.dst
            sdfg_state.add_edge(src, edge.src_conn, dst, edge.dst_conn, edge.data)

        # remove old edges
        for edge in itertools.chain(in_edges1, in_edges2, out_edges1, out_edges2):
            sdfg_state.remove_edge(edge)

        # remove old nested sdfg

        sdfg_state.remove_node(nested_sdfg1)
        sdfg_state.remove_node(nested_sdfg2)