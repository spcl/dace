# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from dace import registry
from dace import properties
from dace.transformation import transformation
from dace.sdfg import nodes as dace_nodes
from dace.sdfg import graph as dace_graph
from dace.sdfg import utils
from dace.sdfg import sdfg as dace_sdfg
from dace.sdfg import state as dace_state

from typing import Dict, List
import itertools


def add_nsdfg_array_prefix(parent_sdfg_state: dace_state.SDFGState,
                           nested_sdfg: dace_nodes.NestedSDFG, prefix: str):
    """ Prepends prefix to all array, connector, and memlet names of given nested_sdfg. Operates inplace. """

    # process arrays
    array_names = list(nested_sdfg.sdfg.arrays.keys())
    for name in array_names:
        new_name = prefix + name
        nested_sdfg.sdfg.replace(name, prefix + name)

    # process connectors
    def process_connectors(connectors):
        new_connectors = {prefix + name: value for name, value in connectors.items()}
        connectors.clear()
        connectors.update(new_connectors)

    process_connectors(nested_sdfg.in_connectors)
    process_connectors(nested_sdfg.out_connectors)

    # process external edges

    for edge in parent_sdfg_state.in_edges(nested_sdfg):
        edge.dst_conn = prefix + edge.dst_conn

    for edge in parent_sdfg_state.out_edges(nested_sdfg):
        edge.src_conn = prefix + edge.src_conn


def get_all_node_paths(
        sdfg_state: dace_state.SDFGState, src_node: dace_nodes.Node,
        dst_node: dace_nodes.Node) -> List[List[dace_nodes.Node]]:

    result: List[List[dace_nodes.Node]] = []

    src_out_edges = sdfg_state.out_edges(src_node)

    for edge in src_out_edges:
        if edge.dst != dst_node:
            tail_paths: List[List[dace_nodes.Node]] = get_all_node_paths(
                sdfg_state, edge.dst, dst_node)
            for tail_path in tail_paths:
                result.append([src_node] + tail_path)
        else:
            result.append([src_node, dst_node])

    return result


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class NestedSDFGFusion(transformation.Transformation):
    """ Fused a pair of nested SDFGs that are connected with a single AccessNode: NestedSDFG->AccessNode->NestedSDFG.
    """

    _nested_sdfg1 = dace_nodes.NestedSDFG(label="",
                                          sdfg=dace_graph.OrderedDiGraph(),
                                          inputs=set(),
                                          outputs=set())
    _access_node = dace_nodes.AccessNode('_')
    _nested_sdfg2 = dace_nodes.NestedSDFG(label="",
                                          sdfg=dace_graph.OrderedDiGraph(),
                                          inputs=set(),
                                          outputs=set())

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(NestedSDFGFusion._nested_sdfg1,
                                  NestedSDFGFusion._access_node,
                                  NestedSDFGFusion._nested_sdfg2)
        ]

    @staticmethod
    def can_be_applied(sdfg_state: dace_state.SDFGState,
                       candidate: Dict[dace_nodes.Node, int],
                       expr_index: int,
                       sdfg: dace_sdfg.SDFG,
                       strict=False):
        nested_sdfg1 = sdfg_state.nodes()[candidate[
            NestedSDFGFusion._nested_sdfg1]]
        access_node = sdfg_state.nodes()[candidate[
            NestedSDFGFusion._access_node]]
        nested_sdfg2 = sdfg_state.nodes()[candidate[
            NestedSDFGFusion._nested_sdfg2]]

        # check that all paths from nested_sdfg1 to nested_sdfg2 have only a single AccessNode

        node_paths = get_all_node_paths(sdfg_state, nested_sdfg1, nested_sdfg2)
        for path in node_paths:
            # check than path has 3 nodes NestedSDFG, AccessNode, NestedSDFG
            if len(path) != 3:
                return False

            # check that intermediate node is AccessNode
            if not isinstance(path[1], dace_nodes.AccessNode):
                return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):
        sdfg_state = sdfg.nodes()[self.state_id]

        nested_sdfg1: dace_nodes.NestedSDFG = sdfg_state.nodes()[self.subgraph[
            NestedSDFGFusion._nested_sdfg1]]
        access_node = sdfg_state.nodes()[self.subgraph[
            NestedSDFGFusion._access_node]]
        nested_sdfg2: dace_nodes.NestedSDFG = sdfg_state.nodes()[self.subgraph[
            NestedSDFGFusion._nested_sdfg2]]

        # make names of arrays and connectors unique for both nested sdfg

        add_nsdfg_array_prefix(sdfg_state, nested_sdfg1, 'n1_')
        add_nsdfg_array_prefix(sdfg_state, nested_sdfg2, 'n2_')

        # for each connector collect list of memlets

        in_edges1 = sdfg_state.in_edges(nested_sdfg1)
        in_edges2 = sdfg_state.in_edges(nested_sdfg2)
        out_edges1 = sdfg_state.out_edges(nested_sdfg1)
        out_edges2 = sdfg_state.out_edges(nested_sdfg2)

        # split nested sdfgs by creating additional copies of access nodes between them

        intermediate_nodes = {e.dst for e in out_edges1}.intersection({e.src for e in in_edges2})
        for n in intermediate_nodes:
            node_copy = sdfg_state.add_access(n.data)
            # use original copy as output and new node as input
            original_input_edges = list(sdfg_state.edges_between(n, nested_sdfg2))
            for e in original_input_edges:
                edge_copy = sdfg_state.add_edge(node_copy, None, nested_sdfg2, e.dst_conn, e.data)
                sdfg_state.remove_edge(e)
                in_edges2.remove(e)
                in_edges2.append(edge_copy)

        # build lists of edges after splitting nested sdfgs

        new_out_edges = out_edges1 + out_edges2
        new_in_edges = in_edges1 + in_edges2

        # create new sdfg that replaces existing ones

        new_nested_sdfg = dace_sdfg.SDFG(name='name')

        # move existing states with state transition edge into the new sdfg

        for node in itertools.chain(nested_sdfg1.sdfg.nodes(),
                                    nested_sdfg2.sdfg.nodes()):
            if node in nested_sdfg2.sdfg.nodes():
                prefix = 'nested1_'
            else:
                prefix = 'nested2_'
            node._label = prefix + node._label  # is there a better way?
            node.parent = new_nested_sdfg

            # ugly fix of nested sdfg field that points to parent
            for n in node.nodes():
                if isinstance(n, dace_nodes.NestedSDFG):
                    s: dace_sdfg.SDFG = n.sdfg
                    s.parent_sdfg = new_nested_sdfg

            new_nested_sdfg.add_node(node)

        for edge in itertools.chain(nested_sdfg1.sdfg.edges(),
                                    nested_sdfg2.sdfg.edges()):
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
        sdfg1_exit_states = [
            n for n in nested_sdfg1.sdfg.nodes()
            if not nested_sdfg1.sdfg.out_edges(n)
        ]

        # create edges from sdfg1 exit states to sdfg2 initial state
        start2_node = nested_sdfg2.sdfg.start_state
        for exit1_node in sdfg1_exit_states:
            new_nested_sdfg.add_edge(exit1_node, start2_node,
                                     dace_sdfg.InterstateEdge())

        # create new NestedSDFG that will replace existing states
        new_nested_sdfg_node: dace_nodes.NestedSDFG = sdfg_state.add_nested_sdfg(
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
            sdfg_state.add_edge(src, edge.src_conn, dst, edge.dst_conn,
                                edge.data)

        # remove old edges
        for edge in itertools.chain(in_edges1, in_edges2, out_edges1,
                                    out_edges2):
            sdfg_state.remove_edge(edge)

        # remove old nested sdfg

        sdfg_state.remove_node(nested_sdfg1)
        sdfg_state.remove_node(nested_sdfg2)