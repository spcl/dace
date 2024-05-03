import functools
from collections import deque
from typing import Dict

import dace
from dace import registry, properties, subsets
from dace.sdfg import nodes, utils as sdfg_utils
from dace.transformation import transformation as xf
from dace import Config
import dace.libraries.onnx as donnx
from dace.util import in_edge_with_name


def expand_library_nodes_except_reshape(self, recursive=True):
    states = list(self.states())
    while len(states) > 0:
        state = states.pop()
        expanded_something = False
        for node in list(state.nodes()):  # Make sure we have a copy
            if isinstance(node, nodes.NestedSDFG):
                node.sdfg.expand_library_nodes()  # Call recursively
            elif isinstance(node, nodes.LibraryNode) and not isinstance(
                    node, donnx.ONNXReshape):
                impl_name = node.expand(self, state)
                if Config.get_bool("debugprint"):
                    print(
                        "Automatically expanded library node \"{}\" with implementation \"{}\"."
                        .format(str(node), impl_name))
                # We made a copy of the original list of nodes, so we keep
                # iterating even though this list has now changed
                if recursive:
                    expanded_something = True
        if expanded_something:
            states.append(state)  # Nodes have changed. Check state again


@properties.make_properties
class ReshapeElimination(xf.SingleStateTransformation):
    """ Merge a reshape into a preceding or following nested SDFG call.
    """
    # pattern matching only checks that the type of the node matches,
    reshape_node = xf.PatternNode(donnx.ONNXReshape)
    access_node = xf.PatternNode(nodes.AccessNode)
    nsdfg = xf.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [
            sdfg_utils.node_path_graph(cls.reshape_node, cls.access_node,
                                       cls.nsdfg)
        ]

    def can_be_applied(self,
                       graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       expr_index: int,
                       sdfg,
                       permissive: bool = False):

        graph: dace.SDFGState
        reshape_node = self.reshape_node
        access_node = self.access_node

        if not sdfg.arrays[access_node.data].transient:
            return False

        in_memlet = in_edge_with_name(reshape_node, graph, "data").data

        def is_memlet_contiguous(mm):
            if (not isinstance(mm.subset, subsets.Range)
                    or any([step != 1 for _, _, step in mm.subset])):
                return False
            return True

        # check that the in memlets is contiguous (this check can be relaxed)
        for mm in [in_memlet] + [e.data for e in graph.out_edges(access_node)]:
            if not is_memlet_contiguous(mm):
                return False

        def _prod(sequence):
            return functools.reduce(lambda a, b: a * b, sequence, 1)

        # check that the in arrays are contiguous
        def is_desc_contiguous(desc):
            expected_strides = [
                _prod(desc.shape[i + 1:]) for i in range(len(desc.shape))
            ]
            return all(es == s
                       for es, s in zip(expected_strides, desc.strides))

        for desc in [
                sdfg.arrays[in_memlet.data], sdfg.arrays[access_node.data]
        ]:
            if not is_desc_contiguous(desc):
                return False

        return True

    @classmethod
    def match_to_str(cls, graph, candidate):
        node = graph.nodes()[candidate[cls.reshape_node]]
        return "Eliminate {}".format(node)

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG):
        # Extract the subgraph, execute it and insert an AccessNode to the result
        reshape_node = self.reshape_node
        access_node = self.access_node
        nsdfg_node = self.nsdfg

        old_edge_in = in_edge_with_name(reshape_node, state, "data")
        old_edge_in_shape = in_edge_with_name(reshape_node, state,
                                                    "shape")

        # delete the subgraph that computed shape
        queue = deque([old_edge_in_shape.src])
        while len(queue) > 0:
            current_node = queue.popleft()

            edges = state.in_edges(current_node)
            state.remove_node(current_node)
            for e in edges:
                next_node = e.src
                if len(state.out_edges(next_node)) == 0:
                    queue.append(next_node)

        # get the edges between the the access_node and the nsdfg_node
        old_edges = [
            e for e in state.out_edges(access_node) if e.dst == nsdfg_node
        ]

        for edge in old_edges:
            state.add_edge(old_edge_in.src, old_edge_in.src_conn, edge.dst,
                           edge.dst_conn, old_edge_in.data)
            state.remove_edge(edge)

        # remove the old node and output access node
        state.remove_node(reshape_node)

        if len(state.out_edges(access_node)) == 0:
            state.remove_node(access_node)
