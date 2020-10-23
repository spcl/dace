import networkx as nx

from dace import dtypes, registry, sdfg
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.sdfg import sdfg as dace_sdfg
import itertools


@registry.autoregister_params()
class StateReordering(transformation.Transformation):
    """ Swaps two sequential sdfg_states if there are no data dependencies between them
    """

    _first_state = sdfg.SDFGState()
    _second_state = sdfg.SDFGState()

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(StateReordering._first_state,
                                   StateReordering._second_state)
        ]

    @staticmethod
    def can_be_applied(sdfg: dace_sdfg.SDFG,
                       candidate,
                       expr_index,
                       _the_same_sdfg: dace_sdfg.SDFG,
                       strict=False):
        first_state = sdfg.nodes()[candidate[StateReordering._first_state]]
        second_state = sdfg.nodes()[candidate[StateReordering._second_state]]

        first_in_edges = sdfg.in_edges(first_state)
        second_in_edges = sdfg.in_edges(second_state)
        first_out_edges = sdfg.out_edges(first_state)
        second_out_edges = sdfg.out_edges(second_state)

        # states should be connected with single edge

        if len(first_out_edges) != 1:
            return False
        if len(second_in_edges) != 1:
            return False

        edge: dace_sdfg.InterstateEdge = first_out_edges[0]

        # state transition should have no conditions and assignments
        if not edge.data.is_unconditional():
            return False

        if edge.data.assignments:
            return False

        # Find source/sink (data) nodes
        first_input = first_state.source_nodes()
        first_output = first_state.sink_nodes()
        second_input = second_state.source_nodes()
        second_output = second_state.sink_nodes()

        # data dependencies
        for first_node in itertools.chain(
                first_input,
                first_output):  # consider both RW and WW dependencies
            for second_node in second_output:
                if first_node.label == second_node.label:
                    return False  # dependency found

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_state = graph.nodes()[candidate[StateReordering._first_state]]
        second_state = graph.nodes()[candidate[StateReordering._second_state]]

        return " -> ".join(state.label for state in [first_state, second_state])

    def apply(self, sdfg: dace_sdfg.SDFG):
        first_state = sdfg.nodes()[self.subgraph[StateReordering._first_state]]
        second_state = sdfg.nodes()[self.subgraph[
            StateReordering._second_state]]

        first_in_edges = sdfg.in_edges(first_state)
        second_in_edges = sdfg.in_edges(second_state)
        first_out_edges = sdfg.out_edges(first_state)
        second_out_edges = sdfg.out_edges(second_state)

        edge: dace_sdfg.InterstateEdge = first_out_edges[0]

        for in_edge in first_in_edges:
            sdfg.add_edge(in_edge.src, second_state, in_edge.data)

        for in_edge in first_in_edges:
            sdfg.remove_edge(in_edge)

        for out_edge in second_out_edges:
            sdfg.add_edge(first_state, out_edge.dst, out_edge.data)

        for out_edge in second_out_edges:
            sdfg.remove_edge(out_edge)

        sdfg.add_edge(second_state, first_state, edge.data)
        sdfg.remove_edge(edge)

        # a hack that detects the case when first state is initial state
        if sdfg.start_state == first_state:
            sdfg.start_state = sdfg.node_id(second_state)
