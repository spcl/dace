# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" State replication transformation """

from copy import deepcopy

from dace import SDFG, data
from dace.properties import make_properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState, ControlFlowRegion
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import DetectLoop


@make_properties
class StateReplication(transformation.MultiStateTransformation):
    """
    Creates a copy of a state for each of its incoming edge. Then, redirects every edge to a different copy.
    This results in states with only one incoming edge.
    """

    target_state = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.target_state)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False):
        in_edges, out_edges = graph.in_edges(self.target_state), graph.out_edges(self.target_state)
        if len(in_edges) < 2:
            # If it has only one incoming edge, then there is nothing to replicate.
            return False
        if self.target_state.is_empty() and len(out_edges) < 2:
            # No point replicating an empty state that does not branch out again.
            return False

        # Make sure this is not a loop guard. Application on loops results in the addition of useless states (but the
        # SDFG is still correct). This will not get rid of the loop, meaning that apply_transformations_repeated will
        # never halt.
        if len(out_edges) == 2:
            detect = DetectLoop()
            detect.loop_guard = self.target_state
            detect.loop_begin = out_edges[0].dst
            detect.exit_state = out_edges[1].dst
            if detect.can_be_applied(graph, 0, sdfg):
                return False
            detect.exit_state = out_edges[0].dst
            detect.loop_begin = out_edges[1].dst
            if detect.can_be_applied(graph, 0, sdfg):
                return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        state = self.target_state
        blueprint = state.to_json()

        in_edges, out_edges = sdfg.in_edges(state), sdfg.out_edges(state)
        if not out_edges:
            # If this was a sink state, then create an extra sink state to synchronize on.
            sdfg.add_state_after(state)

        state_names = set(s.label for s in sdfg.nodes())
        for e in in_edges[1:]:
            state_copy = SDFGState.from_json(blueprint, context={'sdfg': sdfg})
            state_copy.label = data.find_new_name(state_copy.label, state_names)
            state_names.add(state_copy.label)
            sdfg.add_node(state_copy)

            # Replace the `e.src -> state` edge with an `e.src -> state_copy` edge.
            sdfg.remove_edge(e)
            sdfg.add_edge(e.src, state_copy, e.data)

            # Replicate the outgoing edges of `state` to `state_copy` too.
            for oe in sdfg.out_edges(state):
                sdfg.add_edge(state_copy, oe.dst, deepcopy(oe.data))
