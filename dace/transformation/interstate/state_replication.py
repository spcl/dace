# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" State replication transformation """

from dace import data as dt, sdfg as sd
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from copy import deepcopy
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.properties import make_properties

@make_properties
class StateReplication(transformation.MultiStateTransformation):
    """
    Creates a copy of a state for each of its incoming edge. Then, redirects every edge to a different copy.
    This results in states with only one incoming edge.
    """

    target_state = transformation.PatternNode(sd.SDFGState)

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.target_state)]


    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        target_state: SDFGState = self.target_state

        out_edges = graph.out_edges(target_state)
        in_edges = graph.in_edges(target_state)

        if len(in_edges) < 2:
            return False

        # avoid useless replications
        if target_state.is_empty() and len(out_edges) < 2:
            return False

        # make sure this is not a loop guard
        if len(out_edges) == 2:
            detect = DetectLoop()
            detect.loop_guard = target_state
            detect.loop_begin = out_edges[0].dst
            detect.exit_state = out_edges[1].dst
            if detect.can_be_applied(graph, 0, sdfg):
                return False
            detect.exit_state = out_edges[0].dst
            detect.loop_begin = out_edges[1].dst
            if detect.can_be_applied(graph, 0, sdfg):
                return False
        
        return True
    
    def apply(self, _, sdfg: sd.SDFG):
        target_state: SDFGState = self.target_state

        if len(sdfg.out_edges(target_state)) == 0:
            sdfg.add_state_after(target_state)

        state_names = set(s.label for s in sdfg.nodes())

        root_blueprint = target_state.to_json()
        for e in sdfg.in_edges(target_state)[1:]:
            state_copy = sd.SDFGState.from_json(root_blueprint, context={'sdfg': sdfg})
            state_copy.label = dt.find_new_name(state_copy.label, state_names)
            state_names.add(state_copy.label)
            sdfg.add_node(state_copy)

            sdfg.remove_edge(e)
            sdfg.add_edge(e.src, state_copy, e.data)

            # connect out edges
            for oe in sdfg.out_edges(target_state):
                sdfg.add_edge(state_copy, oe.dst, deepcopy(oe.data))
