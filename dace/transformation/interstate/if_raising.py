# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace import SDFG, data
from dace.properties import make_properties
from dace.sdfg import InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState, ControlFlowRegion
from dace.transformation import transformation


@make_properties
class IfRaising(transformation.MultiStateTransformation):
    """
    Takes a non-empty If guard state whose branches do not rely on the interior of the guard state, and replicates the
    interior to new unconditional states _after_ the branch leaving the original guard states empty. So, an SDFG like:
            guard_state + interior
               /        \\
        state_1         state_2
              \\        /
             terminal_state
    will become:
           guard_state (empty)
              /        \\
        state_1a         state_2a
      (+ interior)    (+ interior)
            |               |
        state_1         state_2
              \\        /
             terminal_state
    """

    if_guard = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.if_guard)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False):
        if self.if_guard.is_empty():
            # The If state must be doing something to replicate in the first place.
            return False

        out_edges = graph.out_edges(self.if_guard)
        if len(out_edges) != 2:
            # Such an If state must have exactly two branching paths.
            return False

        if_symbols = set(s for e in out_edges for s in e.data.free_symbols)
        _, wset = self.if_guard.read_and_write_sets()
        if if_symbols.intersection(wset):
            # The symbols used on the branch must not be written in the state.
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        raised_if_guard = sdfg.add_state('raised_if_guard')
        sdutil.change_edge_dest(sdfg, self.if_guard, raised_if_guard)

        replica = SDFGState.from_json(self.if_guard.to_json(), context={'sdfg': sdfg})
        replica.label = data.find_new_name(replica.label, {s.label for s in sdfg.nodes()})
        sdfg.add_node(replica)

        # Move conditional edges up.
        if_branch, else_branch = sdfg.out_edges(self.if_guard)
        sdfg.remove_edge(if_branch)
        sdfg.remove_edge(else_branch)

        sdfg.add_edge(self.if_guard, if_branch.dst, InterstateEdge(assignments=if_branch.data.assignments))
        sdfg.add_edge(replica, else_branch.dst, InterstateEdge(assignments=else_branch.data.assignments))

        sdfg.add_edge(raised_if_guard, self.if_guard, InterstateEdge(condition=if_branch.data.condition))
        sdfg.add_edge(raised_if_guard, replica, InterstateEdge(condition=else_branch.data.condition))
