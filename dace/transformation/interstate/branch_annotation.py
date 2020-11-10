# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Conditional branch annotation transformation """

import networkx as nx

from dace.transformation.interstate.branch_detection import DetectBranch
from dace.registry import autoregister


@autoregister
class AnnotateBranch(DetectBranch):
    """
    Annotates states in conditional branch constructs.

    This annotates each processed branch pattern with the boolean attribute
    `_branch_annotated` (True) to make sure it does not get processed in any
    subsequent application of this pattern. Additionally, each guard state to a
    branch pattern that gets fully merged again, receives the attribute
    `full_merge_state`, which points to the state where all the branches meet
    again.
    """
    @staticmethod
    def annotates_memlets():
        # DO NOT REAPPLY MEMLET PROPAGATION!
        return True

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict):
        if not DetectBranch.can_be_applied(graph, candidate, expr_index, sdfg,
                                           strict):
            return False

        guard = graph.node(candidate[DetectBranch._branch_guard])
        # If this is already a loop guard, this isn't treated as a cond. branch.
        if getattr(guard, 'is_loop_guard', False):
            return False

        # Also ensure that this hasn't been annotated yet.
        if getattr(guard, '_branch_annotated', False):
            return False

        return True

    def apply(self, sdfg):
        guard = sdfg.node(self.subgraph[DetectBranch._branch_guard])

        guard.full_merge_state = None

        # We construct the dominance frontier for each state of the graph. For
        # this to work correctly, there needs to be one common explicit exit
        # state.
        dom_frontier = nx.dominance_frontiers(sdfg.nx, sdfg.start_state)

        common_frontier = set()
        out_edges = sdfg.out_edges(guard)

        # Get the dominance frontier for each child state and merge them into
        # one common frontier, representing the loop's dominance frontier. If a
        # state has no dominance frontier, add the state itself to the frontier.
        # This takes care of the case where a branch is fully merged, but one
        # branch contains no states.
        for oedge in out_edges:
            frontier = dom_frontier[oedge.dst]
            if not frontier:
                frontier = {oedge.dst}
            common_frontier |= frontier

        # If the common loop dominance frontier is exactly one state, we know
        # that all the branches merge at that state.
        if len(common_frontier) == 1:
            frontier_state = list(common_frontier)[0]
            guard.full_merge_state = frontier_state
        elif self.expr_index == 1 and sdfg.out_degree(guard) == 2:
            # If we matched the second case of branch detection, and have
            # exactly two branches, we know that the second branch's state is a
            # full merge state.
            guard.full_merge_state = sdfg.node(
                self.subgraph[DetectBranch._second_branch])

        # Mark this conditional branch construct as annotated, so it doesn't
        # get processed again when applied repeatedly.
        guard._branch_annotated = True
