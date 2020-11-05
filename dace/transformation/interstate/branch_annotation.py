# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Conditional branch annotation transformation """

from math import e
import networkx as nx

from dace.transformation.interstate.branch_detection import DetectBranch
from dace import sdfg as sd, symbolic
from dace.registry import autoregister
from dace.sdfg import graph as gr, utils as sdutil
from dace.subsets import Range

@autoregister
class AnnotateBranch(DetectBranch):
    """ Annotates states in conditional branch constructs. """

    @staticmethod
    def annotates_memlets():
        # DO NOT REAPPLY MEMLET PROPAGATION!
        return True


    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict):
        if not DetectBranch.can_be_applied(
            graph, candidate, expr_index, sdfg, strict
        ):
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

        dom_frontier = nx.dominance_frontiers(sdfg.nx, sdfg.start_state)

        common_frontier = {}
        out_edges = sdfg.out_edges(guard)
        branch_states = []
        branch_states.append(out_edges.pop().dst)
        common_frontier = (
            dom_frontier[branch_states[0]] | {branch_states[0]}
        )
        for oedge in out_edges:
            common_frontier &= dom_frontier[oedge.dst] | {oedge.dst}
            branch_states.append(oedge.dst)

        if len(common_frontier) == 1:
            frontier_state = list(common_frontier)[0]

            traversal_queue = []
            visited_states = [guard]

            share = 1 / len(branch_states)
            received_shares = 0

            state = branch_states.pop()
            for branch_state in branch_states:
                traversal_queue.append({
                    'state': branch_state,
                    'share': share,
                })
            while state is not None:
                if state == guard:
                    # We've gone around in a loop, abort.
                    state = None
                    break
                elif state in visited_states or state == frontier_state:
                    received_shares += share
                else:
                    visited_states.append(state)
                    oedges = sdfg.out_edges(state)
                    if len(oedges) == 1:
                        state = oedges[0].dst
                        continue
                    elif len(oedges) > 1:
                        share = share / len(oedges)
                        state = oedges.pop().dst
                        for e in oedges:
                            traversal_queue.append({
                                'state': e.dst,
                                'share': share,
                            })
                        continue
                if len(traversal_queue) > 0:
                    traversal = traversal_queue.pop()
                    state = traversal['state']
                    share = traversal['share']
                    continue
                else:
                    state = None
                    break
            if received_shares == 1:
                guard.full_merge_state = frontier_state

        # Mark this conditional branch construct as annotated, so it doesn't
        # get processed again when applied repeatedly.
        guard._branch_annotated = True
