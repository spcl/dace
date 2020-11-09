# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop annotation transformation """

from dace.transformation.interstate.loop_detection import (DetectLoop,
                                                           find_for_loop)
from dace import sdfg as sd, symbolic
from dace.registry import autoregister
from dace.sdfg import graph as gr, utils as sdutil
from dace.subsets import Range

@autoregister
class AnnotateLoop(DetectLoop):
    """
    Annotates states in loop constructs according to the loop range.

    This sets the `ranges` property for each state with a loop range.
    Additionally, each loop guard gets annotated with the boolean attribute
    `is_loop_guard` (True), the string attribute `itvar` which holds the name
    of the loop's iteration variable, and the attribute `condition_edge`, which
    points to the edge holding the condition leading in to the loop.
    """

    @staticmethod
    def annotates_memlets():
        # DO NOT REAPPLY MEMLET PROPAGATION!
        return True

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict):
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg, strict):
            return False

        # Ensure range was not yet given.
        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])
        guard_inedges = graph.in_edges(guard)
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        if itervar in begin.ranges:
            return False

        return True

    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])

        # Obtain iteration variable, range, and stride.
        itervar, rng, _ = find_for_loop(sdfg, guard, begin)

        # Make sure the range is flipped in a direction such that the stride
        # is positive (in order to match subsets.Range).
        start, stop, stride = rng
        if (stride < 0) == True:
            rng = (stop, start, -stride)

        # Get loop states
        loop_states = list(sdutil.dfs_conditional(
            sdfg,
            sources=[begin],
            condition=lambda _, child: child != guard
        ))
        loop_subgraph = gr.SubgraphView(sdfg, loop_states)
        for v in loop_subgraph.nodes():
            v.ranges[itervar] = Range([rng])
        guard.ranges[itervar] = Range([rng])
        guard.condition_edge = sdfg.edges_between(guard, begin)[0]
        guard.is_loop_guard = True
        guard.itvar = itervar
