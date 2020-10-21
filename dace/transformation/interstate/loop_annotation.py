# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop annotation transformation """

from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace import sdfg as sd, symbolic
from dace.sdfg import graph as gr, utils as sdutil

class AnnotateLoop(DetectLoop):
    """ Annotates states in loop constructs according to the loop range. """


    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        guard_inedges = sdfg.in_edges(guard)
        condition_edge = sdfg.edges_between(guard, begin)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()
        rng = LoopUnroll._loop_range(itervar, guard_inedges, condition)

        # Find the state prior to the loop
        if rng[0] == symbolic.pystr_to_symbolic(
                guard_inedges[0].data.assignments[itervar]):
            before_state: sd.SDFGState = guard_inedges[0].src
            last_state: sd.SDFGState = guard_inedges[1].src
        else:
            before_state: sd.SDFGState = guard_inedges[1].src
            last_state: sd.SDFGState = guard_inedges[0].src

        # Get loop states
        loop_states = list(sdutil.dfs_conditional(
            sdfg,
            sources=[begin],
            condition=lambda _, child: child != guard
        ))
        first_id = loop_states.index(begin)
        last_id = loop_states.index(last_state)
        loop_subgraph = gr.SubgraphView(sdfg, loop_states)
        for v in loop_subgraph.nodes():
            v.ranges[itervar] = rng
