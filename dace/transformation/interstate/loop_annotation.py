# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop annotation transformation """

import sympy as sp
from dace.transformation.interstate.loop_detection import DetectLoop
from dace import sdfg as sd, symbolic
from dace.sdfg import graph as gr, utils as sdutil
from typing import List, Tuple, Optional

class AnnotateLoop(DetectLoop):
    """ Annotates states in loop constructs according to the loop range. """


    @staticmethod
    def _loop_range(
        itervar: str,
        inedges: List[gr.Edge],
        condition: sp.Expr
    ) -> Optional[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
        """
        Finds loop range from state machine.

        :param itersym: String representing the iteration variable.
        :param inedges: Incoming edges into guard state (length must be 2).
        :param condition: Condition as sympy expression.
        :return: A three-tuple of (start, end, stride) expressions, or None if
                 proper for-loop was not detected. ``end`` is inclusive.
        """

        # Find starting expression and stride
        itersym = symbolic.symbol(itervar)

        if (itervar in inedges[0].data.assignments[itervar] and itervar not in inedges[1].data.assignments[itervar]):
            stride = (symbolic.pystr_to_symbolic(inedges[0].data.assignments[itervar]) - itersym)
            start = symbolic.pystr_to_symbolic(inedges[1].data.assignments[itervar])
        elif (itervar in inedges[1].data.assignments[itervar] and itervar not in inedges[0].data.assignments[itervar]):
            stride = (symbolic.pystr_to_symbolic(inedges[1].data.assignments[itervar]) - itersym)
            start = symbolic.pystr_to_symbolic(inedges[0].data.assignments[itervar])
        else:
            return None

        # Find condition by matching expressions
        end: Optional[sp.Expr] = None
        a = sp.Wild('a')
        match = condition.match(itersym < a)
        if match:
            end = match[a] - 1
        if end is None:
            match = condition.match(itersym <= a)
            if match:
                end = match[a]
        if end is None:
            match = condition.match(itersym > a)
            if match:
                end = match[a] + 1
        if end is None:
            match = condition.match(itersym >= a)
            if match:
                end = match[a]
        if end is None:  # No match found
            return None

        return start, end, stride


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
        rng = AnnotateLoop._loop_range(itervar, guard_inedges, condition)

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
