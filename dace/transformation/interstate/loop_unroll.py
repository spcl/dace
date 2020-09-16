# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import copy
import sympy as sp
import networkx as nx
from typing import List, Optional, Tuple

from dace import dtypes, registry, sdfg as sd, symbolic
from dace.properties import Property, make_properties
from dace.sdfg import graph as gr, nodes
from dace.sdfg import utils as sdutil
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation.interstate.loop_detection import DetectLoop


@registry.autoregister
@make_properties
class LoopUnroll(DetectLoop):
    """ Unrolls a state machine for-loop into multiple states """

    count = Property(dtype=int,
                     default=0,
                     desc="Number of iterations to unroll, or zero for all "
                     "iterations (loop must be constant-sized for 0)")

    @staticmethod
    def _loop_range(
            itervar: str, inedges: List[gr.Edge],
            condition: sp.Expr) -> Optional[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
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
        if (itervar in inedges[0].data.assignments[itervar]
                and itervar not in inedges[1].data.assignments[itervar]):
            stride = (symbolic.pystr_to_symbolic(
                inedges[0].data.assignments[itervar]) - itersym)
            start = symbolic.pystr_to_symbolic(
                inedges[1].data.assignments[itervar])
        elif (itervar in inedges[1].data.assignments[itervar]
              and itervar not in inedges[0].data.assignments[itervar]):
            stride = (symbolic.pystr_to_symbolic(
                inedges[1].data.assignments[itervar]) - itersym)
            start = symbolic.pystr_to_symbolic(
                inedges[0].data.assignments[itervar])
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

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg,
                                         strict):
            return False

        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        # Obtain iteration variable, range, and stride
        guard_inedges = graph.in_edges(guard)
        condition_edge = graph.edges_between(guard, begin)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()

        # If loop cannot be detected, fail
        rng = LoopUnroll._loop_range(itervar, guard_inedges, condition)
        if not rng:
            return False

        # If loop is not specialized or constant-sized, fail
        if any(symbolic.issymbolic(r, sdfg.constants) for r in rng):
            return False

        return True

    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(
            self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        guard_inedges = sdfg.in_edges(guard)
        condition_edge = sdfg.edges_between(guard, begin)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()
        rng = LoopUnroll._loop_range(itervar, guard_inedges, condition)

        # Loop must be unrollable
        if self.count == 0 and any(
                symbolic.issymbolic(r, sdfg.constants) for r in rng):
            raise ValueError('Loop cannot be fully unrolled, size is symbolic')
        if self.count != 0:
            raise NotImplementedError  # TODO(later)

        # Find the state prior to the loop
        if rng[0] == symbolic.pystr_to_symbolic(
                guard_inedges[0].data.assignments[itervar]):
            before_state: sd.SDFGState = guard_inedges[0].src
            last_state: sd.SDFGState = guard_inedges[1].src
        else:
            before_state: sd.SDFGState = guard_inedges[1].src
            last_state: sd.SDFGState = guard_inedges[0].src

        # Get loop states
        loop_states = list(
            sdutil.dfs_conditional(sdfg,
                                   sources=[begin],
                                   condition=lambda _, child: child != guard))
        first_id = loop_states.index(begin)
        last_id = loop_states.index(last_state)
        loop_subgraph = gr.SubgraphView(sdfg, loop_states)

        # Evaluate the real values of the loop
        start, end, stride = (symbolic.evaluate(r, sdfg.constants) for r in rng)

        # Create states for loop subgraph
        unrolled_states = []
        for i in range(start, end + 1, stride):
            # Instantiate loop states with iterate value
            new_states = self.instantiate_loop(sdfg, loop_states, loop_subgraph,
                                               itervar, i)

            # Connect iterations with unconditional edges
            if len(unrolled_states) > 0:
                sdfg.add_edge(unrolled_states[-1][1], new_states[first_id],
                              sd.InterstateEdge())

            unrolled_states.append((new_states[first_id], new_states[last_id]))

        # Connect new states to before and after states without conditions
        if unrolled_states:
            sdfg.add_edge(before_state, unrolled_states[0][0],
                          sd.InterstateEdge())
            sdfg.add_edge(unrolled_states[-1][1], after_state,
                          sd.InterstateEdge())

        # Remove old states from SDFG
        sdfg.remove_nodes_from([guard] + loop_states)

    def instantiate_loop(self, sdfg: sd.SDFG, loop_states: List[sd.SDFGState],
                         loop_subgraph: gr.SubgraphView, itervar: str,
                         value: symbolic.SymbolicType):
        # Using to/from JSON copies faster than deepcopy (which will also
        # copy the parent SDFG)
        new_states = [
            sd.SDFGState.from_json(s.to_json(), context={'sdfg': sdfg})
            for s in loop_states
        ]

        # Replace iterate with value in each state
        for state in new_states:
            state.set_label(state.label + '_%s_%d' % (itervar, value))
            state.replace(itervar, value)

        # Add subgraph to original SDFG
        for edge in loop_subgraph.edges():
            src = new_states[loop_states.index(edge.src)]
            dst = new_states[loop_states.index(edge.dst)]

            # Replace conditions in subgraph edges
            data: sd.InterstateEdge = copy.deepcopy(edge.data)
            if data.condition:
                ASTFindReplace({itervar: str(value)}).visit(data.condition)

            sdfg.add_edge(src, dst, data)

        return new_states