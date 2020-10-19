# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import sympy as sp
from typing import Optional

from dace import registry, sdfg as sd, symbolic
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.interstate.loop_unroll import LoopUnroll


@registry.autoregister
@make_properties
class LoopPeeling(LoopUnroll):
    """
    Splits the first `count` iterations of a state machine for-loop into
    multiple, separate states.
    """

    begin = Property(
        dtype=bool,
        default=True,
        desc='If True, peels loop from beginning (first `count` '
        'iterations), otherwise peels last `count` iterations.',
    )

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

        return True

    def _modify_cond(self, condition, var, step):
        condition = pystr_to_symbolic(condition.as_string)
        itersym = pystr_to_symbolic(var)
        # Find condition by matching expressions
        end: Optional[sp.Expr] = None
        a = sp.Wild('a')
        op = ''
        match = condition.match(itersym < a)
        if match:
            op = '<'
            end = match[a] - self.count * step
        if end is None:
            match = condition.match(itersym <= a)
            if match:
                op = '<='
                end = match[a] - self.count * step
        if end is None:
            match = condition.match(itersym > a)
            if match:
                op = '>'
                end = match[a] - self.count * step
        if end is None:
            match = condition.match(itersym >= a)
            if match:
                op = '>='
                end = match[a] - self.count * step
        if len(op) == 0:
            raise ValueError('Cannot match loop condition for peeling')

        res = str(itersym) + op + str(end)
        return res

    def apply(self, sdfg: sd.SDFG):
        ####################################################################
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(
            self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        guard_inedges = sdfg.in_edges(guard)
        condition_edge = sdfg.edges_between(guard, begin)[0]
        not_condition_edge = sdfg.edges_between(guard, after_state)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()
        rng = self._loop_range(itervar, guard_inedges, condition)

        # Find the state prior to the loop
        if rng[0] == symbolic.pystr_to_symbolic(
                guard_inedges[0].data.assignments[itervar]):
            init_edge: sd.InterstateEdge = guard_inedges[0]
            before_state: sd.SDFGState = guard_inedges[0].src
            last_state: sd.SDFGState = guard_inedges[1].src
        else:
            init_edge: sd.InterstateEdge = guard_inedges[1]
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

        ####################################################################
        # Transform

        if self.begin:
            # If begin, change initialization assignment and prepend states before
            # guard
            init_edge.data.assignments[itervar] = str(rng[0] +
                                                      self.count * rng[2])
            append_state = before_state

            # Add `count` states, each with instantiated iteration variable
            for i in range(self.count):
                # Instantiate loop states with iterate value
                state_name: str = 'start_' + itervar + str(i * rng[2])
                state_name = state_name.replace('-',
                                                'm').replace('+', 'p').replace(
                                                    '*', 'M').replace('/', 'D')
                new_states = self.instantiate_loop(
                    sdfg,
                    loop_states,
                    loop_subgraph,
                    itervar,
                    rng[0] + i * rng[2],
                    state_name,
                )

                # Connect states to before the loop with unconditional edges
                sdfg.add_edge(append_state, new_states[first_id],
                              sd.InterstateEdge())
                append_state = new_states[last_id]

            # Reconnect edge to guard state from last peeled iteration
            if append_state != before_state:
                sdfg.remove_edge(init_edge)
                sdfg.add_edge(append_state, guard, init_edge.data)
        else:
            # If begin, change initialization assignment and prepend states before
            # guard
            itervar_sym = pystr_to_symbolic(itervar)
            condition_edge.data.condition = CodeBlock(
                self._modify_cond(condition_edge.data.condition, itervar,
                                  rng[2]))
            not_condition_edge.data.condition = CodeBlock(
                self._modify_cond(not_condition_edge.data.condition, itervar,
                                  rng[2]))
            prepend_state = after_state

            # Add `count` states, each with instantiated iteration variable
            for i in reversed(range(self.count)):
                # Instantiate loop states with iterate value
                state_name: str = 'end_' + itervar + str(-i * rng[2])
                state_name = state_name.replace('-',
                                                'm').replace('+', 'p').replace(
                                                    '*', 'M').replace('/', 'D')
                new_states = self.instantiate_loop(
                    sdfg,
                    loop_states,
                    loop_subgraph,
                    itervar,
                    itervar_sym + i * rng[2],
                    state_name,
                )

                # Connect states to before the loop with unconditional edges
                sdfg.add_edge(new_states[last_id], prepend_state,
                              sd.InterstateEdge())
                prepend_state = new_states[first_id]

            # Reconnect edge to guard state from last peeled iteration
            if prepend_state != after_state:
                sdfg.remove_edge(not_condition_edge)
                sdfg.add_edge(guard, prepend_state, not_condition_edge.data)
