# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import sympy as sp
from typing import Optional

from dace import sdfg as sd
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from dace.transformation.interstate.loop_unroll import LoopUnroll


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

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        guard = self.loop_guard
        begin = self.loop_begin

        # If loop cannot be detected, fail
        found = find_for_loop(sdfg, guard, begin)
        if found is None:
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

    def apply(self, _, sdfg: sd.SDFG):
        ####################################################################
        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        begin: sd.SDFGState = self.loop_begin
        after_state: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range, and stride
        condition_edge = sdfg.edges_between(guard, begin)[0]
        not_condition_edge = sdfg.edges_between(guard, after_state)[0]
        itervar, rng, loop_struct = find_for_loop(sdfg, guard, begin)

        # Get loop states
        loop_states = list(sdutil.dfs_conditional(sdfg, sources=[begin], condition=lambda _, child: child != guard))
        first_id = loop_states.index(begin)
        last_state = loop_struct[1]
        last_id = loop_states.index(last_state)
        loop_subgraph = gr.SubgraphView(sdfg, loop_states)

        ####################################################################
        # Transform

        if self.begin:
            # If begin, change initialization assignment and prepend states before
            # guard
            init_edges = []
            before_states = loop_struct[0]
            for before_state in before_states:
                init_edge = sdfg.edges_between(before_state, guard)[0]
                init_edge.data.assignments[itervar] = str(rng[0] + self.count * rng[2])
                init_edges.append(init_edge)
            append_states = before_states

            # Add `count` states, each with instantiated iteration variable
            for i in range(self.count):
                # Instantiate loop states with iterate value
                state_name: str = 'start_' + itervar + str(i * rng[2])
                state_name = state_name.replace('-', 'm').replace('+', 'p').replace('*', 'M').replace('/', 'D')
                new_states = self.instantiate_loop(
                    sdfg,
                    loop_states,
                    loop_subgraph,
                    itervar,
                    rng[0] + i * rng[2],
                    state_name,
                )

                # Connect states to before the loop with unconditional edges
                for append_state in append_states:
                    sdfg.add_edge(append_state, new_states[first_id], sd.InterstateEdge())
                append_states = [new_states[last_id]]

            # Reconnect edge to guard state from last peeled iteration
            for append_state in append_states:
                if append_state not in before_states:
                    for init_edge in init_edges:
                        sdfg.remove_edge(init_edge)
                    sdfg.add_edge(append_state, guard, init_edges[0].data)
        else:
            # If begin, change initialization assignment and prepend states before
            # guard
            itervar_sym = pystr_to_symbolic(itervar)
            condition_edge.data.condition = CodeBlock(self._modify_cond(condition_edge.data.condition, itervar, rng[2]))
            not_condition_edge.data.condition = CodeBlock(
                self._modify_cond(not_condition_edge.data.condition, itervar, rng[2]))
            prepend_state = after_state

            # Add `count` states, each with instantiated iteration variable
            for i in reversed(range(self.count)):
                # Instantiate loop states with iterate value
                state_name: str = 'end_' + itervar + str(-i * rng[2])
                state_name = state_name.replace('-', 'm').replace('+', 'p').replace('*', 'M').replace('/', 'D')
                new_states = self.instantiate_loop(
                    sdfg,
                    loop_states,
                    loop_subgraph,
                    itervar,
                    itervar_sym + i * rng[2],
                    state_name,
                )

                # Connect states to before the loop with unconditional edges
                sdfg.add_edge(new_states[last_id], prepend_state, sd.InterstateEdge())
                prepend_state = new_states[first_id]

            # Reconnect edge to guard state from last peeled iteration
            if prepend_state != after_state:
                sdfg.remove_edge(not_condition_edge)
                sdfg.add_edge(guard, prepend_state, not_condition_edge.data)
