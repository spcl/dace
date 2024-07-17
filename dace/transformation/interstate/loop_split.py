# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import sympy as sp
from typing import List
from typing import Optional

from dace.properties import CodeBlock
from dace import sdfg as sd, symbolic
import copy
from dace.properties import Property, make_properties, CodeBlock
from dace.frontend.python.astutils import ASTFindReplace
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from dace.transformation import transformation as xf
from dace.transformation.interstate import ConditionalElimination

@make_properties
class LoopSplit(DetectLoop, xf.MultiStateTransformation):
    """
    For now works only on beginning or end.
    Looks for a condition on the loop variable inside a loop and splits the loop in two
    """

    @staticmethod
    def _eliminate_branch(sdfg: sd.SDFG, initial_edge: gr.Edge):
        sdfg.remove_edge(initial_edge)
        if sdfg.in_degree(initial_edge.dst) > 0:
            return
        state_list = [initial_edge.dst]
        while len(state_list) > 0:
            new_state_list = []
            for s in state_list:
                for e in sdfg.out_edges(s):
                    if len(sdfg.in_edges(e.dst)) == 1:
                        new_state_list.append(e.dst)
                sdfg.remove_node(s)
            state_list = new_state_list

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        guard: sd.SDFGState = self.loop_guard
        begin: sd.SDFGState = self.loop_begin
        after_state: sd.SDFGState = self.exit_state

        # If loop cannot be detected, fail
        found = find_for_loop(sdfg, guard, begin)
        if found is None:
            return False
        
        itervar, rng, loop_struct = found
        
        loop_states = list(sdutil.dfs_conditional(sdfg, sources=[begin], condition=lambda _, child: child != guard))

        for s in loop_states:
            edges = sdfg.out_edges(s)
            if len(edges) != 2:
                continue
            e, else_edge = edges
            cond = e.data.condition_sympy()
            else_cond = else_edge.data.condition_sympy()
            # swap edges if necessary
            if isinstance(else_cond, sp.Equality):
                cond, else_cond = else_cond, cond
                e, else_edge = else_edge, e
            if isinstance(cond, sp.Equality):
                if cond.lhs.name == itervar:
                    if cond.rhs == rng[0] or cond.rhs == rng[1]:
                        return True

        return False


    def instantiate_loop(
        self,
        sdfg: sd.SDFG,
        loop_states: List[sd.SDFGState],
        loop_subgraph: gr.SubgraphView,
        itervar: str,
        value: symbolic.SymbolicType,
        state_suffix=None,
    ):
        # Using to/from JSON copies faster than deepcopy (which will also
        # copy the parent SDFG)
        new_states = [sd.SDFGState.from_json(s.to_json(), context={'sdfg': sdfg}) for s in loop_states]

        # Replace iterate with value in each state
        for state in new_states:
            state.label = state.label + '_' + itervar + '_' + (state_suffix if state_suffix is not None else str(value))
            state.replace(itervar, value)

        # Add subgraph to original SDFG
        for edge in loop_subgraph.edges():
            src = new_states[loop_states.index(edge.src)]
            dst = new_states[loop_states.index(edge.dst)]

            # Replace conditions in subgraph edges
            data: sd.InterstateEdge = copy.deepcopy(edge.data)
            if data.condition:
                ASTFindReplace({itervar: str(value)}).visit(data.condition.code[0])

            sdfg.add_edge(src, dst, data)

        return new_states

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
            end = match[a] - step
        if end is None:
            match = condition.match(itersym <= a)
            if match:
                op = '<='
                end = match[a] - step
        if end is None:
            match = condition.match(itersym > a)
            if match:
                op = '>'
                end = match[a] - step
        if end is None:
            match = condition.match(itersym >= a)
            if match:
                op = '>='
                end = match[a] - step
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
        for s in loop_states:
            edges = sdfg.out_edges(s)
            if len(edges) != 2:
                continue
            e, else_edge = edges
            cond = e.data.condition_sympy()
            else_cond = else_edge.data.condition_sympy()
            # swap edges if necessary
            if isinstance(else_cond, sp.Equality):
                cond, else_cond = else_cond, cond
                e, else_edge = else_edge, e
            if isinstance(cond, sp.Equality):
                if cond.lhs.name == itervar:
                    if cond.rhs == rng[0]:
                        init_edges = []
                        before_states = loop_struct[0]
                        for before_state in before_states:
                            init_edge = sdfg.edges_between(before_state, guard)[0]
                            init_edge.data.assignments[itervar] = str(rng[0] + rng[2])
                            init_edges.append(init_edge)
                        append_states = before_states

                        # Instantiate loop states with iterate value
                        state_name: str = 'start_' + itervar
                        state_name = state_name.replace('-', 'm').replace('+', 'p').replace('*', 'M').replace('/', 'D')
                        new_states = self.instantiate_loop(
                            sdfg,
                            loop_states,
                            loop_subgraph,
                            itervar,
                            rng[0],
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
                        
                        self._eliminate_branch(sdfg, e)
                        # remove conditional from else_edge
                        sdfg.remove_edge(else_edge)
                        sdfg.add_edge(else_edge.src, else_edge.dst, sd.InterstateEdge(assignments=else_edge.data.assignments))
                        break
                    elif cond.rhs == rng[1]:
                        condition_edge.data.condition = CodeBlock(self._modify_cond(condition_edge.data.condition, itervar, rng[2]))
                        not_condition_edge.data.condition = CodeBlock(
                            self._modify_cond(not_condition_edge.data.condition, itervar, rng[2]))
                        prepend_state = after_state

                        # Instantiate loop states with iterate value
                        state_name: str = 'end_' + itervar
                        state_name = state_name.replace('-', 'm').replace('+', 'p').replace('*', 'M').replace('/', 'D')
                        new_states = self.instantiate_loop(
                            sdfg,
                            loop_states,
                            loop_subgraph,
                            itervar,
                            rng[1],
                            state_name,
                        )

                        # Connect states to before the loop with unconditional edges
                        sdfg.add_edge(new_states[last_id], prepend_state, sd.InterstateEdge())
                        prepend_state = new_states[first_id]

                        # Reconnect edge to guard state from last peeled iteration
                        if prepend_state != after_state:
                            sdfg.remove_edge(not_condition_edge)
                            sdfg.add_edge(guard, prepend_state, not_condition_edge.data)
                            
                        self._eliminate_branch(sdfg, e)
                        # remove condition from else_edge
                        sdfg.remove_edge(else_edge)
                        sdfg.add_edge(else_edge.src, else_edge.dst, sd.InterstateEdge(assignments=else_edge.data.assignments))
                        break
        
        xform = ConditionalElimination()
        xform.conditional = sp.Ne(rng[0], rng[1])
        xform.apply(None, sdfg)
