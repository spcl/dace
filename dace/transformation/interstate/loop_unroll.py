# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from dace.transformation import transformation as xf

@make_properties
class LoopUnroll(DetectLoop, xf.MultiStateTransformation):
    """ Unrolls a state machine for-loop into multiple states """

    count = Property(
        dtype=int,
        default=0,
        desc='Number of iterations to unroll, or zero for all '
        'iterations (loop must be constant-sized for 0)',
    )

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        # Is this even a loop
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg, permissive):
            return False

        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])
        found = find_for_loop(graph, guard, begin)

        # If loop cannot be detected, fail
        if not found:
            return False
        _, rng, _ = found

        # If loop stride is not specialized or constant-sized, fail
        if symbolic.issymbolic(rng[2], sdfg.constants):
            return False
        # If loop range diff is not constant-sized, fail
        if symbolic.issymbolic(rng[1] - rng[0], sdfg.constants):
            return False
        return True

    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride, together with the last
        # state(s) before the loop and the last loop state.
        itervar, rng, loop_struct = find_for_loop(sdfg, guard, begin)

        # Loop must be fully unrollable for now.
        if self.count != 0:
            raise NotImplementedError  # TODO(later)

        # Get loop states
        loop_states = list(sdutil.dfs_conditional(sdfg, sources=[begin], condition=lambda _, child: child != guard))
        first_id = loop_states.index(begin)
        last_state = loop_struct[1]
        last_id = loop_states.index(last_state)
        loop_subgraph = gr.SubgraphView(sdfg, loop_states)

        try:
            start, end, stride = (r for r in rng)
            stride = symbolic.evaluate(stride, sdfg.constants)
            loop_diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
            is_symbolic = any([symbolic.issymbolic(r) for r in rng[:2]])
        except TypeError:
            raise TypeError('Loop difference and strides cannot be symbolic.')
        # Create states for loop subgraph
        unrolled_states = []

        for i in range(0, loop_diff, stride):
            current_index = start + i
            # Instantiate loop states with iterate value
            new_states = self.instantiate_loop(sdfg, loop_states, loop_subgraph, itervar, current_index,
                                               str(i) if is_symbolic else None)

            # Connect iterations with unconditional edges
            if len(unrolled_states) > 0:
                sdfg.add_edge(unrolled_states[-1][1], new_states[first_id], sd.InterstateEdge())

            unrolled_states.append((new_states[first_id], new_states[last_id]))

        # Get any assignments that might be on the edge to the after state
        after_assignments = (sdfg.edges_between(guard, after_state)[0].data.assignments)

        # Connect new states to before and after states without conditions
        if unrolled_states:
            before_states = loop_struct[0]
            for before_state in before_states:
                sdfg.add_edge(before_state, unrolled_states[0][0], sd.InterstateEdge())
            sdfg.add_edge(unrolled_states[-1][1], after_state, sd.InterstateEdge(assignments=after_assignments))

        # Remove old states from SDFG
        sdfg.remove_nodes_from([guard] + loop_states)

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
            state.set_label(state.label + '_' + itervar + '_' +
                            (state_suffix if state_suffix is not None else str(value)))
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
