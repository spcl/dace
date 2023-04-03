# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Eliminates trivial loop """

from dace import sdfg as sd
from dace.sdfg import graph as gr, utils as sdutil
from dace.properties import CodeBlock
from dace.transformation import helpers, transformation
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from typing import List


class TrivialLoopElimination(DetectLoop, transformation.MultiStateTransformation):
    """
    Eliminates loops with a single loop iteration.
    """

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin

        # Obtain iteration variable, range, and stride
        loop_info = find_for_loop(sdfg, guard, body)
        if not loop_info:
            return False
        _, (start, end, step), _ = loop_info

        try:
            if step > 0 and start + step < end + 1:
                return False
            if step < 0 and start + step > end - 1:
                return False
        except:
            # if the relation can't be determined it's not a trivial loop
            return False

        return True

    def apply(self, _, sdfg: sd.SDFG):
        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin

        # Obtain iteration variable, range and stride
        itervar, (start, end, step), (_, body_end) = find_for_loop(sdfg, guard, body)

        # # Find all loop-body states
        # states = set()
        # to_visit = [body]
        # while to_visit:
        #     state = to_visit.pop(0)
        #     for _, dst, _ in sdfg.out_edges(state):
        #         if dst not in states and dst is not guard:
        #             to_visit.append(dst)
        #     states.add(state)
        states: List[sd.SDFGState] = list(sdutil.dfs_conditional(sdfg, [body], lambda _, c: c is not guard))
        subgraph = gr.SubgraphView(sdfg, states)

        # Remove itervar assignments
        for edge in sdfg.in_edges(guard):
            if itervar in edge.data.assignments:
                del edge.data.assignments[itervar]

        # Replace itervar with start
        replace_dict = {itervar: start}
        for state in subgraph.nodes():
            state.replace_dict(replace_dict)
        for edge in subgraph.edges():
            edge.data.replace_dict(replace_dict)
        for edge in sdfg.edges_between(guard, body):
            edge.data.replace_dict(replace_dict)
        for edge in sdfg.edges_between(body_end, guard):
            edge.data.replace_dict(replace_dict)

        # Remove loop
        guard_body_assignments = dict()
        for edge in sdfg.edges_between(guard, body):
            guard_body_assignments.update(edge.data.assignments)
            sdfg.remove_edge(edge)
        body_guard_assignments = dict()
        for edge in sdfg.edges_between(body_end, guard):
            body_guard_assignments.update(edge.data.assignments)
            sdfg.remove_edge(edge)

        for guard_inedge in sdfg.in_edges(guard):
            guard_inedge.data.assignments.update(guard_body_assignments)
            sdfg.add_edge(guard_inedge.src, body, guard_inedge.data)
            sdfg.remove_edge(guard_inedge)
        for guard_outedge in sdfg.out_edges(guard):
            guard_outedge.data.condition = CodeBlock("1")
            guard_outedge.data.assignments.update(body_guard_assignments)
            sdfg.add_edge(body_end, guard_outedge.dst, guard_outedge.data)

            sdfg.remove_edge(guard_outedge)
        sdfg.remove_node(guard)
        if itervar in sdfg.symbols and helpers.is_symbol_unused(sdfg, itervar):
            sdfg.remove_symbol(itervar)
