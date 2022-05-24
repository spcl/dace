# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Eliminates trivial loop """

from dace import registry, sdfg as sd
from dace.properties import CodeBlock
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)


class TrivialLoopElimination(DetectLoop, transformation.MultiStateTransformation, transformation.SimplifyPass):
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
        after: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range, and stride
        itervar, (start, end, step), _ = find_for_loop(sdfg, guard, body)

        try:
            if step > 0 and start + step < end:
                return False
            if step < 0 and start + step > end:
                return False
        except:
            # if the relation can't be determined it's not a trivial loop
            return False

        return True
      
        
    def apply(self, _, sdfg: sd.SDFG):
        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin
        after: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range and stride
        itervar, (start, end, step), _ = find_for_loop(sdfg, guard, body)
        
        body.free_symbols.remove(itervar)
        body.replace(itervar, start)

        # remove loop
        for body_inedge in sdfg.in_edges(body):
            sdfg.remove_edge(body_inedge)
        for body_outedge in sdfg.out_edges(body):
            sdfg.remove_edge(body_outedge)

        for guard_inedge in sdfg.in_edges(guard):
            guard_inedge.data.assignments = {}
            sdfg.add_edge(guard_inedge.src, body, guard_inedge.data)
            sdfg.remove_edge(guard_inedge)
        for guard_outedge in sdfg.out_edges(guard):
            guard_outedge.data.condition = CodeBlock("1")
            sdfg.add_edge(body, guard_outedge.dst, guard_outedge.data)
            sdfg.remove_edge(guard_outedge)
        sdfg.remove_node(guard)
        if itervar in sdfg.symbols:
            del sdfg.symbols[itervar]