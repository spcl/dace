# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Eliminates trivial loop """

from dace import sdfg as sd
from dace.properties import CodeBlock
from dace.transformation import helpers, transformation
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)


@transformation.single_level_sdfg_only
class TrivialLoopElimination(DetectLoop, transformation.MultiStateTransformation):
    """
    Eliminates loops with a single loop iteration.
    """

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        # Obtain iteration variable, range, and stride
        loop_info = self.loop_information()
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
        # Obtain iteration variable, range and stride
        itervar, (start, end, step), (_, body_end) = self.loop_information()
        states = self.loop_body()

        for state in states:
            state.replace(itervar, start)

        # Remove loop
        sdfg.remove_edge(self.loop_increment_edge())

        init_edge = self.loop_init_edge()
        init_edge.data.assignments = {}
        sdfg.add_edge(init_edge.src, self.loop_begin, init_edge.data)
        sdfg.remove_edge(init_edge)

        exit_edge = self.loop_exit_edge()
        exit_edge.data.condition = CodeBlock("1")
        sdfg.add_edge(body_end, exit_edge.dst, exit_edge.data)
        sdfg.remove_edge(exit_edge)

        sdfg.remove_nodes_from(self.loop_meta_states())
        if itervar in sdfg.symbols and helpers.is_symbol_unused(sdfg, itervar):
            sdfg.remove_symbol(itervar)
