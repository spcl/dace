# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Eliminates trivial loop """

from typing import List
from dace import sdfg as sd
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, transformation
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from dace.transformation.passes.analysis import loop_analysis


@transformation.single_level_sdfg_only
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

        # Find all loop-body states
        states = set()
        to_visit = [body]
        while to_visit:
            state = to_visit.pop(0)
            for _, dst, _ in sdfg.out_edges(state):
                if dst not in states and dst is not guard:
                    to_visit.append(dst)
            states.add(state)

        for state in states:
            state.replace(itervar, start)

        # remove loop
        for body_inedge in sdfg.in_edges(body):
            sdfg.remove_edge(body_inedge)
        for body_outedge in sdfg.out_edges(body_end):
            sdfg.remove_edge(body_outedge)

        for guard_inedge in sdfg.in_edges(guard):
            guard_inedge.data.assignments = {}
            sdfg.add_edge(guard_inedge.src, body, guard_inedge.data)
            sdfg.remove_edge(guard_inedge)
        for guard_outedge in sdfg.out_edges(guard):
            guard_outedge.data.condition = CodeBlock("1")
            sdfg.add_edge(body_end, guard_outedge.dst, guard_outedge.data)
            sdfg.remove_edge(guard_outedge)
        sdfg.remove_node(guard)
        if itervar in sdfg.symbols and helpers.is_symbol_unused(sdfg, itervar):
            sdfg.remove_symbol(itervar)


class TrivialLoopRegionElimination(transformation.MultiStateTransformation):
    """
    Eliminates loops with a single loop iteration.
    """

    loop_node = transformation.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls) -> List[helpers.SubgraphView]:
        return [sdutil.node_path_graph(cls.loop_node)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        try:
            start = loop_analysis.get_init_assignment(self.loop_node)
            end = loop_analysis.get_loop_end(self.loop_node)
            step = loop_analysis.get_loop_stride(self.loop_node)
            if step > 0 and start + step < end + 1:
                return False
            if step < 0 and start + step > end - 1:
                return False
        except:
            # if the relation can't be determined it's not a trivial loop
            return False

        return True

    def apply(self, _, sdfg: sd.SDFG):
        loop = self.loop_node
        start = loop_analysis.get_init_assignment(loop)

        for block in loop.nodes():
            block.replace(loop.loop_variable, start)

        # Add loop contents to loop parent graph.
        sink_nodes = loop.sink_nodes()
        for block in loop.nodes():
            loop.parent_graph.add_node(block)
            if block == loop.start_block:
                if loop.parent_graph.start_block == loop:
                    loop.parent_graph.start_block = block
                for e in loop.parent_graph.in_edges(loop):
                    iedge = e.data
                    loop.parent_graph.add_edge(e.src, block, iedge)
            if block in sink_nodes:
                for e in loop.parent_graph.out_edges(loop):
                    iedge = e.data
                    loop.parent_graph.add_edge(block, e.dst, iedge)
        for edge in loop.edges():
            loop.parent_graph.add_edge(edge.src, edge.dst, edge.data)

        for e in loop.parent_graph.all_edges(loop):
            loop.parent_graph.remove_edge(e)
        loop.parent_graph.remove_node(loop)

        if loop.loop_variable in sdfg.symbols and helpers.is_symbol_unused(sdfg, loop.loop_variable):
            sdfg.remove_symbol(loop.loop_variable)

        sdfg.reset_cfg_list()
