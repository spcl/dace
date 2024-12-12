# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop peeling transformation """

import sympy as sp
from typing import List, Optional

from dace import sdfg as sd
from dace import symbolic
from dace.sdfg.state import ControlFlowRegion
from dace.properties import Property, make_properties, CodeBlock
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible


@make_properties
@explicit_cf_compatible
class LoopPeeling(LoopUnroll):
    """
    Splits the first `count` iterations of loop into multiple, separate control flow regions (one per iteration).
    """

    begin = Property(
        dtype=bool,
        default=True,
        desc='If True, peels loop from beginning (first `count` iterations), otherwise peels last `count` iterations.',
    )

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
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

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # Obtain loop information
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)
        is_symbolic = any([symbolic.issymbolic(r) for r in (start, end)])

        if self.begin:
            # Create states for loop subgraph
            peeled_iterations: List[ControlFlowRegion] = []
            for i in range(self.count):
                # Instantiate loop contents as a new control flow region with iterate value.
                current_index = start + (i * stride)
                iteration_region = self.instantiate_loop_iteration(graph, self.loop, current_index,
                                                                   str(i) if is_symbolic else None)

                # Connect iterations with unconditional edges
                if len(peeled_iterations) > 0:
                    graph.add_edge(peeled_iterations[-1], iteration_region, sd.InterstateEdge())
                peeled_iterations.append(iteration_region)

            # Connect the peeled iterations to the remainder of the loop and adjust the remaining iteration bounds.
            if peeled_iterations:
                for ie in graph.in_edges(self.loop):
                    graph.add_edge(ie.src, peeled_iterations[0], ie.data)
                    graph.remove_edge(ie)
                graph.add_edge(peeled_iterations[-1], self.loop, sd.InterstateEdge())

                new_start = symbolic.evaluate(start + (self.count * stride), sdfg.constants)
                self.loop.init_statement = CodeBlock(f'{self.loop.loop_variable} = {new_start}')
        else:
            # Create states for loop subgraph
            peeled_iterations: List[ControlFlowRegion] = []
            for i in reversed(range(self.count)):
                # Instantiate loop contents as a new control flow region with iterate value.
                current_index = pystr_to_symbolic(self.loop.loop_variable) + (i * stride)
                iteration_region = self.instantiate_loop_iteration(graph, self.loop, current_index,
                                                                   str(i) if is_symbolic else None)

                # Connect iterations with unconditional edges
                if len(peeled_iterations) > 0:
                    graph.add_edge(iteration_region, peeled_iterations[-1], sd.InterstateEdge())
                peeled_iterations.append(iteration_region)

            # Connect the peeled iterations to the remainder of the loop and adjust the remaining iteration bounds.
            if peeled_iterations:
                for oe in graph.out_edges(self.loop):
                    graph.add_edge(peeled_iterations[0], oe.dst, oe.data)
                    graph.remove_edge(oe)
                graph.add_edge(self.loop, peeled_iterations[-1], sd.InterstateEdge())

                new_cond = CodeBlock(self._modify_cond(self.loop.loop_condition, self.loop.loop_variable, stride))
                self.loop.loop_condition = new_cond
