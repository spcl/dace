# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import copy
from typing import List, Optional

from dace import sdfg as sd, symbolic
from dace.properties import Property, make_properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

@make_properties
@xf.explicit_cf_compatible
class LoopUnroll(xf.MultiStateTransformation):
    """ Unrolls a for-loop into multiple individual control flow regions """

    loop = xf.PatternNode(LoopRegion)

    count = Property(
        dtype=int,
        default=0,
        desc='Number of iterations to unroll, or zero for all iterations (loop must be constant-sized for 0)',
    )

    inline_iterations = Property(dtype=bool, default=True,
                                 desc="Whether or not to inline individual iterations' CFGs after unrolling")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        # If loop stride is not specialized or constant-sized, fail
        if symbolic.issymbolic(step, sdfg.constants):
            return False
        # If loop range diff is not constant-sized, fail
        if symbolic.issymbolic(end - start, sdfg.constants):
            return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg):
        # Loop must be fully unrollable for now.
        if self.count != 0:
            raise NotImplementedError # TODO(later)

        # Obtain loop information
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)

        try:
            stride = symbolic.evaluate(stride, sdfg.constants)
            loop_diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
            is_symbolic = any([symbolic.issymbolic(r) for r in (start, end)])
        except TypeError:
            raise TypeError('Loop difference and strides cannot be symbolic.')

        # Create states for loop subgraph
        unrolled_iterations: List[ControlFlowRegion] = []
        for i in range(0, loop_diff, stride):
            # Instantiate loop contents as a new control flow region with iterate value.
            current_index = start + i
            iteration_region = self.instantiate_loop_iteration(graph, self.loop, current_index,
                                                               str(i) if is_symbolic else None)

            # Connect iterations with unconditional edges
            if len(unrolled_iterations) > 0:
                graph.add_edge(unrolled_iterations[-1], iteration_region, sd.InterstateEdge())
            unrolled_iterations.append(iteration_region)

        if unrolled_iterations:
            for ie in graph.in_edges(self.loop):
                graph.add_edge(ie.src, unrolled_iterations[0], ie.data)
            for oe in graph.out_edges(self.loop):
                graph.add_edge(unrolled_iterations[-1], oe.dst, oe.data)

        # Remove old loop.
        graph.remove_node(self.loop)

        if self.inline_iterations:
            for it in unrolled_iterations:
                it.inline()

    def instantiate_loop_iteration(self, graph: ControlFlowRegion, loop: LoopRegion, value: symbolic.SymbolicType,
                                   label_suffix: Optional[str] = None) -> ControlFlowRegion:
        it_label = loop.label + '_' + loop.loop_variable + (label_suffix if label_suffix is not None else str(value))
        iteration_region = ControlFlowRegion(it_label, graph.sdfg, graph)
        graph.add_node(iteration_region)
        block_map = {}
        for block in loop.nodes():
            # Using to/from JSON copies faster than deepcopy.
            new_block = sd.SDFGState.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            block_map[block] = new_block
            new_block.replace(loop.loop_variable, value)
            iteration_region.add_node(new_block, is_start_block=(block is loop.start_block))
        for edge in loop.edges():
            src = block_map[edge.src]
            dst = block_map[edge.dst]
            # Replace conditions in subgraph edges
            data = copy.deepcopy(edge.data)
            if not data.is_unconditional():
                ASTFindReplace({loop.loop_variable: str(value)}).visit(data.condition)
            iteration_region.add_edge(src, dst, data)

        return iteration_region
