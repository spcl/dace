# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import ast
import copy
from typing import List, Optional, Union

from dace import sdfg as sd, symbolic
from dace.properties import Property, make_properties
from dace.sdfg import InterstateEdge, utils as sdutil
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock, SDFGState
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

    inline_iterations = Property(dtype=bool,
                                 default=True,
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
            raise NotImplementedError  # TODO(later)

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
        # A state is returned as a replacement when the loop body is empty
        unrolled_iterations: List[Union[ControlFlowRegion, SDFGState]] = []
        for i in range(0, loop_diff, stride):
            # Instantiate loop contents as a new control flow region with iterate value.
            current_index = start + i
            is_symbolic |= symbolic.issymbolic(current_index)
            iteration_region = self.instantiate_loop_iteration(graph, self.loop, current_index,
                                                               str(i) if is_symbolic else None)

            # Connect iterations with unconditional edges
            if len(unrolled_iterations) > 0:
                assert unrolled_iterations[-1] in graph.nodes()
                assert iteration_region in graph.nodes()
                graph.add_edge(unrolled_iterations[-1], iteration_region, sd.InterstateEdge())
            unrolled_iterations.append(iteration_region)

        if len(unrolled_iterations) == 0:
            s = graph.add_state(label="empty_unroll", is_start_block=True)
            unrolled_iterations.append(s)

        if unrolled_iterations:
            for ie in graph.in_edges(self.loop):
                assert ie.src in graph.nodes()
                assert unrolled_iterations[0] in graph.nodes()
                graph.add_edge(ie.src, unrolled_iterations[0], ie.data)
            for oe in graph.out_edges(self.loop):
                assert unrolled_iterations[-1] in graph.nodes()
                assert oe.dst in graph.nodes()
                graph.add_edge(unrolled_iterations[-1], oe.dst, oe.data)

        # If we remove start block we need to update it
        was_start_block = graph.start_block == self.loop
        graph.remove_node(self.loop)
        if was_start_block:
            oes = graph.out_edges(self.loop)
            if len(oes) > 0:
                dst = oes[0].dst
                graph.remove_node(oe.dst)
                oes2 = graph.out_edges(dst)
                assert len(oes2) <= 1
                if len(oes2) == 1:
                    graph.add_node(oe.dst, is_start_block=True)
                    for oe2 in oes2:
                        graph.add_edge(dst, oe2.dst, copy.deepcopy(oes2))
                else:
                    graph.add_node(oe.dst, is_start_block=True)

        if self.inline_iterations:
            for it in unrolled_iterations:
                # SDFGState does not have an inline attribute
                if isinstance(it, SDFGState):
                    continue
                it.inline()

    def instantiate_loop_iteration(self,
                                   graph: ControlFlowRegion,
                                   loop: LoopRegion,
                                   value: symbolic.SymbolicType,
                                   label_suffix: Optional[str] = None) -> ControlFlowRegion:
        it_label = loop.label + '_' + loop.loop_variable + (label_suffix if label_suffix is not None else str(value))
        iteration_region = ControlFlowRegion(it_label, graph.sdfg, graph)

        graph.add_node(iteration_region)

        block_map = {}

        for block in loop.nodes():
            # Using to/from JSON copies faster than deepcopy.
            if isinstance(block, LoopRegion):
                new_block = LoopRegion.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            elif isinstance(block, ConditionalBlock):
                new_block = ConditionalBlock.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            elif isinstance(block, sd.SDFGState):
                new_block = sd.SDFGState.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            elif isinstance(block, ContinueBlock):
                new_block = ContinueBlock.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            elif isinstance(block, BreakBlock):
                new_block = BreakBlock.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            elif isinstance(block, ReturnBlock):
                new_block = ReturnBlock.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            elif isinstance(block, ControlFlowRegion):
                new_block = ControlFlowRegion.from_json(block.to_json(), context={'sdfg': graph.sdfg})
            else:
                raise Exception(f"Unsupported CFG Type {type(block)}")
            assert block not in block_map
            block_map[block] = new_block
            new_block.replace(loop.loop_variable, value)
            iteration_region.add_node(new_block, is_start_block=(block is loop.start_block))

        for edge in loop.edges():
            src = block_map[edge.src]
            dst = block_map[edge.dst]
            # Replace conditions in subgraph edges
            data = copy.deepcopy(edge.data)
            iteration_region.add_edge(src, dst, data)

        # Replace occurences of the loop variables on all interstate edges
        for edge, parent_graph in iteration_region.all_edges_recursive():  # Recursion needed for nested SDFGs
            if isinstance(edge.data, InterstateEdge):
                src = edge.src
                dst = edge.dst
                assert src in parent_graph.nodes()
                assert dst in parent_graph.nodes()
                if not edge.data.is_unconditional():
                    ASTFindReplace({loop.loop_variable: str(value)}).visit(edge.data.condition)

                new_assignments = dict()
                for k, v in edge.data.assignments.items():
                    k_ast = ast.parse(k)
                    v_ast = ast.parse(v)
                    ASTFindReplace({loop.loop_variable: str(value)}).visit(k_ast)
                    ASTFindReplace({loop.loop_variable: str(value)}).visit(v_ast)
                    new_assignments[ast.unparse(k_ast)] = ast.unparse(v_ast)
                edge.data.assignments = new_assignments

        for node in iteration_region.all_nodes_recursive():
            if isinstance(node, NestedSDFG):
                if loop.loop_variable in node.symbol_mapping:
                    node.symbol_mapping[loop.loop_variable] = ASTFindReplace({
                        loop.loop_variable: str(value)
                    }).visit(node.symbol_mapping[loop.loop_variable])
                if loop.loop_variable in node.symbol_mapping:
                    del node.symbol_mapping[loop.loop_variable]

        graph.reset_cfg_list()
        return iteration_region
