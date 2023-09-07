# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline all scope blocks in SDFGs. """

from typing import Any, Set, Optional

from dace.frontend.python import astutils
from dace.sdfg import SDFG, ControlFlowGraph, InterstateEdge, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import LoopScopeBlock, ScopeBlock
from dace.transformation import transformation


class LoopScopeInline(transformation.MultiStateTransformation):
    """
    Inlines a loop scope block into a legacy-style state machine.
    """

    block = transformation.PatternNode(LoopScopeBlock)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.block)]

    def can_be_applied(self, graph: ControlFlowGraph, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        return True

    def apply(self, graph: ControlFlowGraph, sdfg: SDFG) -> Optional[int]:
        parent: ScopeBlock = graph

        internal_start = self.block.start_block

        # Construct the basic loop state structure.
        init_state = parent.add_state(self.block.label + '_init')
        for b_edge in parent.in_edges(self.block):
            parent.add_edge(b_edge.src, init_state, b_edge.data)
            parent.remove_edge(b_edge)

        guard_state = parent.add_state(self.block.label + '_guard')
        init_edge = InterstateEdge()
        if self.block.init_statement is not None:
            init_edge.assignments = {
                self.block.loop_variable: self.block.init_statement.as_string.rpartition('=')[2].strip()
            }
        parent.add_edge(init_state, guard_state, init_edge)

        end_state = parent.add_state(self.block.label + '_end')
        cond_expr = self.block.scope_condition.code
        parent.add_edge(guard_state, end_state,
                        InterstateEdge(CodeBlock(astutils.negate_expr(cond_expr)).code))
        for a_edge in parent.out_edges(self.block):
            parent.add_edge(end_state, a_edge.dst, a_edge.data)
            parent.remove_edge(a_edge)

        last_loop_state = parent.add_state(self.block.label + '_loop')
        loop_edge = InterstateEdge()
        if self.block.update_statement is not None:
            loop_edge.assignments = {
                self.block.loop_variable: self.block.update_statement.as_string.rpartition('=')[2].strip()
            }
        parent.add_edge(last_loop_state, guard_state, loop_edge)

        to_connect: Set[SDFGState] = set()
        for node in self.block.nodes():
            parent.add_node(node)
            if self.block.out_degree(node) == 0:
                to_connect.add(node)
        for edge in self.block.edges():
            parent.add_edge(edge.src, edge.dst, edge.data)

        # Connect the loop states
        parent.add_edge(guard_state, internal_start,
                        InterstateEdge(CodeBlock(cond_expr).code))
        for node in to_connect:
            parent.add_edge(node, last_loop_state, InterstateEdge())

        parent.remove_node(self.block)
