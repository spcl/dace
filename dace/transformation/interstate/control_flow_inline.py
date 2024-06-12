# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline control flow regions in SDFGs. """

from typing import Set, Optional, Tuple

from dace.frontend.python import astutils
from dace.sdfg import SDFG, InterstateEdge, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ConditionalRegion
from dace.transformation import transformation


class ControlFlowRegionInline(transformation.MultiStateTransformation):
    """
    Inlines a control flow region into a single state machine.
    """

    region = transformation.PatternNode(ControlFlowRegion)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.region)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        if isinstance(self.region, LoopRegion):
            return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG) -> Optional[int]:
        parent: ControlFlowRegion = graph

        internal_start = self.region.start_block

        end_state = parent.add_state(self.region.label + '_end')

        # Add all region states and make sure to keep track of all the ones that need to be connected in the end.
        to_connect: Set[SDFGState] = set()
        for node in self.region.nodes():
            parent.add_node(node)
            if self.region.out_degree(node) == 0:
                to_connect.add(node)

        # Add all region edges.
        for edge in self.region.edges():
            parent.add_edge(edge.src, edge.dst, edge.data)

        # Redirect all edges to the region to the internal start state.
        for b_edge in parent.in_edges(self.region):
            parent.add_edge(b_edge.src, internal_start, b_edge.data)
            parent.remove_edge(b_edge)
        # Redirect all edges exiting the region to instead exit the end state.
        for a_edge in parent.out_edges(self.region):
            parent.add_edge(end_state, a_edge.dst, a_edge.data)
            parent.remove_edge(a_edge)

        for node in to_connect:
            parent.add_edge(node, end_state, InterstateEdge())

        # Remove the original loop.
        parent.remove_node(self.region)

        sdfg.reset_cfg_list()


class LoopRegionInline(transformation.MultiStateTransformation):
    """
    Inlines a loop region into a single state machine.
    """

    loop = transformation.PatternNode(LoopRegion)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # Check that the loop initialization and update statements each only contain assignments, if the loop has any.
        if self.loop.init_statement is not None:
            if isinstance(self.loop.init_statement.code, list):
                for stmt in self.loop.init_statement.code:
                    if not isinstance(stmt, astutils.ast.Assign):
                        return False
        if self.loop.update_statement is not None:
            if isinstance(self.loop.update_statement.code, list):
                for stmt in self.loop.update_statement.code:
                    if not isinstance(stmt, astutils.ast.Assign):
                        return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG) -> Optional[int]:
        parent: ControlFlowRegion = graph

        internal_start = self.loop.start_block

        # Add all boilerplate loop states necessary for the structure.
        init_state = parent.add_state(self.loop.label + '_init')
        guard_state = parent.add_state(self.loop.label + '_guard')
        end_state = parent.add_state(self.loop.label + '_end')
        loop_tail_state = parent.add_state(self.loop.label + '_tail')

        # Add all loop states and make sure to keep track of all the ones that need to be connected in the end.
        connect_to_tail: Set[SDFGState] = set()
        connect_to_end: Set[SDFGState] = set()
        for node in self.loop.nodes():
            node.label = self.loop.label + '_' + node.label
            parent.add_node(node)
            if isinstance(node, LoopRegion.BreakState):
                node.__class__ = SDFGState
                connect_to_end.add(node)
            elif isinstance(node, LoopRegion.ContinueState):
                node.__class__ = SDFGState
                connect_to_tail.add(node)
            elif self.loop.out_degree(node) == 0:
                connect_to_tail.add(node)

        # Add all internal loop edges.
        for edge in self.loop.edges():
            parent.add_edge(edge.src, edge.dst, edge.data)

        # Redirect all edges to the loop to the init state.
        for b_edge in parent.in_edges(self.loop):
            parent.add_edge(b_edge.src, init_state, b_edge.data)
            parent.remove_edge(b_edge)
        # Redirect all edges exiting the loop to instead exit the end state.
        for a_edge in parent.out_edges(self.loop):
            parent.add_edge(end_state, a_edge.dst, a_edge.data)
            parent.remove_edge(a_edge)

        # Add an initialization edge that initializes the loop variable if applicable.
        init_edge = InterstateEdge()
        if self.loop.init_statement is not None:
            init_edge.assignments = {}
            for stmt in self.loop.init_statement.code:
                assign: astutils.ast.Assign = stmt
                init_edge.assignments[assign.targets[0].id] = astutils.unparse(assign.value)
        if self.loop.inverted:
            parent.add_edge(init_state, internal_start, init_edge)
        else:
            parent.add_edge(init_state, guard_state, init_edge)

        # Connect the loop tail.
        update_edge = InterstateEdge()
        if self.loop.update_statement is not None:
            update_edge.assignments = {}
            for stmt in self.loop.update_statement.code:
                assign: astutils.ast.Assign = stmt
                update_edge.assignments[assign.targets[0].id] = astutils.unparse(assign.value)
        parent.add_edge(loop_tail_state, guard_state, update_edge)

        # Add condition checking edges and connect the guard state.
        cond_expr = self.loop.loop_condition.code
        parent.add_edge(guard_state, end_state,
                        InterstateEdge(CodeBlock(astutils.negate_expr(cond_expr)).code))
        parent.add_edge(guard_state, internal_start, InterstateEdge(CodeBlock(cond_expr).code))

        # Connect any end states from the loop's internal state machine to the tail state so they end a
        # loop iteration. Do the same for any continue states, and connect any break states to the end of the loop.
        for node in connect_to_tail:
            parent.add_edge(node, loop_tail_state, InterstateEdge())
        for node in connect_to_end:
            parent.add_edge(node, end_state, InterstateEdge())

        # Remove the original loop.
        parent.remove_node(self.loop)

        sdfg.reset_cfg_list()

class ConditionalRegionInline(transformation.MultiStateTransformation):
    """
    Inlines a conditional region into a single state machine.
    """

    conditional = transformation.PatternNode(ConditionalRegion)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.conditional)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG) -> Tuple[Set[SDFGState], Set[SDFGState]]:
        parent: ControlFlowRegion = graph

        # Add all boilerplate states necessary for the structure.
        guard_state = parent.add_state(self.conditional.label + '_guard')
        endif_state = parent.add_state(self.conditional.label + '_endinf')

        # Add all loop states and make sure to keep track of all the ones that need to be connected in the end.
        connect_to_tail: Set[SDFGState] = set()
        break_states: Set[SDFGState] = set()
        continue_states: Set[SDFGState] = set()
        for node in self.conditional.nodes():
            node.label = self.conditional.label + '_' + node.label
            parent.add_node(node)
            if isinstance(node, LoopRegion.BreakState):
                node.__class__ = SDFGState
                break_states.add(node)
            elif isinstance(node, LoopRegion.ContinueState):
                node.__class__ = SDFGState
                continue_states.add(node)
            elif self.conditional.out_degree(node) == 0:
                connect_to_tail.add(node)

        # Add all internal loop edges.
        for edge in self.conditional.edges():
            parent.add_edge(edge.src, edge.dst, edge.data)

        # Redirect all edges to the loop to the init state.
        for b_edge in parent.in_edges(self.conditional):
            parent.add_edge(b_edge.src, guard_state, b_edge.data)
            parent.remove_edge(b_edge)
        # Redirect all edges exiting the loop to instead exit the end state.
        for a_edge in parent.out_edges(self.conditional):
            parent.add_edge(endif_state, a_edge.dst, a_edge.data)
            parent.remove_edge(a_edge)

        # Add condition checking edges and connect the guard state.
        parent.add_edge(guard_state, self.conditional.start_block, InterstateEdge(self.conditional.condition_expr))
        parent.add_edge(guard_state, self.conditional.else_branch, InterstateEdge(self.conditional.condition_else_expr))
        # Connect any end states from the loop's internal state machine to the tail state so they end a
        # loop iteration. Do the same for any continue states, and connect any break states to the end of the loop.
        for node in connect_to_tail:
            parent.add_edge(node, endif_state, InterstateEdge())
        parent.add_edge(self.conditional.else_branch, endif_state, InterstateEdge())
        # Remove the original loop.
        parent.remove_node(self.conditional)
        # parent.remove_node(self.conditional.else_branch)

        sdfg.reset_cfg_list()

        return break_states, continue_states
