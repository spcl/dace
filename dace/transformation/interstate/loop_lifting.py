# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy
from typing import Optional
from dace import properties
from dace.frontend.python import astutils
from dace.sdfg.sdfg import SDFG, ControlFlowBlock, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import DetectLoop


@properties.make_properties
@transformation.explicit_cf_compatible
class LoopLifting(DetectLoop, transformation.MultiStateTransformation):

    def can_be_applied(self,
                       graph: transformation.ControlFlowRegion,
                       expr_index: int,
                       sdfg: transformation.SDFG,
                       permissive: bool = False) -> bool:
        # Check loop detection with permissive = True, which allows loops where no iteration variable could be detected.
        # We want this to detect while loops.
        if not super().can_be_applied(graph, expr_index, sdfg, permissive=True):
            return False

        # Check that there's a condition edge, that's the only requirement to lift it into loop.
        cond_edge = self.loop_condition_edge()
        if not cond_edge or cond_edge.data.condition is None:
            return False
        return True

    def _get_to_execute_before(self) -> Optional[ControlFlowBlock]:
        # Some loops may have a state that is executed before the condition check - specifically, this is the case for
        # the guard state in for/while loops (expr. index 0/1), and for the begin state in a self-loop (expr. index 6).
        if self.expr_index in (0, 1) and len(self.loop_guard.nodes()) > 0:
            return self.loop_guard
        elif self.expr_index == 4 and len(self.loop_begin.nodes()) > 0:
            return self.loop_begin
        return None

    def _get_to_guard_before_exec(self) -> Optional[ControlFlowBlock]:
        # Some loops have a state executed at the end of the loop body that is only executed if the condition still
        # holds. This is the case for the latch state in loops with explicit test blocks before the latch
        # (expr. index 5/6/7).
        if self.expr_index in (5, 6, 7):
            return self.loop_latch
        return None

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        first_state = self.first_loop_block
        after = self.exit_state

        # Gather loop meta information.
        loop_info = self.loop_information()
        body = self.loop_body()
        meta = self.loop_meta_states()
        full_body = set(body)
        full_body.update(meta)
        cond_edge = self.loop_condition_edge()
        incr_edge = self.loop_increment_edge()
        inverted = self.inverted
        init_edge = self.loop_init_edge()
        exit_edge = self.loop_exit_edge()
        guard_before_exec = self._get_to_guard_before_exec()
        exec_before_loop = self._get_to_execute_before()

        # Extract the incrementation expression.
        label = 'loop_' + first_state.label
        if loop_info is None:
            itvar = None
            init_expr = None
            incr_expr = None
        else:
            incr_expr = f'{loop_info[0]} = {incr_edge.data.assignments[loop_info[0]]}'
            init_expr = f'{loop_info[0]} = {init_edge.data.assignments[loop_info[0]]}'
            itvar = loop_info[0]

        # Extract any assignments that may need to be preserved explicitly after the state machine is lifted and edges
        # are removed.
        left_over_assignments = {}
        for k in init_edge.data.assignments.keys():
            if k != itvar:
                left_over_assignments[k] = init_edge.data.assignments[k]
        left_over_incr_assignments = {}
        if incr_edge is not None:
            for k in incr_edge.data.assignments.keys():
                if k != itvar:
                    left_over_incr_assignments[k] = incr_edge.data.assignments[k]

        if inverted and incr_edge is cond_edge:
            update_before_condition = False
        else:
            update_before_condition = True

        loop = LoopRegion(label,
                          condition_expr=cond_edge.data.condition,
                          loop_var=itvar,
                          initialize_expr=init_expr,
                          update_expr=incr_expr,
                          inverted=inverted,
                          sdfg=sdfg,
                          update_before_condition=update_before_condition)

        # First state is added explicitly to mark the start of the region.
        loop.add_node(first_state, is_start_block=True)

        # If the loop has a latch state that needs to be guarded by the condition before execution, add it with a
        # conditional block.
        latch_cond = None
        if guard_before_exec is not None:
            latch_cond = ConditionalBlock(label + '_latch_guard')
            loop.add_node(latch_cond)
            latch_branch = ConditionalBlock(label + '_latch_guard_if')
            latch_cond.add_branch(properties.CodeBlock(cond_edge.data.condition.code), latch_branch)
            latch_branch.add_node(guard_before_exec, is_start_block=True)

        added = set()
        for e in graph.all_edges(*full_body):
            if e.src in full_body and e.dst in full_body:
                src = e.src if e.src is not guard_before_exec else latch_cond
                dst = e.dst if e.dst is not guard_before_exec else latch_cond
                if not e in added:
                    added.add(e)
                    if e is incr_edge:
                        if left_over_incr_assignments != {}:
                            assignments = left_over_incr_assignments
                            dst = e.dst
                            if e.dst is first_state:
                                if not update_before_condition:
                                    left_over_incr_cond_region = ConditionalBlock(label + '_post_incr_conditional')
                                    incr_graph = ControlFlowRegion(label + '_post_incr')
                                    left_over_incr_cond_region.add_branch(cond_edge.data.condition, incr_graph)
                                    incr_graph.add_edge(
                                        incr_graph.add_state(label + '_post_incr_start', is_start_block=True),
                                        incr_graph.add_state(label + '_post_incr_end'),
                                        InterstateEdge(assignments=left_over_incr_assignments))
                                    dst = left_over_incr_cond_region
                                    assignments = {}
                                else:
                                    dst = loop.add_state(label + '_tail')
                            loop.add_edge(e.src, dst, InterstateEdge(assignments=assignments))
                    elif e is cond_edge:
                        if not inverted:
                            e.data.condition = properties.CodeBlock('1')
                            loop.add_edge(src, dst, e.data)
                    else:
                        loop.add_edge(src, dst, e.data)

        # Remove old loop.
        for n in full_body:
            graph.remove_node(n)

        # If the loop has a guard state that is executed before the loop condition is checked for the first time, we
        # need to construct a conditional check that executes the guard state ONLY, if the loop condition initially
        # does not hold. If the loop condition holds (i.e., else), the loop is executed instead.
        to_connect = None
        if exec_before_loop is not None:
            loop_guard_conditional = ConditionalBlock(label + '_guard_conditional')
            graph.add_node(loop_guard_conditional)
            new_init_edge = InterstateEdge(condition=init_edge.data.condition, assignments=left_over_assignments)
            if loop_info is not None:
                new_init_edge.assignments[loop_info[0]] = init_edge.data.assignments[loop_info[0]]
            graph.add_edge(init_edge.src, loop_guard_conditional, new_init_edge)

            loop_not_executed = ControlFlowRegion(label + '_loop_not_executed')
            negative_cond = astutils.negate_expr(cond_edge.data.condition.code[0])
            loop_guard_conditional.add_branch(properties.CodeBlock([negative_cond]), loop_not_executed)
            exec_before_loop_copy = deepcopy(exec_before_loop)
            loop_not_executed.add_node(exec_before_loop_copy, is_start_block=True)

            loop_executed_branch = ControlFlowRegion(label + '_loop_executed')
            loop_guard_conditional.add_branch(None, loop_executed_branch)
            loop_executed_branch.add_node(loop, is_start_block=True)

            to_connect = loop_guard_conditional
        else:
            graph.add_node(loop)
            graph.add_edge(init_edge.src, loop,
                           InterstateEdge(condition=init_edge.data.condition, assignments=left_over_assignments))
            to_connect = loop

        # Connect the loop to everything after the loop.
        graph.add_edge(to_connect, after, InterstateEdge(assignments=exit_edge.data.assignments))

        sdfg.root_sdfg.using_explicit_control_flow = True
        sdfg.reset_cfg_list()
