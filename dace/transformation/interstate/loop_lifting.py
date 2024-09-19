# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace import properties
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import DetectLoop


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class LoopLifting(DetectLoop, transformation.MultiStateTransformation):

    def can_be_applied(self, graph: transformation.ControlFlowRegion, expr_index: int, sdfg: transformation.SDFG,
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

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        first_state = self.loop_guard if self.expr_index <= 1 else self.loop_begin
        after = self.exit_state

        loop_info = self.loop_information()

        body = self.loop_body()
        meta = self.loop_meta_states()
        full_body = set(body)
        full_body.update(meta)
        cond_edge = self.loop_condition_edge()
        incr_edge = self.loop_increment_edge()
        inverted = self.expr_index in (2, 3)
        init_edge = self.loop_init_edge()
        exit_edge = self.loop_exit_edge()

        label = 'loop_' + first_state.label
        if loop_info is None:
            itvar = None
            init_expr = None
            incr_expr = None
        else:
            incr_expr = f'{loop_info[0]} = {incr_edge.data.assignments[loop_info[0]]}'
            init_expr = f'{loop_info[0]} = {init_edge.data.assignments[loop_info[0]]}'
            itvar = loop_info[0]

        left_over_assignments = {}
        for k in init_edge.data.assignments.keys():
            if k != itvar:
                left_over_assignments[k] = init_edge.data.assignments[k]
        left_over_incr_assignments = {}
        for k in incr_edge.data.assignments.keys():
            if k != itvar:
                left_over_incr_assignments[k] = incr_edge.data.assignments[k]

        loop = LoopRegion(label, condition_expr=cond_edge.data.condition, loop_var=itvar, initialize_expr=init_expr,
                          update_expr=incr_expr, inverted=inverted, sdfg=sdfg)

        graph.add_node(loop)
        graph.add_edge(init_edge.src, loop,
                       InterstateEdge(condition=init_edge.data.condition, assignments=left_over_assignments))
        graph.add_edge(loop, after, InterstateEdge(assignments=exit_edge.data.assignments))

        loop.add_node(first_state, is_start_block=True)
        for n in full_body:
            if n is not first_state:
                loop.add_node(n)
        added = set()
        for e in graph.all_edges(*full_body):
            if e.src in full_body and e.dst in full_body:
                if not e in added:
                    added.add(e)
                    if e is incr_edge:
                        if left_over_incr_assignments != {}:
                            loop.add_edge(e.src, loop.add_state(label + '_tail'),
                                          InterstateEdge(assignments=left_over_incr_assignments))
                    elif e is cond_edge:
                        e.data.condition = properties.CodeBlock('1')
                        loop.add_edge(e.src, e.dst, e.data)
                    else:
                        loop.add_edge(e.src, e.dst, e.data)

        # Remove old loop.
        for n in full_body:
            graph.remove_node(n)

        sdfg.recheck_using_experimental_blocks()
        sdfg.reset_cfg_list()
