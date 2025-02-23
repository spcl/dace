# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace import properties
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import DetectLoop


@properties.make_properties
@transformation.explicit_cf_compatible
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
        first_state = self.first_loop_block
        after = self.exit_state

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
        if incr_edge is not None:
            for k in incr_edge.data.assignments.keys():
                if k != itvar:
                    left_over_incr_assignments[k] = incr_edge.data.assignments[k]

        if inverted and incr_edge is cond_edge:
            update_before_condition = False
        else:
            update_before_condition = True

        loop = LoopRegion(label, condition_expr=cond_edge.data.condition, loop_var=itvar, initialize_expr=init_expr,
                          update_expr=incr_expr, inverted=inverted, sdfg=sdfg,
                          update_before_condition=update_before_condition)

        graph.add_node(loop)
        graph.add_edge(init_edge.src, loop,
                       InterstateEdge(condition=init_edge.data.condition, assignments=left_over_assignments))
        graph.add_edge(loop, after, InterstateEdge(assignments=exit_edge.data.assignments))

        loop.add_node(first_state, is_start_block=True)
        added = set()
        for e in graph.all_edges(*full_body):
            if e.src in full_body and e.dst in full_body:
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
                                    incr_graph.add_edge(incr_graph.add_state(label + '_post_incr_start',
                                                                            is_start_block=True),
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
                            loop.add_edge(e.src, e.dst, e.data)
                    else:
                        loop.add_edge(e.src, e.dst, e.data)

        # Remove old loop.
        for n in full_body:
            graph.remove_node(n)

        sdfg.root_sdfg.using_explicit_control_flow = True
        sdfg.reset_cfg_list()
