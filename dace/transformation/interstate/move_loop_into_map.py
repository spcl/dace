# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Moves a loop around a map into the map """

import dace.transformation.helpers as helpers
from dace.sdfg.scope import ScopeTree
from dace import nodes, registry, sdfg as sd, symbolic, symbol
from dace.properties import CodeBlock
from dace.sdfg import nodes, propagation
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)


def fold(memlet_subset_ranges, itervar, lower, upper):
    return [
        (r[0].replace(symbol(itervar), lower), r[1].replace(symbol(itervar), upper), r[2])
        for r in memlet_subset_ranges
    ]

def offset(memlet_subset_ranges, value):
    return (
        memlet_subset_ranges[0] + value,
        memlet_subset_ranges[1] + value,
        memlet_subset_ranges[2]
        )


class MoveLoopIntoMap(DetectLoop, transformation.MultiStateTransformation):
    """
    Moves a loop around a map into the map
    """

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin
        after: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range, and stride
        itervar, (start, end, step), _ = find_for_loop(sdfg, guard, body)

        if step not in [-1, 1]:
            return False

        # Check if body contains exactly one map
        is_map = [isinstance(node, nodes.MapEntry) for node in body.nodes()]
        if is_map.count(True) != 1:
            return False

        #TODO: Add test that map is independant of itervar!

        for node in body.nodes():
            if isinstance(node, nodes.AccessNode):
                if body.in_edges(node).count(True) > 1:
                    return False
                if body.out_edges(node).count(True) > 1:
                    return False

        return True
   
        
    def apply(self, _, sdfg: sd.SDFG):
        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin
        after: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range, and stride
        itervar, (start, end, step), _ = find_for_loop(sdfg, guard, body)

        forward_loop = step > 0
        if forward_loop:
            lower_loop_bound = start
            upper_loop_bound = end
        else:
            lower_loop_bound = end
            upper_loop_bound = start
        
        for node in body.nodes():
            if isinstance(node, nodes.MapEntry):
                map_entry = node
            if isinstance(node, nodes.MapExit):
                map_exit = node

        # save old in and out memlets { name : ranges }
        old_in_memlets = { edge.data.data : edge.data.subset.ranges for edge in body.out_edges(map_entry) }
        old_out_memlets = { edge.data.data : edge.data.subset.ranges for edge in body.in_edges(map_exit) }

        # nest map's content in sdfg
        map_subgraph = body.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        nsdfg = helpers.nest_state_subgraph(sdfg, body, map_subgraph)
        nstate = nsdfg.sdfg.nodes()[0]

        # correct the memlets going into the nsdfg
        for edge in body.out_edges(map_entry):
            edge.data.subset.ranges = fold(edge.data.subset.ranges, itervar, lower_loop_bound, upper_loop_bound)

        # correct the memlets coming from the nsdfg
        for edge in body.in_edges(map_exit):
            edge.data.subset.ranges = fold(edge.data.subset.ranges, itervar, lower_loop_bound, upper_loop_bound)
        
        # correct the input and output memlets inside the nsdfg
        for access_node in nstate.nodes():
            if isinstance(access_node, nodes.AccessNode):
                # input memlets
                for internal_edge in nstate.out_edges(access_node):
                    new_range = []
                    if internal_edge.data.data in old_in_memlets:
                        for old_r, new_r in zip(old_in_memlets[internal_edge.data.data], internal_edge.data.subset.ranges):
                            if any(symbolic.issymbolic(x) and symbol(itervar) in x.free_symbols for x in old_r):
                                new_range.append(offset(new_r, symbol(itervar) - lower_loop_bound))
                            else:
                                new_range.append(new_r)
                        internal_edge.data.subset.ranges = new_range
                # output memlets
                for internal_edge in nstate.in_edges(access_node):
                    new_range = []
                    if internal_edge.data.data in old_out_memlets:
                        for old_r, new_r in zip(old_out_memlets[internal_edge.data.data], internal_edge.data.subset.ranges):
                            if any(symbolic.issymbolic(x) and symbol(itervar) in x.free_symbols for x in old_r):
                                new_range.append(offset(new_r, symbol(itervar) - lower_loop_bound))
                            else:
                                new_range.append(new_r)
                        internal_edge.data.subset.ranges = new_range

        # replicate loop in nested sdfg
        nsdfg.sdfg.add_loop(
            before_state = None,
            loop_state = nsdfg.sdfg.nodes()[0],
            loop_end_state = None,
            after_state = None,
            loop_var = itervar,
            initialize_expr = f'{start}',
            condition_expr = f'{itervar} <= {end}' if forward_loop else f'{itervar} >= {end}',
            increment_expr = f'{itervar} + {step}' if forward_loop else f'{itervar} - {abs(step)}'
        )

        # remove outer loop
        for body_inedge in sdfg.in_edges(body):
            sdfg.remove_edge(body_inedge)
        for body_outedge in sdfg.out_edges(body):
            sdfg.remove_edge(body_outedge)
        for guard_inedge in sdfg.in_edges(guard):
            guard_inedge.data.assignments = {}
            sdfg.add_edge(guard_inedge.src, body, guard_inedge.data)
            sdfg.remove_edge(guard_inedge)
        for guard_outedge in sdfg.out_edges(guard):
            guard_outedge.data.condition = CodeBlock("1")
            sdfg.add_edge(body, guard_outedge.dst, guard_outedge.data)
            sdfg.remove_edge(guard_outedge)
        sdfg.remove_node(guard)
        del nsdfg.symbol_mapping[itervar]
        if itervar in sdfg.symbols:
            del sdfg.symbols[itervar]

        # propagate scope for correct volumes
        scope_tree = ScopeTree(map_entry, map_exit)
        scope_tree.parent = ScopeTree(None, None)
        propagation.propagate_memlets_scope(sdfg, body, scope_tree)