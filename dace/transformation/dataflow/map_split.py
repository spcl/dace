# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Map split transformation """

from dace import sdfg as sd, subsets as sbs
from dace.sdfg import nodes, utils as sdutil
from dace.transformation import transformation
from copy import deepcopy
import sympy as sp
from dace.transformation.dataflow import TrivialMapElimination
from dace.properties import CodeBlock


class MapSplit(transformation.SingleStateTransformation):
    """ Map split will detect nested SDFGs where an interstate edge has a
        condition that checks if the loop variable is equal the min/max
        value of the iteration range. If so, the map is peeled and the
        nested SDFG is simplified.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    map_exit = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]

    @classmethod
    def _find_breakpoint(self, sdfg, range_begin, range_end):
        for n in sdfg.nodes():
            for e in sdfg.out_edges(n):
                if e.data.is_unconditional():
                    continue
                cond = e.data.condition_sympy()
                if isinstance(cond, sp.Eq):
                    if ((cond.rhs == range_begin) is True) or ((cond.rhs == range_end) is True):
                        else_edge = next(candidate for candidate in sdfg.out_edges(e.src) if candidate is not e)
                        return cond.rhs, e, else_edge
        return None


    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        nested_sdfg = self.nested_sdfg

        range_begin = map_entry.map.range.min_element()[0]
        range_end = map_entry.map.range.max_element()[0]

        # avoid degenerate map
        if range_begin == range_end:
            return False

        if MapSplit._find_breakpoint(nested_sdfg.sdfg, range_begin, range_end) is not None:
            return True

        return False


    def apply(self, state: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        nested_sdfg = self.nested_sdfg
        map_exit = self.map_exit

        entry_degenerate = deepcopy(map_entry)
        nested_degenerate = nodes.NestedSDFG.from_json(nested_sdfg.to_json(state),
                context={'sdfg_state': state, 'sdfg': sdfg})
        exit_degenerate = deepcopy(map_exit)
        exit_degenerate.map = entry_degenerate.map

        state.add_nodes_from([entry_degenerate, nested_degenerate, exit_degenerate])
        for e in state.in_edges(map_entry):
            state.add_edge(e.src, e.src_conn, entry_degenerate, e.dst_conn, deepcopy(e.data))
        for e in state.out_edges(map_entry):
            state.add_edge(entry_degenerate, e.src_conn, nested_degenerate, e.dst_conn, deepcopy(e.data))
        for e in state.out_edges(nested_sdfg):
            state.add_edge(nested_degenerate, e.src_conn, exit_degenerate, e.dst_conn, deepcopy(e.data))
        for e in state.out_edges(map_exit):
            state.add_edge(exit_degenerate, e.src_conn, e.dst, e.dst_conn, deepcopy(e.data))

        range_begin = map_entry.map.range.min_element()[0]
        range_end = map_entry.map.range.max_element()[0]

        (breaking_point, eq_edge, else_edge) = MapSplit._find_breakpoint(nested_sdfg.sdfg, range_begin, range_end)
        nested_sdfg.sdfg.remove_branch(eq_edge)
        else_edge.data.condition = CodeBlock("1")
        (_, eq_edge, else_edge) = MapSplit._find_breakpoint(nested_degenerate.sdfg, range_begin, range_end)
        nested_degenerate.sdfg.remove_branch(else_edge)
        eq_edge.data.condition = CodeBlock("1")

        entry_degenerate.map.range = sbs.Range([(breaking_point, breaking_point, 1)])
        exit_degenerate.map.range = sbs.Range([(breaking_point, breaking_point, 1)])

        map_elimination = TrivialMapElimination()
        map_elimination.map_entry = entry_degenerate
        map_elimination.apply(state, sdfg)

        if (range_begin == breaking_point) is True:
            map_entry.map.range = sbs.Range([(breaking_point+1, range_end, 1)])
            map_exit.map.range = sbs.Range([(breaking_point+1, range_end, 1)])
        elif (range_end == breaking_point) is True:
            map_entry.map.range = sbs.Range([(range_begin, breaking_point-1, 1)])
            map_exit.map.range = sbs.Range([(range_begin, breaking_point-1, 1)])
