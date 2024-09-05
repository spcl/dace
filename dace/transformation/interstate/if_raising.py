# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" If raising transformation """

from dace import data as dt, sdfg as sd
from dace.sdfg import InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties

@make_properties
class IfRaising(transformation.MultiStateTransformation):
    """
    Duplicates an if guard and anticipates the evaluation of the condition
    """

    if_guard = transformation.PatternNode(sd.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.if_guard)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if_guard: SDFGState = self.if_guard

        out_edges = graph.out_edges(if_guard)

        if len(out_edges) != 2:
            return False

        if if_guard.is_empty():
            return False

        # check that condition does not depend on computations in the state
        condition_symbols = out_edges[0].data.condition.get_free_symbols()
        _, wset = if_guard.read_and_write_sets()
        if len(condition_symbols.intersection(wset)) != 0:
            return False

        return True


    def apply(self, _, sdfg: sd.SDFG):
        if_guard: SDFGState = self.if_guard

        raised_if_guard = sdfg.add_state('raised_if_guard')
        sdutil.change_edge_dest(sdfg, if_guard, raised_if_guard)

        replica = sd.SDFGState.from_json(if_guard.to_json(), context={'sdfg': sdfg})
        all_block_names = set([s.label for s in sdfg.nodes()])
        replica.label = dt.find_new_name(replica.label, all_block_names)
        sdfg.add_node(replica)

        # move conditional edges up
        if_branch, else_branch = sdfg.out_edges(if_guard)
        sdfg.remove_edge(if_branch)
        sdfg.remove_edge(else_branch)

        sdfg.add_edge(if_guard, if_branch.dst, InterstateEdge(assignments=if_branch.data.assignments))
        sdfg.add_edge(replica, else_branch.dst, InterstateEdge(assignments=else_branch.data.assignments))

        sdfg.add_edge(raised_if_guard, if_guard, InterstateEdge(condition=if_branch.data.condition))
        sdfg.add_edge(raised_if_guard, replica, InterstateEdge(condition=else_branch.data.condition))
