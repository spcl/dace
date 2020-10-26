# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Conditional branch detection transformation """

from dace import sdfg as sd
from dace.transformation import transformation


class DetectBranch(transformation.Transformation):
    """ Detects a conditional branch construct from an SDFG. """

    _branch_guard = sd.SDFGState()
    _first_branch = sd.SDFGState()
    _second_branch = sd.SDFGState()

    @staticmethod
    def expressions():
        # Any subgraph with at least 2 unconnected children.
        sdfg = sd.SDFG('_')
        sdfg.add_nodes_from([
            DetectBranch._branch_guard,
            DetectBranch._first_branch,
            DetectBranch._second_branch,
        ])
        sdfg.add_edge(
            DetectBranch._branch_guard,
            DetectBranch._first_branch,
            sd.InterstateEdge()
        )
        sdfg.add_edge(
            DetectBranch._branch_guard,
            DetectBranch._second_branch,
            sd.InterstateEdge()
        )
        return [sdfg]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        guard = graph.node(candidate[DetectBranch._branch_guard])
        out_edges = graph.out_edges(guard)

        # A branch must have at least two outgoing edges.
        if len(out_edges) < 2:
            return False

        # All outgoing edges must not have assignments and must all be
        # conditional.
        if any(len(e.data.assignments) > 0 or e.data.is_unconditional() for e in out_edges):
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return (
            'Conditional branch on ' +
            graph.node(candidate[DetectBranch._branch_guard]).label
        )

    def apply(self, sdfg):
        pass
