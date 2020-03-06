""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import networkx as nx

from dace import registry
from dace.transformation import pattern_matching


@registry.autoregister
class FPGATransformSDFG(pattern_matching.Transformation):
    """ Implements the FPGATransformSDFG transformation, which takes an entire
        SDFG and transforms it into an FPGA-capable SDFG. """
    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        # Match anything
        return [nx.DiGraph()]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # Avoid import loops
        from dace.transformation.interstate import FPGATransformState

        # Condition match depends on matching FPGATransformState for each state
        for state_id, state in enumerate(sdfg.nodes()):
            candidate = {FPGATransformState._state: state_id}
            if not FPGATransformState.can_be_applied(sdfg, candidate,
                                                     expr_index, sdfg):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg):
        # Avoid import loops
        from dace.transformation.interstate import NestSDFG
        from dace.transformation.interstate import FPGATransformState

        sdfg_id = sdfg.sdfg_list.index(sdfg)
        nesting = NestSDFG(sdfg_id, -1, {}, self.expr_index)
        nesting.promote_global_trans = True
        nesting.apply(sdfg)

        fpga_transform = FPGATransformState(sdfg_id, -1,
                                            {FPGATransformState._state: 0},
                                            self.expr_index)
        fpga_transform.apply(sdfg)
