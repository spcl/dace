""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import copy
import itertools
import networkx as nx

import dace
from dace import data, memlet, types, sdfg as sd, subsets, symbolic
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching


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

        fpga_transform = FPGATransformState(
            sdfg_id, -1, {FPGATransformState._state: 0}, self.expr_index)
        fpga_transform.apply(sdfg)


pattern_matching.Transformation.register_stateflow_pattern(FPGATransformSDFG)
