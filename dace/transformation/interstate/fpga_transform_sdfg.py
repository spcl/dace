# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import networkx as nx

from dace import registry, properties
from dace.transformation import transformation


@registry.autoregister
@properties.make_properties
class FPGATransformSDFG(transformation.Transformation):
    """ Implements the FPGATransformSDFG transformation, which takes an entire
        SDFG and transforms it into an FPGA-capable SDFG. """

    promote_global_trans = properties.Property(
        dtype=bool,
        default=True,
        desc="If True, transient arrays that are fully internal are pulled out so "
        "that they can be allocated on the host.")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        # Match anything
        return [nx.DiGraph()]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        # Avoid import loops
        from dace.transformation.interstate import FPGATransformState

        # Condition match depends on matching FPGATransformState for each state
        for state_id, state in enumerate(sdfg.nodes()):
            candidate = {FPGATransformState._state: state_id}
            if not FPGATransformState.can_be_applied(sdfg, candidate, expr_index, sdfg):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg):
        # Avoid import loops
        from dace.transformation.interstate import NestSDFG
        from dace.transformation.interstate import FPGATransformState

        sdfg_id = sdfg.sdfg_id
        nesting = NestSDFG(sdfg_id, -1, {}, self.expr_index)
        nesting.promote_global_trans = self.promote_global_trans
        nesting.apply(sdfg)

        fpga_transform = FPGATransformState(sdfg_id, -1, {FPGATransformState._state: 0}, self.expr_index)
        fpga_transform.apply(sdfg)
