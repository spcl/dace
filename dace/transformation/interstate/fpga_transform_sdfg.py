# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import networkx as nx

from dace import properties
from dace.transformation import transformation


@properties.make_properties
class FPGATransformSDFG(transformation.MultiStateTransformation):
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

    @classmethod
    def expressions(cls):
        # Match anything
        return [nx.DiGraph()]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Avoid import loops
        from dace.transformation.interstate import FPGATransformState

        # Condition match depends on matching FPGATransformState for each state
        for state_id, state in enumerate(sdfg.nodes()):
            fps = FPGATransformState(sdfg, graph.sdfg_id, -1, {FPGATransformState.state: state_id}, 0)
            if not fps.can_be_applied(sdfg, expr_index, sdfg):
                return False

        return True

    def apply(self, _, sdfg):
        # Avoid import loops
        from dace.transformation.interstate import NestSDFG
        from dace.transformation.interstate import FPGATransformState

        sdfg_id = sdfg.sdfg_id
        nesting = NestSDFG(sdfg, sdfg_id, -1, {}, self.expr_index)
        nesting.promote_global_trans = self.promote_global_trans
        nesting.apply(sdfg, sdfg)

        # The state ID is zero since we applied NestSDFG and have only one state in the new SDFG
        fpga_transform = FPGATransformState(sdfg, sdfg_id, -1, {FPGATransformState.state: 0}, self.expr_index)
        fpga_transform.apply(sdfg, sdfg)
