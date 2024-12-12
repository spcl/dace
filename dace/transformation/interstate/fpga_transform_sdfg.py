# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import networkx as nx

from dace import properties
from dace.sdfg.sdfg import SDFG
from dace.transformation import transformation


@properties.make_properties
@transformation.explicit_cf_compatible
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

    def can_be_applied(self, graph, expr_index, sdfg: SDFG, permissive=False):
        # Avoid import loops
        from dace.transformation.interstate import FPGATransformState

        # Condition match depends on matching FPGATransformState for each state
        for state in sdfg.states():
            fps = FPGATransformState()
            fps.setup_match(sdfg, state.parent_graph.cfg_id, -1, {FPGATransformState.state: state.block_id}, 0)
            if not fps.can_be_applied(state.parent_graph, expr_index, sdfg):
                return False

        return True

    def apply(self, _, sdfg: SDFG):
        # Avoid import loops
        from dace.transformation.interstate import NestSDFG
        from dace.transformation.interstate import FPGATransformState

        cfg_id = sdfg.cfg_id
        nesting = NestSDFG()
        nesting.setup_match(sdfg, cfg_id, -1, {}, self.expr_index)
        nesting.promote_global_trans = self.promote_global_trans
        nesting.apply(sdfg, sdfg)

        # The state ID is zero since we applied NestSDFG and have only one state in the new SDFG
        fpga_transform = FPGATransformState()
        fpga_transform.setup_match(sdfg, cfg_id, -1, {FPGATransformState.state: 0}, self.expr_index)
        fpga_transform.apply(sdfg, sdfg)
