# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module initializes the inter-state transformations package."""

from .block_fusion import BlockFusion
from .condition_fusion import ConditionFusion
from .condition_map_interchange import ConditionMapInterchange
from .state_fusion import StateFusion
from .state_fusion_with_happens_before import StateFusionExtended
from .state_elimination import (EndStateElimination, StartStateElimination, StateAssignElimination,
                                SymbolAliasPromotion, HoistState)
from .fpga_transform_state import FPGATransformState
from .fpga_transform_sdfg import FPGATransformSDFG
from .gpu_transform_sdfg import GPUTransformSDFG
from .sdfg_nesting import NestSDFG, InlineSDFG, InlineTransients, RefineNestedAccess
from .loop_unroll import LoopUnroll
from .loop_overwrite_elimination import LoopOverwriteElimination
from .loop_peeling import LoopPeeling
from .loop_to_map import LoopToMap
from .move_loop_into_map import MoveLoopIntoMap
from .trivial_loop_elimination import TrivialLoopElimination
from .multistate_inline import InlineMultistateSDFG
from .move_assignment_outside_if import MoveAssignmentOutsideIf
