# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG, ControlFlowRegion
from dace.sdfg.state import SDFGState
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from enum import Enum, auto
from typing import Optional, Sequence


class StateBoundaryBehavior(Enum):
    STATE_TRANSITION = auto()  #: Creates multiple states with a state transition
    EMPTY_MEMLET = auto()  #: Happens-before empty memlet edges in the same state


def from_schedule_tree(stree: tn.ScheduleTreeRoot,
                       state_boundary_behavior: StateBoundaryBehavior = StateBoundaryBehavior.STATE_TRANSITION) -> SDFG:
    """
    Converts a schedule tree into an SDFG.
    
    :param stree: The schedule tree root to convert.
    :param state_boundary_behavior: Sets the behavior upon encountering a state boundary (e.g., write-after-write).
                                    See the ``StateBoundaryBehavior`` enumeration for more details.
    :return: An SDFG representing the schedule tree.
    """
    # Set SDFG descriptor repository
    result = SDFG(stree.name, propagate=False)
    result.arg_names = copy.deepcopy(stree.arg_names)
    result._arrays = copy.deepcopy(stree.containers)
    result.constants_prop = copy.deepcopy(stree.constants)
    result.symbols = copy.deepcopy(stree.symbols)

    # TODO: Fill SDFG contents
    stree = insert_state_boundaries_to_tree(stree)  # after WAW, before label, etc.

    # TODO: create_state_boundary
    # TODO: create_loop_block
    # TODO: create_conditional_block
    # TODO: create_dataflow_scope

    return result


def insert_state_boundaries_to_tree(stree: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
    """
    Inserts StateBoundaryNode objects into a schedule tree where more than one SDFG state would be necessary.
    Operates in-place on the given schedule tree.
    
    This happens when there is a:
      * write-after-write dependency;
      * write-after-read dependency that cannot be fulfilled via memlets;
      * control flow block (for/if); or
      * otherwise before a state label (which means a state transition could occur, e.g., in a gblock)

    :param stree: The schedule tree to operate on.
    """

    # Simple boundary node inserter for control flow blocks and state labels
    class SimpleStateBoundaryInserter(tn.ScheduleNodeTransformer):

        def visit_scope(self, scope: tn.ScheduleTreeScope):
            if isinstance(scope, tn.ControlFlowScope):
                return [tn.StateBoundaryNode(), self.generic_visit(scope)]
            return self.generic_visit(scope)

        def visit_StateLabel(self, node: tn.StateLabel):
            return [tn.StateBoundaryNode(), self.generic_visit(node)]

    # First, insert boundaries around labels and control flow
    stree = SimpleStateBoundaryInserter().visit(stree)

    # TODO: Insert boundaries after unmet memory dependencies
    # TODO: Implement generic methods that get input/output memlets for stree scopes and nodes
    # TODO: Implement method that searches for a memlet in a dictionary of memlets (even if that memlet
    #       is a subset of a dictionary key) and returns that key. If intersection indeterminate, assume
    #       intersects and replace key with union key. Implement in dace.sdfg.memlet_utils.

    return stree


#############################################################################
# SDFG content creation functions


def create_state_boundary(bnode: tn.StateBoundaryNode, sdfg_region: ControlFlowRegion, state: SDFGState,
                          behavior: StateBoundaryBehavior) -> SDFGState:
    """
    Creates a boundary between two states

    :param bnode: The state boundary node to generate.
    :param sdfg_region: The control flow block in which to generate the boundary (e.g., SDFG).
    :param state: The last state prior to this boundary.
    :param behavior: The state boundary behavior with which to create the boundary.
    :return: The newly created state.
    """
    # TODO: Some boundaries (control flow, state labels with goto) could not be fulfilled with every
    #       behavior. Fall back to state transition in that case.
    scope: tn.ControlFlowScope = bnode.parent
    assert scope is not None
    pass
