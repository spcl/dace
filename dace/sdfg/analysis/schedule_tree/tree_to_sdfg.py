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
    insert_state_boundaries_to_tree(stree)  # after WAW, before label, etc.

    # TODO: create_state_boundary
    # TODO: create_loop_block
    # TODO: create_conditional_block
    # TODO: create_dataflow_scope

    return result


def insert_state_boundaries_to_tree(stree: tn.ScheduleTreeRoot) -> None:
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
    pass


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
    scope: tn.ControlFlowScope = bnode.parent
    assert scope is not None
    pass
