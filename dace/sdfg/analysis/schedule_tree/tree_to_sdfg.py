# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.sdfg.sdfg import SDFG
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def from_schedule_tree(stree: tn.ScheduleTreeRoot) -> SDFG:
    """
    Converts a schedule tree into an SDFG.
    
    :param stree: The schedule tree root to convert.
    :return: An SDFG representing the schedule tree.
    """
    # Set SDFG descriptor repository
    result = SDFG(stree.name, propagate=False)
    result.arg_names = copy.deepcopy(stree.arg_names)
    result._arrays = copy.deepcopy(stree.containers)
    result.constants_prop = copy.deepcopy(stree.constants)
    result.symbols = copy.deepcopy(stree.symbols)

    # TODO: Fill SDFG contents

    return result
