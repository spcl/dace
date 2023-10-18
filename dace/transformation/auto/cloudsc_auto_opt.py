# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List

from dace.sdfg import utils as sdutil, SDFG, SDFGState
from dace.transformation.optimizer import Optimizer

# Transformations
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation import helpers as xfh

from utils.general import save_graph


def loop_to_map_outside_first(sdfg: SDFG,
                              validate: bool = True,
                              validate_all: bool = True,
                              program: str = None) -> SDFG:
    """
    Performs LoopToMap transformation by applying it to the outer loop first

    :param sdfg: The SDFG to work with
    :type sdfg: SDFG
    :param validate: If True, validates the SDFG after all transformations
                     have been applied, defaults to True
    :type validate: bool, optional
    :param validate_all: If True, validates the SDFG after every step, defaults to True
    :type validate_all: bool, optional
    :param program: The name of the program, used for debug saving graphs, is Optional. If not given will not save
    graphs
    :type program: str
    :return: The optimised SDFG
    :rtype: SDFG
    :note: Works by applying LoopToMap to the outermost loop where the
    transformation can be applied. Has not been thoroughly tested yet.
    """

    sdfg.simplify(validate=False, validate_all=validate_all)
    number_of_transformations_performed = 1
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    for s in sdfg.sdfg_list:
        xfh.split_interstate_edges(s)
    if program:
        save_graph(sdfg, program, "after_split_interstate_edges")

    while number_of_transformations_performed > 0:
        outside_loop_transformations = []
        # Get list of all possible transformations
        transformations = [xform for xform in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap])]

        # Find the transformation which is applied to the outermost loop
        for xform in transformations:
            is_outside_loop = True
            # Check if it is the outermoost loop by checking if the loop guard is in any of the loop states of the other
            # found transformations. This could in theory find several outermost loops
            for other_form in transformations:
                if other_form != xform:
                    other_states: List[SDFGState] = list(
                        sdutil.dfs_conditional(sdfg.sdfg_list[other_form.sdfg_id], [other_form.loop_begin],
                                               lambda _, c: c is not other_form.loop_guard))
                    if xform.loop_guard in other_states:
                        is_outside_loop = False
            if is_outside_loop:
                outside_loop_transformations.append(xform)

        # Apply the first of the found transformations
        number_of_transformations_performed = min(len(outside_loop_transformations), 1.0)
        if len(outside_loop_transformations) > 0:
            xform = outside_loop_transformations[0]
            # Apply for the LoopToMap transformations does not use the first argument, thus None is passed here
            xform.apply(None, sdfg.sdfg_list[xform.sdfg_id])
    if program is not None:
        save_graph(sdfg, program, "after_outer_loop_to_map")
    sdfg.validate()
    sdfg.apply_transformations_repeated([RefineNestedAccess], validate=validate, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_outer_refine_nested_access")

    return sdfg
