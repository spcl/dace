# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various analyses concerning LopoRegions, and utility functions to get information about LoopRegions for other passes.
"""

from typing import Dict, Optional, Union
from dace.frontend.python import astutils

import sympy

from dace import symbolic
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion
from dace.subsets import Range, SubsetUnion, intersects


def get_loop_end(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region to identify the end value of the iteration variable under normal loop termination (no break).
    """
    if loop.loop_variable is None or loop.loop_variable == '':
        return None
    end: Optional[symbolic.SymbolicType] = None
    a = sympy.Wild('a')
    condition = symbolic.pystr_to_symbolic(loop.loop_condition.as_string)
    itersym = symbolic.pystr_to_symbolic(loop.loop_variable)
    match = condition.match(itersym < a)
    if match:
        end = match[a] - 1
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            end = match[a] + 1
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            end = match[a]
    return end


def get_init_assignment(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region's init statement to identify the exact init assignment expression.
    """
    init_stmt = loop.init_statement
    if init_stmt is None:
        return None

    init_codes_list = init_stmt.code if isinstance(init_stmt.code, list) else [init_stmt.code]
    assignments: Dict[str, str] = {}
    for code in init_codes_list:
        visitor = astutils.FindAssignment()
        visitor.visit(code)
        if visitor.multiple:
            return None
        for assign in visitor.assignments:
            if assign in assignments:
                return None
            assignments[assign] = visitor.assignments[assign]

    if loop.loop_variable in assignments:
        return symbolic.pystr_to_symbolic(assignments[loop.loop_variable])

    return None


def get_update_assignment(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region's update statement to identify the exact update assignment expression.
    """
    update_stmt = loop.update_statement
    if update_stmt is None:
        return None

    update_codes_list = update_stmt.code if isinstance(update_stmt.code, list) else [update_stmt.code]
    assignments: Dict[str, str] = {}
    for code in update_codes_list:
        visitor = astutils.FindAssignment()
        visitor.visit(code)
        if visitor.multiple:
            return None
        for assign in visitor.assignments:
            if assign in assignments:
                return None
            assignments[assign] = visitor.assignments[assign]

    if loop.loop_variable in assignments:
        return symbolic.pystr_to_symbolic(assignments[loop.loop_variable])

    return None


def get_loop_stride(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    update_assignment = get_update_assignment(loop)
    if update_assignment:
        return update_assignment - symbolic.pystr_to_symbolic(loop.loop_variable)
    return None


def _loop_read_intersects_loop_write(loop: LoopRegion, write_subset: Union[SubsetUnion, Range],
                                     read_subset: Union[SubsetUnion, Range], update: sympy.Basic) -> bool:
    """
    Check if a write subset intersects a read subset after being offset by the loop stride. The offset is performed
    based on the symbolic loop update assignment expression.
    """
    offset = update - symbolic.symbol(loop.loop_variable)
    offset_list = []
    for i in range(write_subset.dims()):
        if loop.loop_variable in write_subset.get_free_symbols_by_indices([i]):
            offset_list.append(offset)
        else:
            offset_list.append(0)
    offset_write = write_subset.offset_new(offset_list, True)
    return intersects(offset_write, read_subset)

def get_loop_carry_dependencies(loop: LoopRegion) -> Optional[Dict[Memlet, Memlet]]:
    """
    Compute loop carry dependencies.
    :return: A dictionary mapping loop reads to writes in the same loop, from which they may carry a RAW dependency.
             None if the loop cannot be analyzed.
    """
    update_assignment = None
    raw_deps: Dict[Memlet, Memlet] = dict()
    for data in loop.possible_reads:
        if not data in loop.possible_writes:
            continue

        input = loop.possible_reads[data]
        read_subset = input.src_subset or input.subset
        if loop.loop_variable and loop.loop_variable in input.free_symbols:
            # If the iteration variable is involved in an access, we need to first offset it by the loop
            # stride and then check for an overlap/intersection. If one is found after offsetting, there
            # is a RAW loop carry dependency.
            output = loop.possible_writes[data]
            # Get and cache the update assignment for the loop.
            if update_assignment is None:
                update_assignment = get_update_assignment(loop)
                if update_assignment is None:
                    return None

            if isinstance(output.subset, SubsetUnion):
                if any([_loop_read_intersects_loop_write(loop, s, read_subset, update_assignment)
                        for s in output.subset.subset_list]):
                    raw_deps[input] = output
            elif _loop_read_intersects_loop_write(loop, output.subset, read_subset, update_assignment):
                raw_deps[input] = output
        else:
            # Check for basic overlaps/intersections in RAW loop carry dependencies, when there is no
            # iteration variable involved.
            output = loop.possible_writes[data]
            if intersects(output.subset, read_subset):
                raw_deps[input] = output
    return raw_deps
