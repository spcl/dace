# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple

import sympy

from dace import SDFG, properties, symbolic, transformation
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion
from dace.subsets import Range, SubsetUnion
from dace.transformation import pass_pipeline as ppl


class FindAssignment(ast.NodeVisitor):

    assignments: Dict[str, str]
    multiple: bool

    def __init__(self):
        self.assignments = {}
        self.multiple = False

    def visit_Assign(self, node: ast.Assign) -> Any:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                if tgt.id in self.assignments:
                    self.multiple = True
                self.assignments[tgt.id] = ast.unparse(node.value)
        return self.generic_visit(node)


def get_loop_end(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region to identify the end value of the iteration variable under normal loop termination (no break).
    """
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
        visitor = FindAssignment()
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
        visitor = FindAssignment()
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
