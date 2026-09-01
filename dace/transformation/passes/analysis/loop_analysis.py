# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various analyses concerning LopoRegions, and utility functions to get information about LoopRegions for other passes.
"""

from typing import Dict, Optional
from dace.frontend.python import astutils

import sympy

from dace import symbolic
from dace.sdfg.state import LoopRegion


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


def _provably_le(a: symbolic.SymbolicType, b: symbolic.SymbolicType) -> bool:
    """Prove ``a <= b`` SOUNDLY, returning ``False`` when it cannot be decided (never a guess).

    Beyond a concrete numeric verdict and sympy's own non-positivity engine (which respects the
    codebase's non-negative symbols), this proves the ``Min`` / ``Max`` clamp shape a range split
    leaves behind: ``Min(..., t) <= t`` for any of its own args ``t``, ``t <= Max(..., t)``, and the
    combined ``Min(..., t) <= t <= Max(..., t)`` sharing a term ``t``.
    """
    diff = symbolic.simplify(a - b)
    if diff.is_number:
        return bool(diff <= 0)
    if diff.is_nonpositive:  # sympy assumption engine; None (undecided) is falsy -> not proven
        return True
    a_min = a.args if isinstance(a, sympy.Min) else ()
    b_max = b.args if isinstance(b, sympy.Max) else ()
    if b in a_min:  # a = Min(..., b, ...) <= b
        return True
    if a in b_max:  # a <= Max(..., a, ...) = b
        return True
    return bool(a_min and b_max and (set(a_min) & set(b_max)))  # a <= shared t <= b


def loop_provably_at_most_one_iteration(loop: LoopRegion) -> bool:
    """Whether ``loop`` provably runs at most once (zero or one iterations).

    Such a loop carries no cross-iteration dependence by construction, so it is trivially DOALL --
    a ``LoopToMap`` can map it without any dependence analysis. Restricted to the unit ascending
    stride so the inclusive trip count is exactly ``end - start + 1``; the loop runs at most once iff
    ``end <= start``. Conservative: returns ``False`` whenever the bound cannot be proven.
    """
    start = get_init_assignment(loop)
    end = get_loop_end(loop)  # inclusive
    step = get_loop_stride(loop)
    if start is None or end is None or step is None or symbolic.simplify(step) != 1:
        return False
    return _provably_le(symbolic.simplify(end), symbolic.simplify(start))
