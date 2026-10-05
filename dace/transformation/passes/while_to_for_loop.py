# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turns while loops that step a counter by a constant into for loops (with an init and an update statement)."""
from typing import Any, Dict, Optional, Set, Tuple

import sympy

from dace import SDFG, symbolic
from dace.properties import CodeBlock
from dace.sdfg.state import (AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, LoopRegion, ReturnBlock)
from dace.transformation import pass_pipeline as ppl, transformation

_DIRECTIONS = {
    sympy.StrictLessThan: (1, True),
    sympy.LessThan: (1, False),
    sympy.StrictGreaterThan: (-1, True),
    sympy.GreaterThan: (-1, False),
}


@transformation.explicit_cf_compatible
class WhileToForLoop(ppl.Pass):
    """
    Turns while loops into for loops, if their condition compares a counter with a loop-invariant bound and every
    iteration steps the counter once by a constant, on an interstate edge inside the loop body.

    The step becomes the update statement of the loop, the value the counter has when the loop is entered becomes its
    init statement, and the blocks after the step in the body read ``counter + step`` instead of the counter. The
    counter has the same values in and after the loop as before. Loops are converted to the form
    ``for (i = start; i < end; i = i + step)`` (or ``i > end`` and ``i = i - step`` if counting down), which analyses
    of for loops (e.g., automatic differentiation) expect.

    The value of the counter when the loop is entered must be assigned on a chain of edges that leads to the loop
    (through states with a single predecessor).
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.InterstateEdges | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.CFG | ppl.Modifies.InterstateEdges)

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """
        :return: The number of converted loops, or None if no loop was converted.
        """
        converted = 0
        for region in list(sdfg.all_control_flow_regions(recursive=True)):
            if isinstance(region, LoopRegion) and self.convert(region):
                converted += 1
        return converted or None

    def convert(self, loop: LoopRegion) -> bool:
        """
        Converts a while loop into a for loop, if possible.

        :param loop: The loop to convert.
        :return: True if the loop was converted.
        """
        if (loop.init_statement is not None or loop.update_statement is not None or loop.loop_variable or loop.inverted
                or loop.has_continue or loop.has_break):
            return False
        if any(isinstance(block, ReturnBlock) for block in loop.all_control_flow_blocks()):
            return False
        if loop.has_cycles():
            return False

        # Symbols (and data) whose value may change in the loop
        variant = {symbolic.symbol(name) for name in _assigned_symbols(loop) | set(loop.sdfg.arrays.keys())}
        comparison = _comparison(loop.loop_condition)
        if comparison is None:
            return False
        counter, bound, sign, strict = comparison
        if bound.free_symbols & variant:
            return False

        # The counter is stepped on exactly one edge of the loop, which every iteration takes once
        steps = [edge for edge in loop.all_interstate_edges() if counter in edge.data.assignments]
        if len(steps) != 1 or steps[0] not in loop.edges():
            return False
        if any(
                isinstance(region, LoopRegion) and region.loop_variable == counter
                for region in loop.all_control_flow_regions() if region is not loop):
            return False
        step_edge = steps[0]
        counter_symbol = symbolic.symbol(counter)
        step = symbolic.pystr_to_symbolic(step_edge.data.assignments[counter]) - counter_symbol
        if step.free_symbols & variant:  # Includes the counter
            return False
        if (sign > 0 and not step.is_positive) or (sign < 0 and not step.is_negative):
            return False
        after = _blocks_after(loop, step_edge)
        if after is None:
            return False

        start = _value_on_entry(loop, counter)
        if start is None:
            return False

        # The blocks after the step read the stepped counter
        stepped = f'({counter} + {symbolic.symstr(step)})'
        replacement = {counter: stepped}
        symbolic_replacement = {counter_symbol: counter_symbol + step}
        for edge in loop.edges():
            if edge.src in after:
                edge.data.replace_dict(replacement, replace_keys=False)
        for block in after:
            block.replace_dict(replacement, symbolic_replacement)
        del step_edge.data.assignments[counter]

        end = bound if strict else bound + sign
        operator = '<' if sign > 0 else '>'
        update = f'{counter} + {symbolic.symstr(step)}' if sign > 0 else f'{counter} - {symbolic.symstr(-step)}'
        loop.loop_variable = counter
        loop.init_statement = CodeBlock(f'{counter} = {symbolic.symstr(start)}')
        loop.loop_condition = CodeBlock(f'{counter} {operator} {symbolic.symstr(end)}')
        loop.update_statement = CodeBlock(f'{counter} = {update}')
        return True


def _comparison(condition: CodeBlock) -> Optional[Tuple[str, Any, int, bool]]:
    """
    Matches a loop condition that compares a symbol with a bound.

    :return: The symbol, the bound, the direction in which the symbol must move to end the loop (``1`` if it is
             compared with ``<`` or ``<=``), and whether the comparison is strict; or None.
    """
    try:
        expression = symbolic.pystr_to_symbolic(condition.as_string)
    except (TypeError, SyntaxError, sympy.SympifyError):
        return None
    # Truth tests of a comparison: ``(i < n) != 0`` and ``(i < n) == 1``
    while isinstance(expression, (sympy.Ne, sympy.Eq)):
        lhs, rhs = expression.args
        if not isinstance(lhs, sympy.core.relational.Relational):
            lhs, rhs = rhs, lhs
        if not isinstance(lhs, sympy.core.relational.Relational):
            break
        if isinstance(expression, sympy.Ne) and rhs in (sympy.false, sympy.Integer(0)):
            expression = lhs
        elif isinstance(expression, sympy.Eq) and rhs in (sympy.true, sympy.Integer(1)):
            expression = lhs
        else:
            return None
    direction = _DIRECTIONS.get(type(expression))
    if direction is None:
        return None
    sign, strict = direction
    lhs, rhs = expression.args
    if not isinstance(lhs, sympy.Symbol):
        if not isinstance(rhs, sympy.Symbol):
            return None
        lhs, rhs, sign = rhs, lhs, -sign  # ``n > i`` is ``i < n``
    if lhs in rhs.free_symbols:
        return None
    return str(lhs), rhs, sign, strict


def _assigned_symbols(region: AbstractControlFlowRegion) -> Set[str]:
    """Symbols that a region (including nested regions) assigns."""
    assigned = set()
    for edge in region.all_interstate_edges():
        assigned |= edge.data.assignments.keys()
    for nested in region.all_control_flow_regions():
        if isinstance(nested, LoopRegion) and nested.loop_variable:
            assigned.add(nested.loop_variable)
    return assigned


def _blocks_after(loop: LoopRegion, step_edge) -> Optional[Set[ControlFlowBlock]]:
    """
    The blocks of the loop body that execute after the step, or None if some path through the body does not take the
    step edge (the body is acyclic).
    """
    after = _successors(loop, step_edge.dst)
    before = set()
    stack = [loop.start_block]
    while stack:
        block = stack.pop()
        if block in before:
            continue
        before.add(block)
        stack.extend(edge.dst for edge in loop.out_edges(block) if edge is not step_edge)
    if before & after or step_edge.src not in before:
        return None
    if any(loop.out_degree(block) == 0 for block in before):
        return None  # An iteration that ends without the step
    return after


def _successors(region: ControlFlowRegion, block: ControlFlowBlock) -> Set[ControlFlowBlock]:
    result = set()
    stack = [block]
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(edge.dst for edge in region.out_edges(current))
    return result


def _value_on_entry(loop: LoopRegion, counter: str) -> Optional[Any]:
    """
    The value of ``counter`` when the loop is entered: the value of the last assignment on the chain of edges that
    leads to the loop, if the symbols it reads are not reassigned on the way.
    """
    graph = loop.parent_graph
    block = loop
    reassigned = set()
    while graph.in_degree(block) == 1:
        edge = graph.in_edges(block)[0]
        if counter in edge.data.assignments:
            value = symbolic.pystr_to_symbolic(edge.data.assignments[counter])
            names = {str(symbol) for symbol in value.free_symbols}
            # Assignments of an edge read the values from before the edge
            if names & (reassigned | edge.data.assignments.keys()):
                return None
            if names & set(loop.sdfg.arrays.keys()):
                return None
            return value
        reassigned |= edge.data.assignments.keys()
        block = edge.src
        if isinstance(block, AbstractControlFlowRegion):
            reassigned |= _assigned_symbols(block)
            if counter in reassigned:
                return None
    return None
