# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Merging consecutive loops with identical iteration spaces."""
import copy

import sympy

from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import iteration_spaces, repository_of, trip_count


def _same_iteration_space(first: tn.ForScope, second: tn.ForScope) -> bool:
    if first.loop.loop_variable != second.loop.loop_variable:
        return False
    return all(
        getattr(first.loop, statement).as_string == getattr(second.loop, statement).as_string
        for statement in ("init_statement", "loop_condition", "update_statement")
    )


def _overmerge_bounds(first: tn.ForScope, second: tn.ForScope, repository):
    if first.loop.loop_variable != second.loop.loop_variable:
        return None
    first_spaces, second_spaces = iteration_spaces(first, repository), iteration_spaces(second, repository)
    if len(first_spaces) != 1 or len(second_spaces) != 1:
        return None
    first_space, second_space = first_spaces[0][2], second_spaces[0][2]
    if first_space is None or second_space is None:
        return None
    first_count, second_count = trip_count(first_space), trip_count(second_space)
    if first_count is None or second_count is None or first_count == 0 or second_count == 0:
        return None
    if first_space.ascending != second_space.ascending:
        return None

    if sympy.simplify(first_space.stride - second_space.stride) != 0:
        return None
    stride = sympy.simplify(first_space.stride)
    if not stride.is_number or stride == 0:
        return None
    offset = sympy.simplify((first_space.start - second_space.start) / stride)
    if offset.is_integer is not True:
        return None
    return first_space, second_space


def _overmerge(first: tn.ForScope, second: tn.ForScope, first_space, second_space) -> None:
    variable = first.loop.loop_variable
    assert variable is not None
    ascending = first_space.ascending

    if ascending:
        start_expr = f"min({symbolic.symstr(first_space.start)}, {symbolic.symstr(second_space.start)})"
        end_expr = f"max({symbolic.symstr(first_space.end)}, {symbolic.symstr(second_space.end)})"
        first_guard = f"{variable} >= {symbolic.symstr(first_space.start)} and {variable} <= {symbolic.symstr(first_space.end)}"
        second_guard = f"{variable} >= {symbolic.symstr(second_space.start)} and {variable} <= {symbolic.symstr(second_space.end)}"
        condition = f"{variable} <= {end_expr}"
    else:
        start_expr = f"max({symbolic.symstr(first_space.start)}, {symbolic.symstr(second_space.start)})"
        end_expr = f"min({symbolic.symstr(first_space.end)}, {symbolic.symstr(second_space.end)})"
        first_guard = f"{variable} <= {symbolic.symstr(first_space.start)} and {variable} >= {symbolic.symstr(first_space.end)}"
        second_guard = f"{variable} <= {symbolic.symstr(second_space.start)} and {variable} >= {symbolic.symstr(second_space.end)}"
        condition = f"{variable} >= {end_expr}"

    first_if = tn.IfScope(condition=CodeBlock(first_guard), children=first.children, parent=first)
    second_if = tn.IfScope(condition=CodeBlock(second_guard), children=second.children, parent=second)
    for child in first.children:
        child.parent = first_if
    for child in second.children:
        child.parent = second_if
    first.children = [first_if]
    second.children = [second_if]
    first.loop = copy.deepcopy(first.loop)
    first.loop.init_statement = CodeBlock(f"{variable} = {start_expr}")
    first.loop.loop_condition = CodeBlock(condition)


def merge_consecutive_loops(
    stree: tn.ScheduleTreeScope,
    *,
    over_merge: bool = False,
    iterator: str | None = None
) -> int:
    """Merge adjacent ``ForScope`` nodes with identical iteration spaces.

    The loop bodies are concatenated in their original order. By default, loops need identical headers. With
    ``over_merge=True``, loops of different sizes with the same iterator, direction, and stride may be widened to their
    union, with guards preserving each body's original range. ``iterator`` restricts merging to that loop variable.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param over_merge: Merge loops with different bounds by widening and guarding their bodies.
    :param iterator: Merge only loops using this iterator, or all iterators if ``None``.
    :return: The number of loops merged.
    """
    merged = 0
    repository = repository_of(stree.get_root())

    def visit(scope: tn.ScheduleTreeScope):
        nonlocal merged
        for child in scope.children:
            if isinstance(child, tn.ScheduleTreeScope):
                visit(child)

        children = []
        for child in scope.children:
            if children and isinstance(children[-1], tn.ForScope) and isinstance(child, tn.ForScope):
                target = children[-1]
                matches_iterator = iterator is None or target.loop.loop_variable == iterator
                if matches_iterator and _same_iteration_space(target, child):
                    target.children.extend(child.children)
                    for grandchild in child.children:
                        grandchild.parent = target
                    merged += 1
                    continue
                bounds = _overmerge_bounds(target, child, repository) if over_merge and matches_iterator else None
                if bounds is not None:
                    _overmerge(target, child, *bounds)
                    target.children.extend(child.children)
                    for grandchild in child.children:
                        grandchild.parent = target
                    merged += 1
                    continue
                children.append(child)
            else:
                children.append(child)

        if len(children) != len(scope.children):
            scope.children = []
            scope.add_children(children)

    visit(stree)
    return merged