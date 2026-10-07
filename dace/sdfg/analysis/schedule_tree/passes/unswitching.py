# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Moving conditions that do not change within a loop out of it (loop unswitching)."""
import ast
from typing import List, Optional

from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, bound_names, clone_body, condition_of, is_pure,
                                                            iteration_spaces, make_scope, names_in_subtrees,
                                                            names_written, range_analysis, repository_of)


def _chain_at(children: List[tn.ScheduleTreeNode], start: int) -> int:
    """The end (exclusive) of the if/elif/else chain starting at ``children[start]``."""
    end = start + 1
    while end < len(children) and isinstance(children[end], (tn.ElifScope, tn.ElseScope)):
        end += 1
        if isinstance(children[end - 1], tn.ElseScope):
            break
    return end


def _runs_at_least_once(scope: tn.ScheduleTreeScope, repository) -> bool:
    lrr = range_analysis()
    spaces = iteration_spaces(scope, repository)
    if not spaces or any(space is None for _, _, space in spaces):
        return False
    return all(
        lrr.provably_ge(space.end, space.start) if space.ascending else lrr.provably_ge(space.start, space.end)
        for _, _, space in spaces)


def _chain(conditions: List[Optional[CodeBlock]], bodies: List[list]) -> List[tn.ScheduleTreeNode]:
    """An if/elif/else chain from its conditions (``None`` for ``else``) and bodies, without empty trailing branches;
    ``if c: <nothing> else: B`` becomes ``if not c: B``."""
    from dace.frontend.python import astutils  # Avoid import loops
    while bodies and not bodies[-1] and len(bodies) > 1:
        conditions, bodies = conditions[:-1], bodies[:-1]
    if len(bodies) == 2 and not bodies[0] and conditions[1] is None:
        condition = conditions[0].code[0]
        negated = astutils.negate_expr(astutils.copy_tree(getattr(condition, 'value', condition))).value
        conditions, bodies = [CodeBlock([ast.fix_missing_locations(ast.Expr(value=negated))])], [bodies[1]]
    if len(bodies) == 1 and not bodies[0]:
        return []
    result = []
    for k, (condition, body) in enumerate(zip(conditions, bodies)):
        if condition is None:
            result.append(tn.ElseScope(children=body))
        elif k == 0:
            result.append(tn.IfScope(condition=condition, children=body))
        else:
            result.append(tn.ElifScope(condition=condition, children=body))
    return result


def unswitch_invariant_guards(stree: tn.ScheduleTreeScope, max_copies: int = 8) -> int:
    """
    Move conditions that do not change within a loop (or map) out of it, duplicating the loop per branch (loop
    unswitching).

    An if/elif/else chain directly in the body of a loop is unswitched if every condition in it is pure and invariant
    in the loop: it reads neither the iteration variables nor anything the loop writes, and the loop creates no aliases
    (views, references). ``for i: S1; if c: A else: B; S2`` becomes ``if c: for i: S1; A; S2 else: for i: S1; B; S2``;
    a chain without ``else`` gets one that runs the rest of the body. Inner loops are unswitched first, so a condition
    invariant in several enclosing loops ends up above all of them. The conditions are then evaluated once before the
    loop, also when the loop would not run at all, so conditions that read data (rather than only symbols) are only
    moved out of loops that provably run at least once.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_copies: Create at most this many copies of any one loop.
    :return: The number of chains moved out of loops.
    """
    root = stree.get_root()
    repository = repository_of(root)
    index = AccessIndex(root)
    unswitched = 0

    def invariant(loop: tn.ScheduleTreeScope, chain: list) -> bool:
        written = names_in_subtrees([loop], names_written) | bound_names(loop)
        if any(isinstance(n, (tn.ViewNode, tn.RefSetNode)) for n in loop.preorder_traversal()):
            return False
        reads_data = False
        for branch in chain:
            if isinstance(branch, tn.ElseScope):
                continue
            condition = condition_of(branch)
            if condition is None or not is_pure(condition):
                return False
            names = set(symbolic.symbols_in_ast(condition))
            if names & written:
                return False
            reads_data |= bool(names & root.containers.keys())
        return not reads_data or _runs_at_least_once(loop, repository)

    def process(loop: tn.ScheduleTreeScope, budget: int) -> List[tn.ScheduleTreeNode]:
        """``loop`` with the invariant chains of its body moved out, as the nodes that replace it."""
        nonlocal unswitched
        children = loop.children
        for start, child in enumerate(children):
            if not isinstance(child, tn.IfScope) or isinstance(child, tn.StateIfScope):
                continue
            end = _chain_at(children, start)
            chain = children[start:end]
            branches = len(chain) + (0 if isinstance(chain[-1], tn.ElseScope) else 1)
            if branches > budget or not invariant(loop, chain):
                continue
            before, after = children[:start], children[end:]
            bodies = [branch.children for branch in chain] + ([[]] if branches > len(chain) else [])
            # A copy that would run nothing is left out, unless the loop's final variable value is used afterwards
            droppable = isinstance(loop, tn.MapScope) or not index.used_outside(loop, loop.loop.loop_variable)
            copies, made = [], 0
            for k, body in enumerate(bodies):
                content = before + list(body) + after
                if not content and droppable:
                    copies.append(None)
                    continue
                if made > 0:
                    content = clone_body(loop, content, index.used_outside)
                copy_ = make_scope(loop, 0 if isinstance(loop, tn.MapScope) else None, None, None, [], k)
                copy_.add_children(content)
                copy_.parent = loop.parent
                if made > 0:
                    index.add(copy_)
                made += 1
                copies.append(copy_)
            index.changed()
            unswitched += 1
            branch_nodes = [[] if copy_ is None else process(copy_, budget // branches) for copy_ in copies]
            conditions = [None if isinstance(b, tn.ElseScope) else b.condition for b in chain]
            conditions += [None] * (branches - len(chain))
            return _chain(conditions, branch_nodes)
        return [loop]

    def visit(scope: tn.ScheduleTreeScope):
        for child in scope.children:
            if isinstance(child, tn.ScheduleTreeScope):
                visit(child)  # Innermost first
        result, changed = [], False
        for child in scope.children:
            if isinstance(child, (tn.ForScope, tn.MapScope)):
                replacement = process(child, max_copies)
                changed |= not (len(replacement) == 1 and replacement[0] is child)
                result += replacement
            else:
                result.append(child)
        if changed:
            scope.children = []
            scope.add_children(result)

    visit(stree)
    return unswitched
