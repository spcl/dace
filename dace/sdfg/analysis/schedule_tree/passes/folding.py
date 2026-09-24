# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Range facts about iteration variables, and folding the conditions they decide."""
import ast
import copy
from typing import Dict, List, Optional, Tuple

from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (condition_of, iteration_spaces, names_in_subtrees,
                                                            names_written, range_analysis, repository_of)


def _boolean_leaves(node: ast.expr) -> int:
    """Number of non-boolean operands in a boolean expression."""
    if isinstance(node, ast.BoolOp):
        return sum(_boolean_leaves(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _boolean_leaves(node.operand)
    return 1


class RangeFacts:
    """What is known about iteration variables at a point of the tree: for each variable of an enclosing loop or
    map, its iteration space and an interval containing its value (the iteration range, narrowed by the enclosing
    conditions). Valid because an analyzable iteration space guarantees that neither the variable nor the symbols of
    its bounds change within the scope."""

    def __init__(self, repository, max_enumeration: int):
        self.repository = repository
        self.max_enumeration = max_enumeration
        self.known: Dict[str, tuple] = {}
        self.truths: Dict[str, bool] = {}  # Outcomes of condition atoms (by ``ast.dump``) of enclosing branches

    def _with(self, known: Dict[str, tuple], truths: Optional[Dict[str, bool]] = None) -> 'RangeFacts':
        result = copy.copy(self)
        result.known = known
        if truths is not None:
            result.truths = truths
        return result

    def within(self, child: tn.ScheduleTreeScope, previous: Optional[tn.ScheduleTreeNode]) -> 'RangeFacts':
        """The facts inside ``child`` (a scope whose preceding sibling is ``previous``)."""
        lrr = range_analysis()
        if isinstance(child, (tn.ForScope, tn.MapScope)):
            known = dict(self.known)
            for _, var, space in iteration_spaces(child, self.repository):
                if space is None:
                    known.pop(var, None)
                else:
                    known[var] = (space, space.iteration_range)
            return self._with(known)
        if isinstance(child, tn.ElseScope) and isinstance(previous, tn.IfScope):
            condition = condition_of(previous)
            clauses = None if condition is None else lrr.disjunctive_normal_form(condition, negate=True)
        else:
            condition = condition_of(child) if isinstance(child, tn.IfScope) else None
            clauses = None if condition is None else lrr.disjunctive_normal_form(condition)
        if not clauses or len(clauses) != 1:
            return self
        result = self.assuming(clauses[0])
        # The atoms hold throughout the branch unless it writes what they read
        written = names_in_subtrees([child], names_written)
        truths = dict(self.truths)
        for atom in clauses[0]:
            if not set(symbolic.symbols_in_ast(atom)) & written:
                truths[ast.dump(atom)] = True
        return result._with(result.known, truths)

    def assuming(self, atoms: List[ast.expr]) -> 'RangeFacts':
        """The facts where all of ``atoms`` hold (narrowing the known intervals where an atom is a single one)."""
        lrr = range_analysis()
        known = dict(self.known)
        for atom in atoms:
            for var in set(symbolic.symbols_in_ast(atom)) & known.keys():
                space, interval = known[var]
                intervals = lrr.atom_intervals(atom, space, self.max_enumeration)
                if intervals is not None and len(intervals) == 1:
                    known[var] = (space, lrr.intersect(interval, intervals[0]))
        return self._with(known)

    def verdict(self, atom: ast.expr) -> Optional[bool]:
        """Whether ``atom`` always (``True``) or never (``False``) holds here, or ``None`` if undecided."""
        lrr = range_analysis()
        if ast.dump(atom) in self.truths:
            return self.truths[ast.dump(atom)]
        negation = lrr.disjunctive_normal_form(atom, negate=True)
        if negation is not None and len(negation) == 1 and len(negation[0]) == 1:
            if ast.dump(negation[0][0]) in self.truths:
                return not self.truths[ast.dump(negation[0][0])]
        candidates = [self.known[var] for var in set(symbolic.symbols_in_ast(atom)) & self.known.keys()]
        if not candidates:  # Possibly an atom over compile-time constants alone
            candidates = [(lrr.IterationSpace(self.repository, '', 0, 0, 1, ast.Lt, set()), lrr.UNBOUNDED)]
        for space, interval in candidates:
            intervals = lrr.atom_intervals(atom, space, self.max_enumeration)
            if intervals is None:
                continue
            if all(lrr.provably_empty(lrr.intersect(interval, iv)) for iv in intervals):
                return False
            if any(lrr.provably_within(interval, iv) for iv in intervals):
                return True
        return None

    def fold(self, condition: ast.expr) -> Tuple[Optional[bool], Optional[ast.expr]]:
        """``(verdict, simplified condition)``: the verdict if the condition always or never holds, else the
        condition without its decided atoms (``None`` if nothing was decided)."""
        lrr = range_analysis()
        clauses = lrr.disjunctive_normal_form(condition)
        if clauses is None:
            return None, None
        kept_clauses, changed = [], False
        for clause in clauses:
            kept, dead = [], False
            for atom in clause:
                verdict = self.verdict(atom)
                if verdict is False:
                    dead = True
                    break
                if verdict is None:
                    kept.append(atom)
            changed |= dead or len(kept) < len(clause)
            if dead:
                continue
            if not kept:
                return True, None
            kept_clauses.append(kept)
        if not kept_clauses:
            return False, None
        if not changed:
            return None, None
        disjuncts = [lrr.conjunction(clause) for clause in kept_clauses]
        simplified = disjuncts[0] if len(disjuncts) == 1 else ast.BoolOp(op=ast.Or(), values=disjuncts)
        if _boolean_leaves(simplified) > _boolean_leaves(condition):
            return None, None  # The normal form would only grow the condition
        return None, simplified


def _fold_chains(children: List[tn.ScheduleTreeNode], facts: RangeFacts) -> Tuple[List[tn.ScheduleTreeNode], int]:
    """``children`` with every if/elif/else chain among them folded under ``facts`` (including the chains that
    folding exposes), and the number of conditions folded or simplified. Does not descend into other scopes."""
    result, folded, k = [], 0, 0
    while k < len(children):
        node = children[k]
        k += 1
        if condition_of(node) is None or not isinstance(node, tn.IfScope):
            result.append(node)
            continue
        chain = [node]
        while k < len(children) and isinstance(children[k], (tn.ElifScope, tn.ElseScope)):
            chain.append(children[k])
            k += 1
            if isinstance(chain[-1], tn.ElseScope):
                break
        kept: List[tn.ScheduleTreeScope] = []
        for branch in chain:
            if isinstance(branch, tn.ElseScope):
                if kept:
                    kept.append(branch)
                else:  # Every preceding branch was folded away
                    spliced, count = _fold_chains(branch.children, facts)
                    result += spliced
                    folded += count
                break
            condition = condition_of(branch)
            verdict, simplified = (None, None) if condition is None else facts.fold(condition)
            if verdict is not None or simplified is not None:
                folded += 1
            if verdict is False:
                continue
            if verdict is True:
                if kept:  # Taken whenever reached: the remaining branches are dead
                    kept.append(tn.ElseScope(children=branch.children))
                else:
                    spliced, count = _fold_chains(branch.children, facts)
                    result += spliced
                    folded += count
                break
            if simplified is not None:
                branch.condition = CodeBlock([ast.fix_missing_locations(ast.Expr(value=simplified))])
            if not kept and isinstance(branch, tn.ElifScope):
                branch = tn.IfScope(condition=branch.condition, children=branch.children)
            kept.append(branch)
        result += kept
    return result, folded


def fold_scope(scope: tn.ScheduleTreeScope, facts: RangeFacts) -> int:
    """Fold the conditions in the body of ``scope`` (whose own facts are ``facts``), recursively."""
    children, folded = _fold_chains(scope.children, facts)
    scope.children = []
    scope.add_children(children)
    previous = None
    for child in scope.children:
        if isinstance(child, tn.ScheduleTreeScope):
            folded += fold_scope(child, facts.within(child, previous))
        previous = child
    return folded


def facts_at(scope: tn.ScheduleTreeScope, max_enumeration: int) -> RangeFacts:
    """The facts inside ``scope``, from all of its ancestors."""
    path = [scope]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    facts = RangeFacts(repository_of(path[-1]), max_enumeration)
    for parent, child in zip(reversed(path), list(reversed(path))[1:]):
        index = next(k for k, c in enumerate(parent.children) if c is child)
        facts = facts.within(child, parent.children[index - 1] if index > 0 else None)
    return facts


def fold_guards(stree: tn.ScheduleTreeScope, max_enumeration: int = 1 << 20) -> int:
    """
    Fold conditions that the iteration ranges of the enclosing loops and maps decide.

    Within a loop or map, the iteration variable only takes values in the iteration range, narrowed further by the
    conditions of the enclosing ``if`` scopes. A condition atom that holds (or fails) for every such value is replaced
    by its outcome: in ``for i in range(1, N): if i >= 1 and A[i] > 0: X`` the guard becomes ``if A[i] > 0``. An
    ``if`` that always holds is replaced by its body and one that never holds is removed, promoting the ``elif`` and
    ``else`` scopes that follow accordingly. Atoms are analyzed as in
    :func:`~dace.transformation.passes.loop_range_reduction.atom_intervals`: symbolic comparisons with the iteration
    variable, and expressions over it and compile-time constants. Nothing is duplicated or reordered.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_enumeration: Upper bound on the iterates evaluated for an atom over compile-time constant data.
    :return: The number of conditions folded or simplified.
    """
    return fold_scope(stree, facts_at(stree, max_enumeration))
