# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Moving statements and subtrees of a schedule tree past their siblings.

:class:`MotionAnalysis` decides whether a subtree may run after (or before) some of its siblings instead of where it
is, and which iterations of its outer loop must then stay in place. :func:`move_statements_to_readers` uses it to move
statements to the readers of their values.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

import sympy

from dace import data, dtypes, symbolic
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.dataflow import Box, ReachingAnalysis, disjoint
from dace.sdfg.analysis.schedule_tree.passes.common import (
    assigned_in,
    bound_names,
    child_containing,
    clone_subtree,
    condition_element_accesses,
    condition_of,
    expression_symbols,
    inside,
    iteration_spaces,
    make_scope,
    memlet_element,
    memlets_of,
    names_in_subtrees,
    names_read,
    names_written,
    path_to,
    range_analysis,
    repository_of,
)
from dace.sdfg.analysis.schedule_tree.passes.readers import (
    Read,
    collect_memlet_uses,
    collect_reads,
    guard_candidates,
    guarded_anywhere,
    some_read_guarded,
    value_reads,
)
from dace.sdfg.analysis.schedule_tree.passes.transient_reuse import Liveness, Use

# ----------------------------------------------------------------------------------------------------------------------
# Reordering siblings
# ----------------------------------------------------------------------------------------------------------------------

# Nodes that create aliases or jump to arbitrary places: subtrees that contain them are not reordered
_OPAQUE_NODES = (tn.ViewNode, tn.NViewEnd, tn.RefSetNode, tn.GotoNode, tn.StateLabel)

#: A subtree, or statements of one loop body (in body order) that move together
Statements = Union[tn.ScheduleTreeNode, Sequence[tn.ScheduleTreeNode]]


def _statements(unit: Statements) -> tuple[tn.ScheduleTreeNode, ...]:
    return (unit,) if isinstance(unit, tn.ScheduleTreeNode) else tuple(unit)


@dataclass
class Access:
    """An access of a container by a node: through a memlet, or (if ``memlet`` is ``None``) at elements that are not
    known, as for an array read in a condition at indices that are not analyzed."""

    name: str
    write: bool
    node: tn.ScheduleTreeNode
    memlet: Memlet | None


@dataclass
class Footprint:
    """What a subtree may do when it runs (see :meth:`MotionAnalysis.footprint`)."""

    accesses: list[Access] = field(default_factory=list)
    reads: set[str] = field(default_factory=set)  #: Symbols read that are not bound within the subtree
    assigns: set[str] = field(default_factory=set)  #: Symbols assigned, other than loop and map variables
    binds: set[str] = field(default_factory=set)  #: Loop and map variables bound
    opaque: bool = False  #: Whether it may do more than its accesses say (side effects, aliases, jumps out)


@dataclass
class Reordering:
    """How a subtree may move past some of its siblings: entirely, or with the iterations of its outer loop whose
    variable values lie in ``stay`` (an inclusive interval, lowest value first) left in place in a loop of their own."""

    stay: tuple[int, int] | None = None


class MotionAnalysis:
    """
    Accesses of subtrees and the conflicts between them, for running siblings in another order. Results are cached
    and describe the tree when the analysis was created: create a new analysis after changing the tree.

    Reordering two siblings ``A`` and ``B`` of one scope is valid if, in every execution of the scope, they conflict
    on nothing: no container region that one writes and the other accesses (regions over the loops inside each, with
    the variables of the scopes around both left symbolic, since both run with the same values of them), no symbol
    that one assigns and the other reads or assigns, and neither has side effects, creates aliases or jumps out of
    itself. Loop variables count as assigned by their loops, except that two loops binding the same variable do not
    conflict on it unless its value after the loop is read somewhere.

    :param root: The schedule tree.
    """

    def __init__(self, root: tn.ScheduleTreeRoot):
        self.root = root
        self.containers = root.containers
        self.repository = repository_of(root)
        self.reaching = ReachingAnalysis(root)
        self._footprints: dict[tuple[int, int], Footprint] = {}
        self._escaping: set[str] | None = None

    # ------------------------------------------------------------------------------------------------------------------
    # What subtrees do

    def footprint(self, unit: Statements, scope: tn.ScheduleTreeScope) -> Footprint:
        """
        What ``unit`` does when it runs in one execution of ``scope``, an ancestor of it. The loops between the two
        count as part of the unit, as when statements are split off from the perfect loop nest around them into a nest
        of their own; scopes there other than ``for`` loops and maps make the unit opaque.

        :param unit: The subtree, or statements of one body (siblings, in order).
        :param scope: An ancestor of ``unit`` (or its parent).
        :return: The accesses, symbols and opacity of the unit.
        """
        statements = _statements(unit)
        key = (tuple(id(s) for s in statements), id(scope))
        if key in self._footprints:
            return self._footprints[key]
        result = Footprint()
        bound: set[str] = set()
        for loop in reversed(path_to(statements[0], scope)[1:]):  # Outermost first
            if not isinstance(loop, (tn.ForScope, tn.MapScope)):
                result.opaque = True
            own = bound_names(loop)
            for name in names_read(loop) - bound - own:
                if name in self.containers:
                    result.accesses.append(Access(name, False, loop, None))
                else:
                    result.reads.add(name)
            result.binds |= own
            bound |= own
        for statement in statements:
            self._walk(statement, statement.parent, bound, result)
        for access in result.accesses:
            desc = self.containers[access.name]
            if (
                isinstance(desc, (data.View, data.Reference))
                or not isinstance(desc, (data.Scalar, data.Array))
                or desc.may_alias
            ):
                result.opaque = True  # Other names may refer to the same memory
        self._footprints[key] = result
        return result

    def _walk(self, node: tn.ScheduleTreeNode, around: tn.ScheduleTreeScope, bound: set[str], result: Footprint):
        if isinstance(node, _OPAQUE_NODES):
            result.opaque = True
        elif isinstance(node, (tn.TaskletNode, tn.LibraryCall)) and node.node.has_side_effects(self.repository):
            result.opaque = True
        elif isinstance(node, (tn.BreakNode, tn.ContinueNode)):
            if not any(isinstance(n, tn.LoopScope) for n in path_to(node, around)[1:]):
                result.opaque = True  # Leaves a loop around the unit
        own = bound_names(node)
        read, written = set(), set()
        for memlet in memlets_of(node, "in_memlets") + memlets_of(node, "memlet"):
            result.accesses.append(Access(memlet.data, False, node, memlet))
            read.add(memlet.data)
        for memlet in memlets_of(node, "out_memlets"):
            result.accesses.append(Access(memlet.data, True, node, memlet))
            written.add(memlet.data)
        condition = condition_of(node) if isinstance(node, (tn.IfScope, tn.ElifScope)) else None
        elements = None if condition is None else condition_element_accesses(condition, self.containers)
        for name, element, _, _ in elements or []:
            result.accesses.append(Access(name, False, node, Memlet(element)))
            read.add(name)
        for name in names_read(node) - read - bound - own:
            if name in self.containers:
                result.accesses.append(Access(name, False, node, None))
            else:
                result.reads.add(name)
        for name in names_written(node) - written - own:
            if name in self.containers:
                result.accesses.append(Access(name, True, node, None))
            else:
                result.assigns.add(name)
        result.binds |= own
        for child in getattr(node, "children", ()):
            self._walk(child, around, bound | own, result)

    @property
    def escaping(self) -> set[str]:
        """Loop and map variables that are read where no loop or map binds them, i.e., after their loops."""
        if self._escaping is None:
            tree = self.footprint(self.root, self.root)
            self._escaping = tree.reads & tree.binds
        return self._escaping

    # ------------------------------------------------------------------------------------------------------------------
    # Conflicts

    def _region(self, access: Access, context: tn.ScheduleTreeNode) -> Box:
        if access.memlet is None:
            return [None] * len(self.containers[access.name].shape)
        return self.reaching.region(access.node, access.memlet, context)

    def _outer_space(self, unit: Statements, scope: tn.ScheduleTreeScope) -> tuple[tn.ForScope, int, int, int] | None:
        """``(loop, first, last, stride)`` of the outer loop of a unit (the child of ``scope`` that contains it), if it
        is a ``for`` loop with constant bounds and unit stride."""
        outer = child_containing(scope, _statements(unit)[0])
        if not isinstance(outer, tn.ForScope) or not outer.loop.loop_variable:
            return None
        space = self.reaching.ranges(outer).get(outer.loop.loop_variable)
        if space is None or not all(sympy.sympify(v).is_Integer for v in space) or abs(space[2]) != 1:
            return None
        return outer, int(space[0]), int(space[1]), int(space[2])

    def _iterations(
        self, access: Access, outer: tn.ForScope, low: int, high: int, theirs: Box
    ) -> tuple[int, int] | None:
        """The values of the outer loop's variable (between ``low`` and ``high``) for which an access of the unit may
        touch the region ``theirs``; ``None`` if they may be all, an empty interval (first > last) if there are
        none."""
        if access.memlet is None or access.node is outer:
            return None
        variable = outer.loop.loop_variable
        result = (low, high)
        restricted = False
        for mine, other in zip(self._region(access, outer), theirs):
            if mine is None or other is None:
                continue
            offsets = []
            for bound in mine:
                bound = sympy.expand(bound)
                symbol = next((s for s in bound.free_symbols if str(s) == variable), None)
                slope = sympy.diff(bound, symbol) if symbol is not None else None
                offsets.append(None if slope not in (1, -1) else (int(slope), sympy.expand(bound - slope * symbol)))
            if None in offsets or offsets[0][0] != offsets[1][0]:
                continue  # Does not depend on the variable in this dimension, or not in a way analyzed here
            (slope, start), (_, end) = offsets
            if not all(v.is_Integer for v in (start, end, sympy.sympify(other[0]), sympy.sympify(other[1]))):
                continue
            start, end, first, last = int(start), int(end), int(other[0]), int(other[1])
            # ``slope * v + start <= last`` and ``first <= slope * v + end`` (the bounds swap for a negative slope)
            if slope == 1:
                interval = (first - end, last - start)
            else:
                interval = (start - last, end - first)
            result = (max(result[0], interval[0]), min(result[1], interval[1]))
            restricted = True
        return result if restricted else None

    def reorder_conflicts(
        self, unit: Statements, scope: tn.ScheduleTreeScope, between: list[tn.ScheduleTreeNode]
    ) -> list[tuple[int, int]] | None:
        """
        The iterations of the outer loop of ``unit`` that conflict with the siblings it would move past.

        :param unit: What moves: a child of ``scope``, or statements of the body of a perfect nest of loops that is
                     a child of ``scope`` (moved with copies of the loops around them, see :meth:`footprint`). Its
                     outer loop is the child of ``scope`` that contains it.
        :param scope: The scope in which it moves.
        :param between: The children of ``scope`` it would move past (all before or all after it).
        :return: Inclusive intervals of values of the outer loop's variable whose iterations conflict (empty if none
                 does), or ``None`` if the unit conflicts as a whole: some conflict cannot be confined to iterations of
                 an outer ``for`` loop with constant bounds and unit stride.
        """
        mine = self.footprint(unit, scope)
        if mine.opaque:
            return None
        outer = self._outer_space(unit, scope)
        conflicts: list[tuple[int, int]] = []
        for sibling in between:
            theirs = self.footprint(sibling, scope)
            if theirs.opaque:
                return None
            if (
                mine.reads & (theirs.assigns | theirs.binds)
                or theirs.reads & (mine.assigns | mine.binds)
                or mine.assigns & theirs.assigns
                or mine.binds & theirs.binds & self.escaping
            ):
                return None
            for access in mine.accesses:
                region = None
                for other in theirs.accesses:
                    if other.name != access.name or not (access.write or other.write):
                        continue
                    if region is None:
                        region = self._region(access, scope)
                    other_region = self._region(other, scope)
                    if disjoint(region, other_region):
                        continue
                    if outer is None:
                        return None
                    loop, first, last, _ = outer
                    interval = self._iterations(access, loop, min(first, last), max(first, last), other_region)
                    if interval is None:
                        return None
                    if interval[0] <= interval[1]:
                        conflicts.append(interval)
        return conflicts

    def iterations_independent(self, unit: Statements, scope: tn.ScheduleTreeScope) -> bool:
        """
        Whether the iterations of the outer loop of ``unit`` (see :meth:`reorder_conflicts`) may run in any order:
        the unit assigns no symbols, and every access of each container it writes touches, in each iteration, one
        slice of the container that the same index expression in the loop variable selects (with coefficient one or
        minus one), so iterations touch disjoint parts of what any of them writes.

        :param unit: What moves (see :meth:`reorder_conflicts`).
        :param scope: The scope in which it moves.
        :return: True if the iterations are independent, False if that is not shown.
        """
        footprint = self.footprint(unit, scope)
        outer = self._outer_space(unit, scope)
        if footprint.opaque or footprint.assigns or outer is None:
            return False
        loop = outer[0]
        variable = loop.loop.loop_variable
        written = {a.name for a in footprint.accesses if a.write}
        for name in written:
            selectors = None  # The dimensions that select one slice per iteration in every access so far
            for access in footprint.accesses:
                if access.name != name:
                    continue
                if access.memlet is None or access.node is loop:
                    return False
                here = set()
                for dim, extent in enumerate(self._region(access, loop)):
                    if extent is None or sympy.expand(extent[0] - extent[1]) != 0:
                        continue
                    index = sympy.expand(extent[0])
                    symbol = next((s for s in index.free_symbols if str(s) == variable), None)
                    if symbol is not None and sympy.diff(index, symbol) in (1, -1):
                        here.add((dim, index))
                selectors = here if selectors is None else selectors & here
                if not selectors:
                    return False
        return True

    def reordering(
        self, unit: Statements, scope: tn.ScheduleTreeScope, between: list[tn.ScheduleTreeNode], downward: bool
    ) -> Reordering | None:
        """
        How ``unit`` may move past ``between`` (see :meth:`reorder_conflicts`): entirely if nothing conflicts, else
        leaving in place the iterations of its outer loop that conflict, if they are the first or last ones. The
        others run in a copy of the loop at the new place. If the iterations left in place are those that run
        later (the last ones for a move down, the first ones for a move up), the order of iterations changes, which
        is only valid if they are independent (see :meth:`iterations_independent`).

        :param unit: What moves (see :meth:`reorder_conflicts`).
        :param scope: The scope in which it moves.
        :param between: The children of ``scope`` it moves past.
        :param downward: Whether it moves down (past later siblings) or up (past earlier ones).
        :return: The reordering, or ``None`` if the unit cannot move.
        """
        conflicts = self.reorder_conflicts(unit, scope, between)
        if conflicts is None:
            return None
        if not conflicts:
            return Reordering()
        _, first, last, stride = self._outer_space(unit, scope)
        low, high = min(first, last), max(first, last)
        lo, hi = min(c[0] for c in conflicts), max(c[1] for c in conflicts)
        if lo == low and hi < high:
            stay, at_start = (low, hi), stride > 0
        elif hi == high and lo > low:
            stay, at_start = (lo, high), stride < 0
        else:
            return None
        if at_start != downward and not self.iterations_independent(unit, scope):
            return None
        spaces = iteration_spaces(child_containing(scope, _statements(unit)[0]), self.repository)
        if len(spaces) != 1 or spaces[0][2] is None:
            return None  # The loop cannot be split
        return Reordering(stay)


# ----------------------------------------------------------------------------------------------------------------------
# Units of motion
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class Unit:
    """
    What moves among the children of ``scope``: one child (``statements == [child]``, no ``nest``), or statements of
    the innermost body of a perfect nest of ``for`` loops that is a child (``nest``, outermost first), which move with
    copies of the loops around them (loop fission). Statements keep their order.
    """

    scope: tn.ScheduleTreeScope
    statements: list[tn.ScheduleTreeNode]
    nest: list[tn.ForScope] = field(default_factory=list)

    @property
    def outer(self) -> tn.ScheduleTreeNode:
        """The child of ``scope`` that contains the unit."""
        return self.nest[0] if self.nest else self.statements[0]

    @property
    def whole(self) -> bool:
        """Whether the unit is all of ``outer``, which then moves as it is."""
        return not self.nest or len(self.statements) == len(self.nest[-1].children)


def subtree_unit(child: tn.ScheduleTreeNode) -> Unit:
    """A child of a scope, as a unit."""
    return Unit(child.parent, [child])


def _nest_around(statement: tn.ScheduleTreeNode) -> list[tn.ForScope] | None:
    """The loops of the perfect nest whose innermost body contains ``statement``, outermost first."""
    loops = []
    scope = statement.parent
    while type(scope) is tn.ForScope:
        loops.append(scope)
        if type(scope.parent) is not tn.ForScope or len(scope.parent.children) != 1:
            break
        scope = scope.parent
    return loops[::-1] or None


def _injective(element: tuple[str, ...], variables: list[str]) -> bool:
    """Whether every variable is the index of some dimension, up to a constant (so iterations access distinct
    elements)."""
    for variable in variables:
        if not any(
            symbolic.simplify(symbolic.pystr_to_symbolic(index) - symbolic.pystr_to_symbolic(variable)).is_number
            for index in element
        ):
            return False
    return True


def split_off(statements: list[tn.ScheduleTreeNode], motion: MotionAnalysis) -> Unit | None:
    """
    Statements of the innermost body of a perfect nest of ``for`` loops, as a unit that runs in copies of the loops
    (loop fission), if splitting them off is valid: the statements are tasklets that access single elements, without
    side effects, and the rest of the body

    * neither reads nor writes what they write,
    * assigns no symbol they read and does not jump (``break``, ``continue``),
    * writes what they read only before all of them that read it, at the one element they read, with every loop
      variable of the nest indexing that element (each iteration then reads what it read before).

    With all of the body, the unit is the whole nest.

    :param statements: Children of the innermost loop of the nest.
    :param motion: The analysis of the tree.
    :return: The unit, or ``None`` if the statements cannot be split off.
    """
    nest = _nest_around(statements[0])
    if nest is None or any(not loop.loop.loop_variable for loop in nest) or nest[0].parent is None:
        return None
    body = nest[-1].children
    if any(s.parent is not nest[-1] for s in statements):
        return None
    positions = sorted(next(k for k, c in enumerate(body) if c is s) for s in statements)
    statements = [body[k] for k in positions]
    unit = Unit(nest[0].parent, statements, nest)
    if unit.whole:
        return unit
    variables = [loop.loop.loop_variable for loop in nest]
    outputs: set[str] = set()
    inputs: dict[str, tuple[str, ...]] = {}  # Containers read and not written, by the element read
    first_read: dict[str, int] = {}  # By the position of the first statement reading them
    for position, statement in zip(positions, statements):
        if not isinstance(statement, tn.TaskletNode) or statement.node.has_side_effects(motion.repository):
            return None
        for memlet in statement.out_memlets.values():
            outputs.add(memlet.data)
        for memlet in statement.in_memlets.values():
            element = memlet_element(memlet)
            if element is None or inputs.get(memlet.data, element) != element:
                return None
            inputs[memlet.data] = element
            first_read.setdefault(memlet.data, position)
    for name in outputs:
        inputs.pop(name, None)
    mine = motion.footprint(statements, nest[-1])
    rest = [(k, c) for k, c in enumerate(body) if not any(c is s for s in statements)]
    for k, child in rest:
        theirs = motion.footprint(child, nest[-1])
        if theirs.opaque or mine.reads & (theirs.assigns | theirs.binds):
            return None
        if (names_in_subtrees([child], names_read) | names_in_subtrees([child], names_written)) & outputs:
            return None
        for node in child.preorder_traversal():
            for name in names_written(node) & inputs.keys():
                if k > first_read[name] or not isinstance(node, tn.TaskletNode):
                    return None
                for memlet in node.out_memlets.values():
                    if memlet.data == name and (
                        memlet_element(memlet) != inputs[name] or not _injective(inputs[name], variables)
                    ):
                        return None
    return unit


def statement_group(seed: tn.TaskletNode, motion: MotionAnalysis) -> Unit | None:
    """
    A statement of the innermost body of a perfect nest of ``for`` loops together with the earlier statements of the
    body whose values only the group uses there (e.g., a value and its positive part), split off from the nest
    (see :func:`split_off`); the statement alone if the group cannot be split off.

    :param seed: The statement.
    :param motion: The analysis of the tree.
    :return: The unit, or ``None`` if not even the statement can be split off.
    """
    if _nest_around(seed) is None:
        return None
    body = seed.parent.children
    index = next(k for k, c in enumerate(body) if c is seed)
    group = [seed]
    for k in range(index - 1, -1, -1):
        candidate = body[k]
        if not isinstance(candidate, tn.TaskletNode):
            continue
        written = {m.data for m in candidate.out_memlets.values()}
        if not written & names_in_subtrees(group, names_read):
            continue
        others = [c for c in body if c is not candidate and not any(c is g for g in group)]
        if not (names_in_subtrees(others, names_read) | names_in_subtrees(others, names_written)) & written:
            group.insert(0, candidate)
    unit = split_off(group, motion) if len(group) > 1 else None
    return unit or split_off([seed], motion)


def move_unit(unit: Unit, before: tn.ScheduleTreeNode | None, stay: tuple[int, int] | None = None) -> None:
    """
    Move a unit to just before the child ``before`` of its scope (to the end if ``None``), as decided by
    :meth:`MotionAnalysis.reordering`: the iterations of its outer loop whose variable values lie in ``stay`` remain
    where they are, in a copy of the loop. A split-off unit runs in copies of the loops of its nest; the rest of the
    nest stays.

    :param unit: The unit.
    :param before: The child of the unit's scope to move it before.
    :param stay: Inclusive interval of values of the outer loop's variable whose iterations stay in place.
    """
    scope = unit.scope
    children = list(scope.children)
    if unit.whole and stay is None:
        children = [c for c in children if c is not unit.outer]
        children.insert(next((k for k, c in enumerate(children) if c is before), len(children)), unit.outer)
        scope.children = []
        scope.add_children(children)
        return
    nest = unit.nest or [unit.outer]
    statements = unit.statements if unit.nest else list(unit.outer.children)
    space = None
    if stay is not None:
        spaces = iteration_spaces(nest[0], repository_of(scope.get_root()))
        if len(spaces) != 1 or spaces[0][2] is None:
            raise ValueError("The outer loop of the unit cannot be split")
        space = spaces[0][2]
    innermost = nest[-1]
    innermost.children = [c for c in innermost.children if not any(c is s for s in statements)]
    lrr = range_analysis()

    def copy_nest(interval, k: int) -> tn.ForScope:
        body: list[tn.ScheduleTreeNode] = [clone_subtree(s) for s in statements]
        for loop in reversed(nest[1:]):
            body = [make_scope(loop, None, None, None, body, k)]
        return make_scope(nest[0], None, space, interval, body, k)

    moved_interval = None
    if stay is not None:
        first, last = (space.start, space.end) if space.ascending else (space.end, space.start)
        lo, hi = stay
        moved_interval = lrr.Interval(hi + 1, last) if lo == first else lrr.Interval(first, lo - 1)
        children.insert(next(k for k, c in enumerate(children) if c is nest[0]) + 1, copy_nest(lrr.Interval(*stay), 1))
    children.insert(
        next((k for k, c in enumerate(children) if c is before), len(children)), copy_nest(moved_interval, 2)
    )
    if not innermost.children:
        children = [c for c in children if c is not nest[0]]
    scope.children = []
    scope.add_children(children)


def move_plan(unit: Unit, before: tn.ScheduleTreeNode | None, motion: MotionAnalysis) -> Reordering | None:
    """
    Whether and how ``unit`` may move to just before the child ``before`` of its scope (to the end if ``None``), up or
    down (see :meth:`MotionAnalysis.reordering`).

    :param unit: The unit.
    :param before: The child of the unit's scope to move it before.
    :param motion: The analysis of the tree.
    :return: How to move it (pass ``stay`` to :func:`move_unit`), or ``None`` if it cannot move there.
    """
    children = unit.scope.children
    position = next(k for k, c in enumerate(children) if c is unit.outer)
    target = next((k for k, c in enumerate(children) if c is before), len(children))
    downward = target > position
    between = children[position + 1 : target] if downward else children[target:position]
    if not between:
        return Reordering()
    return motion.reordering(unit.statements, unit.scope, between, downward)


# ----------------------------------------------------------------------------------------------------------------------
# Moving statements to their readers
# ----------------------------------------------------------------------------------------------------------------------


def move_statements_to_readers(stree: tn.ScheduleTreeScope) -> int:
    """
    Move statements past what prevents putting them under the guard of their readers: an effect-free tasklet in the
    body of a perfect nest of ``for`` loops, whose values are read only under ``if`` scopes in a later nest of the same
    scope, is split off into a nest of its own (loop fission), together with the earlier statements whose values only
    it uses (see :func:`statement_group`), and placed right after the last nest in between that assigns what those
    guards read. :func:`sink_into_guards` can then put the statements under the guard of their readers.

    The split must be valid (see :func:`split_off`) and the split-off nest must not conflict with the nests in between
    (see :class:`MotionAnalysis`); where it does so only in the first or last iterations of the outermost loop, those
    iterations stay in place in a nest of their own.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of statement groups moved.
    """
    root = stree.get_root()
    containers = root.containers
    moved = 0
    while True:
        reads, other = collect_reads(root)
        liveness = Liveness(root, trust_reads=False)
        reaching = ReachingAnalysis(root)
        memlet_uses = collect_memlet_uses(root)
        motion = MotionAnalysis(root)
        cache: dict[str, tuple] = {}
        for node in stree.preorder_traversal():
            if isinstance(node, tn.TaskletNode):
                plan = _move_plan(node, reads, other, containers, liveness, reaching, memlet_uses, motion, cache)
                if plan is not None:
                    move_unit(*plan)
                    moved += 1
                    break  # The analyses describe the tree before the change
        else:
            return moved


def _move_plan(
    statement: tn.TaskletNode,
    reads: dict[str, list[Read]],
    other: set[str],
    containers: dict[str, data.Data],
    liveness: Liveness,
    reaching: ReachingAnalysis,
    memlet_uses: dict[str, list[Use]],
    motion: MotionAnalysis,
    cache: dict[str, tuple],
) -> tuple | None:
    """``(unit, target, stay)`` to move ``statement`` and the statements split off with it to before the sibling
    ``target`` of their nest, keeping the iterations of the outermost loop whose values lie in ``stay`` in place (or
    ``None``), or ``None`` if it cannot or need not be moved."""
    tasklet = statement.node
    if tasklet.language != dtypes.Language.Python or tasklet.side_effects:
        return None
    if not all(some_read_guarded(statement, reads.get(m.data, [])) for m in statement.out_memlets.values()):
        return None  # Quick filter before the other analyses
    unit = statement_group(statement, motion)
    if unit is None or not unit.nest:
        return None
    for member in unit.statements:
        if member.node.language != dtypes.Language.Python or member.node.side_effects:
            return None
        elements: dict[str, tuple[str, ...]] = {}
        for memlet in member.out_memlets.values():
            element = memlet_element(memlet)
            desc = containers.get(memlet.data)
            if (
                element is None
                or desc is None
                or not desc.transient
                or memlet.data in other
                or memlet.wcr is not None
                or elements.get(memlet.data, element) != element
            ):
                return None
            elements[memlet.data] = element

    # The readers: under guards, all in one later sibling nest, with nests in between (else there is no reason to move)
    scope, nest = unit.scope, unit.nest
    siblings = scope.children
    position = next(k for k, c in enumerate(siblings) if c is nest[0])
    target = None
    guard_names = set()
    for member in unit.statements:
        for name in {m.data for m in member.out_memlets.values()}:
            found, _ = value_reads(member, name, reads.get(name, []), liveness, reaching, memlet_uses, cache)
            for read in found:
                if any(read.node is s for s in unit.statements):
                    continue  # Moves along
                if not guarded_anywhere(read) or not inside(read.node, scope) or inside(read.node, nest[0]):
                    return None
                container = child_containing(scope, read.node)
                if target is not None and container is not target:
                    return None
                target = container
                guard_names |= set().union(*(expression_symbols(c) for c in guard_candidates(read, scope)))
    if target is None:
        return None
    # Move only past the nests that assign what the readers' guards read, i.e., that prevent sinking it, and no further
    readers_position = next(k for k, c in enumerate(siblings) if c is target)
    blockers = [k for k in range(position + 1, readers_position) if assigned_in([siblings[k]]) & guard_names]
    if not blockers:
        return None
    target = siblings[blockers[-1] + 1]
    reordering = move_plan(unit, target, motion)
    if reordering is None:
        return None
    return unit, target, reordering.stay
