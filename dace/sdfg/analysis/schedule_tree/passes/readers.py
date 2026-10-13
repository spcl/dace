# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which statements read the values a statement writes: reads of containers by tasklets and ``if`` conditions, the
guards they are under, and the reads that a written value may reach."""

import ast

from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.dataflow import ReachingAnalysis
from dace.sdfg.analysis.schedule_tree.passes.common import (
    ancestors,
    condition_element_accesses,
    condition_of,
    conjuncts,
    inside,
    is_pure,
    memlet_element,
    names_read,
    normalized_indices,
    path_to,
)
from dace.sdfg.analysis.schedule_tree.passes.transient_reuse import Liveness, Use


class Read:
    """A read of a container: by the memlets of a tasklet, or in the condition of an ``if`` scope (then with the
    conjuncts of the condition before the first one that reads it)."""

    def __init__(
        self, node: tn.ScheduleTreeNode, elements: list[tuple[str, ...] | None], preceding: list[ast.expr] | None = None
    ):
        self.node = node
        self.elements = elements
        self.preceding = preceding or []


def collect_reads(root: tn.ScheduleTreeRoot) -> tuple[dict[str, list[Read]], set[str]]:
    """The reads of every container by tasklets and ``if`` conditions, and the containers read in other ways (which
    are left alone)."""
    containers = root.containers
    reads: dict[str, list[Read]] = {}
    other: set[str] = set()
    for node in root.preorder_traversal():
        if isinstance(node, tn.TaskletNode):
            by_name: dict[str, list] = {}
            for memlet in node.in_memlets.values():
                by_name.setdefault(memlet.data, []).append(memlet_element(memlet))
            for name, elements in by_name.items():
                reads.setdefault(name, []).append(Read(node, elements))
            other |= set(map(str, node.node.free_symbols)) & containers.keys()
            continue
        condition = condition_of(node) if type(node) is tn.IfScope else None
        accesses = None if condition is None else condition_element_accesses(condition, containers)
        if accesses is None:
            other |= names_read(node) & containers.keys()
            continue
        parts = conjuncts(condition)
        found: dict[str, tuple[list, int]] = {}
        for name, _, access, _ in accesses:
            position = next(k for k, c in enumerate(parts) if any(n is access for n in ast.walk(c)))
            if isinstance(access, ast.Subscript):
                indices = access.slice.elts if isinstance(access.slice, ast.Tuple) else [access.slice]
                element = normalized_indices([ast.unparse(i) for i in indices])
            else:  # A single-element container by name
                element = ("0",)
            elements, first = found.get(name, ([], position))
            found[name] = (elements + [element], min(first, position))
        for name, (elements, position) in found.items():
            reads.setdefault(name, []).append(Read(node, elements, parts[:position]))
    return reads, other


def guard_candidates(read: Read, scope: tn.ScheduleTreeScope) -> list[ast.expr]:
    """The conditions that hold whenever ``read`` happens, from the ``if`` scopes between it and ``scope``."""
    candidates = list(read.preceding)
    for node in path_to(read.node, scope)[1:]:
        if type(node) is tn.IfScope:
            condition = condition_of(node)
            if condition is not None and is_pure(condition):
                candidates += conjuncts(condition)
    return candidates


def guarded_anywhere(read: Read) -> bool:
    """Whether some ``if`` condition holds whenever ``read`` happens (a quick filter before the full analysis)."""
    return bool(read.preceding) or any(type(n) is tn.IfScope for n in ancestors(read.node))


def some_read_guarded(statement: tn.ScheduleTreeNode, reads: list[Read]) -> bool:
    """Whether a container that ``statement`` writes is never read by others, or some read is under a guard."""
    others = [r for r in reads if r.node is not statement]
    return not others or any(guarded_anywhere(r) for r in others)


def collect_memlet_uses(root: tn.ScheduleTreeRoot) -> dict[str, list[Use]]:
    """The accesses of every container through the memlets of statements (see ``Liveness``)."""
    uses: dict[str, list[Use]] = {}
    for node in root.preorder_traversal():
        for attr, write in (("in_memlets", False), ("out_memlets", True)):
            memlets = getattr(node, attr, None)
            if isinstance(memlets, dict):
                for connector, memlet in memlets.items():
                    uses.setdefault(memlet.data, []).append(Use(node, memlets, connector, write))
    return uses


def read_memlets(read: Read, name: str) -> list[Memlet] | None:
    """The accesses of ``name`` by a read (``None`` if some element is not known)."""
    if isinstance(read.node, tn.TaskletNode):
        return [m for m in read.node.in_memlets.values() if m.data == name]
    if any(element is None for element in read.elements):  # In a condition
        return None
    return [Memlet(f"{name}[{', '.join(element)}]") for element in read.elements]


def reached(
    statement: tn.TaskletNode,
    name: str,
    read: Read,
    reaching: ReachingAnalysis,
    later_in: tn.ScheduleTreeScope | None = None,
) -> bool:
    """Whether a value ``statement`` writes to ``name`` may be what ``read`` reads (with ``later_in``: in a later
    execution of that scope than the one that wrote it)."""
    accessed = read_memlets(read, name)
    if accessed is None:
        return True
    writes = [w for w in statement.out_memlets.values() if w.data == name]
    if later_in is not None:
        return any(reaching.may_reach_later(statement, w, read.node, r, later_in) for w in writes for r in accessed)
    return any(reaching.may_reach(statement, w, read.node, r) for w in writes for r in accessed)


def value_reads(
    statement: tn.TaskletNode,
    name: str,
    reads: list[Read],
    liveness: Liveness,
    reaching: ReachingAnalysis,
    memlet_uses: dict[str, list[Use]],
    cache: dict[str, tuple],
) -> tuple[list[Read], tn.ScheduleTreeScope | None]:
    """The reads that may see a value ``statement`` writes to ``name`` (those in its live segment that the value may
    reach: a segment may span unrelated values of a container that several temporaries share), and the lowest scope
    above the statement that contains those reads, in which no value of the statement is read in a later execution
    than the one that wrote it and, if the statement reads ``name`` too (as an accumulation does), in which ``name`` is
    written before it is read: no value flows into one execution of the scope from outside or from a previous one.
    ``(reads, None)`` if there is no such scope."""
    uses = list(memlet_uses.get(name, []))
    for read in reads:
        if not isinstance(read.node, tn.TaskletNode):  # Reads in conditions
            for element in read.elements:
                memlet = Memlet(f"{name}[{', '.join(element)}]")
                uses.append(Use(read.node, {"__cond": memlet}, "__cond", False))
    if name not in cache:  # The same for every statement writing ``name`` (until the tree changes)
        cache[name] = liveness.segments(uses)[0]
    segments = cache[name]
    position = liveness.position[id(statement)]
    segment = next(((first, last) for first, last in segments if first <= position <= last), None)
    if segment is None:
        return [], statement.parent  # The values are never read

    def in_segment(node: tn.ScheduleTreeNode) -> bool:
        return segment[0] <= liveness.position[id(node)] <= segment[1]

    found = [
        r for r in reads if r.node is not statement and in_segment(r.node) and reached(statement, name, r, reaching)
    ]
    accumulates = any(m.data == name for m in statement.in_memlets.values())
    scope = statement.parent
    while scope is not None:
        if all(inside(r.node, scope) for r in found) and not any(
            reached(statement, name, r, reaching, later_in=scope) for r in found
        ):
            if not accumulates or not liveness.exposed_in(scope, [u for u in uses if inside(u.node, scope)]):
                return found, scope
        scope = scope.parent
    return found, None
