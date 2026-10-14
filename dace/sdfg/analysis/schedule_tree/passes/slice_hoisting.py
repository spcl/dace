# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Hoisting the invariant slice of a loop body out of the loop.

Statements in the body of a ``for`` loop may compute the same values in every iteration and still not be hoistable as
they are (see :func:`~dace.sdfg.analysis.schedule_tree.passes.code_motion.hoist_invariant_statements`): they run under
``if`` scopes, or read state that every iteration recomputes in the same way, such as a search position that each
iteration resets and then advances in ``while`` loops. :func:`hoist_invariant_slices` runs a copy of the work their
values depend on (their backward slice, with the scopes that control it) once before the loop, stores the values per
iteration of the loops around the statements, and lets the loop load them instead of computing them. The loop keeps
its control flow: only the stored computations leave it.
"""

import ast
import copy
from dataclasses import dataclass

import sympy

from dace import data, dtypes
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.dataflow import ReachingAnalysis
from dace.sdfg.analysis.schedule_tree.passes.common import (
    bound_names,
    clone_subtree,
    condition_element_accesses,
    condition_of,
    iteration_spaces,
    make_scope,
    memlets_of,
    names_read,
    names_written,
    repository_of,
    trip_count,
)
from dace.sdfg.analysis.schedule_tree.passes.tasklet_splitting import (
    EXPENSIVE_WEIGHT,
    TaskletCode,
    dependencies,
    operation_cost,
    parse_tasklet,
    tasklet_part,
    value_types,
)
from dace.sdfg.state import LoopRegion

# Nodes that create aliases or jump: loop bodies that contain them are not analyzed
_UNSUPPORTED = (tn.ViewNode, tn.NViewEnd, tn.RefSetNode, tn.GotoNode, tn.StateLabel, tn.BreakNode, tn.ContinueNode)

# Scopes the slice may run in (copied with their headers)
_COPYABLE = (tn.ForScope, tn.WhileScope, tn.IfScope, tn.ElifScope, tn.ElseScope)


@dataclass(eq=False)
class _Item:
    """A statement of a tasklet whose code is parsed (``index``), or another node as a whole (``index`` is ``None``)."""

    node: tn.ScheduleTreeNode
    index: int | None
    child: int  #: Position of the child of the loop body that contains it
    reads: set[str]  #: Container and symbol names read, including those in the indices of its memlets
    writes: set[str]  #: Container and symbol names written
    accesses: list[tuple[str, Memlet]]  #: Containers read, with the memlets read through
    opaque: bool = False  #: Whether it has side effects or assigns symbols


@dataclass
class _Stored:
    """A tasklet whose invariant statements (``removed``) leave the loop, and the values the loop loads instead."""

    node: tn.TaskletNode
    code: TaskletCode
    removed: set[int]
    values: list[str]
    saving: int
    types: dict[str, dtypes.typeclass]
    space: dict[str, list[tuple[str, int, str]]] = None  #: The dimensions of each value (see :func:`_store_space`)
    memlets: dict[str, Memlet] = None  #: Where each value is stored, once decided


class _LoopBody:
    """
    The statements of the body of a ``for`` loop and which of them compute the same values in every iteration.

    A statement is *variant* if it reads the loop variable, a symbol the body assigns or a variant name; if it runs
    under a scope whose header reads a variant name; or if it has side effects. A container is variant if a variant statement writes it, or if some read of it in the body may see a
    value written before the iteration (it is carried from one iteration to the next). Invariant statements under
    invariant control compute the same values in every iteration: every value they read was written in the same
    iteration by invariant statements, or before the loop.
    """

    def __init__(self, loop: tn.ForScope, reaching: ReachingAnalysis, repository):
        self.loop = loop
        self.reaching = reaching
        self.containers = repository.arrays
        self.codes: dict[int, TaskletCode] = {}  # Parsed tasklets, by id
        self.tasklets: dict[int, tn.TaskletNode] = {}
        self.items: list[_Item] = []
        self.statement_items: dict[tuple[int, int], _Item] = {}  # By the tasklet's id and the statement index
        self.node_items: dict[int, _Item] = {}  # Nodes that are items as a whole, by id
        self.scopes: list[tn.ScheduleTreeScope] = []  # In preorder
        self.supported = True
        bound: set[str] = set()
        for position, child in enumerate(loop.children):
            for node in child.preorder_traversal():
                if isinstance(node, _UNSUPPORTED):
                    self.supported = False
                if isinstance(node, tn.ScheduleTreeScope):
                    self.scopes.append(node)
                    bound |= bound_names(node)
                else:
                    self._add(node, position, repository)
        self.bound = bound  #: Loop and map variables bound in the body
        self.writers: dict[str, list[_Item]] = {}
        for item in self.items:
            for name in item.writes:
                self.writers.setdefault(name, []).append(item)
        # Whether something reads a loop variable of the body outside its loop (a copy of the loop before the loop
        # would change what the first iteration reads)
        self.escapes = any(self._escaped(i.reads, i.node) for i in self.items) or any(
            self._escaped(names_read(s), s) for s in self.scopes
        )
        self.variant_names: set[str] = set()
        self._variant_items: set[int] = set()
        self._variant_scopes: set[int] = set()

    def _add(self, node: tn.ScheduleTreeNode, position: int, repository) -> None:
        code = parse_tasklet(node) if isinstance(node, tn.TaskletNode) else None
        if code is None or any(c not in node.in_memlets for c in code.inputs):
            opaque = isinstance(node, tn.AssignNode) or (
                isinstance(node, (tn.TaskletNode, tn.LibraryCall)) and node.node.has_side_effects(repository)
            )
            accesses = [(m.data, m) for m in memlets_of(node, "in_memlets") + memlets_of(node, "memlet")]
            item = _Item(node, None, position, names_read(node), names_written(node), accesses, opaque)
            self.items.append(item)
            self.node_items[id(node)] = item
            return
        self.codes[id(node)] = code
        self.tasklets[id(node)] = node
        for index, target in enumerate(code.targets):
            reads, accesses = set(), []
            for name in code.reads[index]:
                if name in code.inputs:
                    memlet = node.in_memlets[name]
                    reads |= {memlet.data} | {str(s) for s in memlet.free_symbols}
                    accesses.append((memlet.data, memlet))
                elif code.defining(name) is None:
                    reads.add(name)  # Symbols (and builtins)
            writes = set()
            if target in code.outputs and target in node.out_memlets:
                memlet = node.out_memlets[target]
                writes.add(memlet.data)
                reads |= {str(s) for s in memlet.free_symbols}
            item = _Item(node, index, position, reads, writes, accesses)
            self.items.append(item)
            self.statement_items[(id(node), index)] = item

    # ------------------------------------------------------------------------------------------------------------------
    # Headers of scopes

    def chain(self, scope: tn.ScheduleTreeScope) -> list[tn.ScheduleTreeScope]:
        """An ``else``/``elif`` scope and the branches before it, up to the ``if`` (just the scope for others)."""
        result = [scope]
        if isinstance(scope, (tn.ElifScope, tn.ElseScope)):
            siblings = scope.parent.children
            position = next(k for k, c in enumerate(siblings) if c is scope)
            while position > 0 and isinstance(siblings[position], (tn.ElifScope, tn.ElseScope)):
                position -= 1
                result.append(siblings[position])
        return result

    def header_reads(self, scope: tn.ScheduleTreeScope) -> set[str]:
        """The names the header of a scope reads, with the conditions of the branches before an ``else``/``elif``."""
        return set().union(*(names_read(s) - bound_names(s) for s in self.chain(scope)))

    def header_accesses(self, scope: tn.ScheduleTreeScope) -> list[tuple[str, Memlet]]:
        """The containers the header of a scope itself reads, with the elements read (all of them if not known)."""
        if isinstance(scope, tn.WhileScope):
            code = scope.loop.loop_condition
            single = (
                code.language == dtypes.Language.Python and len(code.code) == 1 and isinstance(code.code[0], ast.Expr)
            )
            expression = code.code[0].value if single else None
        else:
            expression = condition_of(scope)
        if expression is not None:
            found = condition_element_accesses(expression, self.containers)
            if found is not None:
                return [(name, Memlet(element)) for name, element, _, _ in found]
        names = (names_read(scope) - bound_names(scope)) & self.containers.keys()
        return [(name, Memlet.from_array(name, self.containers[name])) for name in sorted(names)]

    # ------------------------------------------------------------------------------------------------------------------
    # Variance

    def _escaped(self, names: set[str], node: tn.ScheduleTreeNode) -> bool:
        """Whether ``names`` contain a variable of a loop of the body that is not bound around ``node``."""
        free = names & self.bound
        scope = node if isinstance(node, tn.ScheduleTreeScope) else node.parent
        while free and scope is not self.loop:
            free -= bound_names(scope)
            scope = scope.parent
        return bool(free)

    def analyze(self) -> None:
        """Finds the variant statements, scopes and names."""
        self.variant_names = {self.loop.loop.loop_variable}
        checked: set[str] = set()  # Containers whose reads were checked for values from earlier iterations
        while True:
            self._propagate()
            written = {n for n in self.writers if n in self.containers} - self.variant_names - checked
            reads: dict[str, list[tuple[tn.ScheduleTreeNode, Memlet]]] = {}
            for item in self.items:
                if not self.variant(item):
                    for name, memlet in item.accesses:
                        if name in written:
                            reads.setdefault(name, []).append((item.node, memlet))
            for scope in self.scopes:
                if not self.scope_variant(scope):
                    for name, memlet in self.header_accesses(scope):
                        if name in written:
                            reads.setdefault(name, []).append((scope, memlet))
            carried = {
                name
                for name, accesses in reads.items()
                if any(
                    self.reaching.may_reach_later(node, memlet, node, memlet, self.loop) for node, memlet in accesses
                )
            }
            checked |= set(reads)
            if not carried:
                return
            self.variant_names |= carried

    def _propagate(self) -> None:
        changed = True
        while changed:
            changed = False
            for scope in self.scopes:
                if id(scope) in self._variant_scopes:
                    continue
                if (scope.parent is not self.loop and id(scope.parent) in self._variant_scopes) or self.header_reads(
                    scope
                ) & self.variant_names:
                    self._variant_scopes.add(id(scope))
                    changed = True
            for item in self.items:
                if id(item) in self._variant_items:
                    continue
                code = self.codes.get(id(item.node))
                local = code is not None and any(
                    id(self.statement_items[(id(item.node), code.defining(n))]) in self._variant_items
                    for n in code.reads[item.index]
                    if code.defining(n) is not None
                )
                parent = item.node.parent
                if (
                    item.opaque
                    or local
                    or item.reads & self.variant_names
                    or (parent is not self.loop and id(parent) in self._variant_scopes)
                ):
                    self._variant_items.add(id(item))
                    self.variant_names |= item.writes
                    changed = True

    def variant(self, item: _Item) -> bool:
        return id(item) in self._variant_items

    def scope_variant(self, scope: tn.ScheduleTreeScope) -> bool:
        return id(scope) in self._variant_scopes


def _store_space(node: tn.ScheduleTreeNode, body: _LoopBody) -> list[tuple[str, int, str]] | None:
    """``(variable, extent, index)`` of each ``for`` loop around ``node`` in the body (outermost first): arrays with these
    dimensions hold a value per execution of ``node`` in an iteration of the loop, if the loops have constant bounds
    and unit stride and the other scopes around ``node`` are ``if``/``elif``/``else`` scopes (which run at most once per
    iteration of the loops around them). ``None`` otherwise."""
    result = []
    scope = node.parent
    while scope is not body.loop:
        if type(scope) is tn.ForScope:
            variable = scope.loop.loop_variable
            space = body.reaching.ranges(scope).get(variable)
            if (
                not variable
                or any(variable == v for v, _, _ in result)
                or space is None
                or not all(sympy.sympify(v).is_Integer for v in space)
                or abs(space[2]) != 1
            ):
                return None
            low, high = sorted((int(space[0]), int(space[1])))
            result.insert(0, (variable, high - low + 1, f"{variable} - {low}" if low else variable))
        elif not isinstance(scope, (tn.IfScope, tn.ElifScope, tn.ElseScope)) or isinstance(scope, tn.StateIfScope):
            return None
        scope = scope.parent
    return result


def _dimensions(node: tn.TaskletNode, code: TaskletCode, body: _LoopBody, variables: set[str]) -> list[set[str]]:
    """For each statement of a tasklet, the ``variables`` (of loops around it in the body) its value may depend on:
    those in its code and in the memlets of the connectors it reads, transitively over the statements it reads, or all
    if it reads a container that the body writes (whose elements may differ between iterations)."""
    result: list[set[str]] = []
    for k, names in enumerate(code.reads):
        found = names & variables
        for name in names:
            if name in code.inputs:
                memlet = node.in_memlets[name]
                found |= {str(s) for s in memlet.free_symbols} & variables
                if memlet.data in body.writers:
                    found |= variables
            elif code.defining(name) is not None:
                found |= result[code.defining(name)]
        result.append(found)
    return result


def _stored_tasklets(body: _LoopBody, min_saving: int) -> list[_Stored]:
    """The tasklets of the body whose invariant statements are worth computing once (see :class:`_Stored`): values
    that differ between iterations of the loops around them in the body (others are the compiler's business), saving
    at least ``min_saving``."""
    result = []
    for key, code in body.codes.items():
        node = body.tasklets[key]
        space = _store_space(node, body)
        if space is None:
            continue
        dimensions = _dimensions(node, code, body, {v for v, _, _ in space})
        statements = range(len(code.statements))
        # Copies are as cheap to repeat as to load; values the same in all iterations are the compiler's business
        removed = {
            k
            for k in statements
            if not body.variant(body.statement_items[(key, k)]) and not code.trivial(k) and dimensions[k]
        }
        if not removed:
            continue
        kept = [k for k in statements if k not in removed]
        needed = set().union(*(code.reads[k] for k in kept)) if kept else set()
        values = [code.targets[k] for k in sorted(removed) if code.targets[k] in code.outputs | needed]
        only_removed = {n for k in removed for n in code.reads[k] if n in code.inputs} - needed
        saving = sum(operation_cost(code.statements[k]) for k in removed) - len(values) + len(only_removed)
        types = value_types(node, body.containers)
        if saving >= max(min_saving, 1) and all(types.get(v) not in (None, dtypes.typeclass(None)) for v in values):
            entry = _Stored(node, code, removed, values, saving, types)
            entry.space = {code.targets[k]: [d for d in space if d[0] in dimensions[k]] for k in removed}
            result.append(entry)
    return result


@dataclass
class _Slice:
    """The items and scopes of a loop body that the values of stored tasklets depend on."""

    items: set[int]
    statements: dict[int, set[int]]  #: Statement indices of the items in each tasklet, by the tasklet's id
    scopes: set[int]
    read: set[str]  #: Containers read by the items and the headers of the scopes
    written: set[str]  #: Containers written by the items
    cost: int  #: Rough cost of the items (see :func:`operation_cost`)
    last: int  #: Position of the last child of the loop body that contains a stored tasklet


def _backward_slice(body: _LoopBody, stored: list[_Stored]) -> _Slice | None:
    """The statements and scopes that the stored values depend on, in the children of the loop body up to the last
    one that contains a stored tasklet, or ``None`` if one of them is variant or cannot be copied."""
    pending: list[_Item] = []
    for entry in stored:
        defining = {entry.code.defining(v) for v in entry.values}
        pending += [body.statement_items[(id(entry.node), k)] for k in dependencies(entry.code, defining)]
    last = max(item.child for item in pending)
    result = _Slice(set(), {}, set(), set(), set(), 0, last)

    def depend_on(names: set[str]) -> None:
        for name in names:
            pending.extend(w for w in body.writers.get(name, []) if w.child <= last)

    while pending:
        item = pending.pop()
        if id(item) in result.items:
            continue
        if body.variant(item):
            return None
        result.items.add(id(item))
        code = body.codes.get(id(item.node))
        if code is not None:
            result.statements.setdefault(id(item.node), set()).add(item.index)
            pending += [body.statement_items[(id(item.node), k)] for k in dependencies(code, {item.index})]
            result.cost += operation_cost(code.statements[item.index])
        else:
            result.cost += 1
            result.read |= item.writes & body.containers.keys()  # The copy writes them all
        result.read |= {name for name, _ in item.accesses}
        result.written |= item.writes & body.containers.keys()
        depend_on(item.reads)
        scope = item.node.parent
        while scope is not body.loop:
            if not isinstance(scope, _COPYABLE) or isinstance(scope, tn.StateIfScope):
                return None
            for member in body.chain(scope):
                if id(member) not in result.scopes:
                    if body.scope_variant(member):
                        return None
                    result.scopes.add(id(member))
                    result.read |= {name for name, _ in body.header_accesses(member)}
                    depend_on(names_read(member) - bound_names(member))
            scope = scope.parent
    for name in result.read | result.written:
        desc = body.containers[name]
        if (
            not isinstance(desc, (data.Scalar, data.Array))
            or isinstance(desc, (data.View, data.Reference))
            or desc.may_alias
        ):
            return None
    return result


def _rename_code(code: CodeBlock, renamed: dict[str, str]) -> CodeBlock:
    code = copy.deepcopy(code)
    for statement in code.code:
        astutils.ASTFindReplace(dict(renamed)).visit(statement)
    return code


def _copy_slice(
    children: list[tn.ScheduleTreeNode],
    body: _LoopBody,
    cut: _Slice,
    stored: dict[int, _Stored],
    renamed: dict[str, str],
) -> list[tn.ScheduleTreeNode]:
    """The slice of ``children`` (of a scope in the loop body): copies of its items and of the scopes around them, with
    the containers they write renamed."""
    result = []
    position = 0
    while position < len(children):
        node = children[position]
        if type(node) is tn.IfScope:
            members = [node]
            while position + len(members) < len(children) and isinstance(
                children[position + len(members)], (tn.ElifScope, tn.ElseScope)
            ):
                members.append(children[position + len(members)])
            needed = [k for k, member in enumerate(members) if id(member) in cut.scopes]
            for member in members[: max(needed) + 1] if needed else []:
                result.append(_copy_scope(member, _copy_slice(member.children, body, cut, stored, renamed), renamed))
            position += len(members)
            continue
        position += 1
        if isinstance(node, tn.ScheduleTreeScope):
            if id(node) in cut.scopes:
                result.append(_copy_scope(node, _copy_slice(node.children, body, cut, stored, renamed), renamed))
            continue
        if id(node) in body.node_items:
            if id(body.node_items[id(node)]) in cut.items:
                copied = clone_subtree(node)
                for memlet in (
                    memlets_of(copied, "in_memlets") + memlets_of(copied, "out_memlets") + memlets_of(copied, "memlet")
                ):
                    memlet.data = renamed.get(memlet.data, memlet.data)
                result.append(copied)
            continue
        indices = cut.statements.get(id(node))
        if not indices:
            continue
        code = body.codes[id(node)]
        outputs = []
        for k in sorted(indices):
            target = code.targets[k]
            memlet = node.out_memlets.get(target) if target in code.outputs else None
            if memlet is not None and memlet.data in cut.read:  # Only what the slice reads back
                memlet = copy.deepcopy(memlet)
                memlet.data = renamed[memlet.data]
                outputs.append((target, memlet))
        entry = stored.get(id(node))
        if entry is not None:
            outputs += [(value, entry.memlets[value]) for value in entry.values]
        types = entry.types if entry is not None else value_types(node, body.containers)
        part = tasklet_part(node, code, indices, outputs, [], "slice", types)
        for memlet in part.in_memlets.values():
            memlet.data = renamed.get(memlet.data, memlet.data)
        result.append(part)
    return result


def _copy_scope(
    scope: tn.ScheduleTreeScope, children: list[tn.ScheduleTreeNode], renamed: dict[str, str]
) -> tn.ScheduleTreeScope:
    if type(scope) is tn.ForScope:
        return make_scope(scope, None, None, None, children, 1)
    if type(scope) is tn.WhileScope:
        header = LoopRegion(
            f"{scope.loop.label}_slice", condition_expr=_rename_code(scope.loop.loop_condition, renamed)
        )
        return tn.WhileScope(loop=header, children=children)
    if type(scope) is tn.IfScope:
        return tn.IfScope(condition=_rename_code(scope.condition, renamed), children=children)
    if type(scope) is tn.ElifScope:
        return tn.ElifScope(condition=_rename_code(scope.condition, renamed), children=children)
    return tn.ElseScope(children=children)


def hoist_invariant_slices(
    stree: tn.ScheduleTreeScope, min_trip_count: int = 2, min_saving: int = EXPENSIVE_WEIGHT
) -> int:
    """
    Compute values that are the same in every iteration of a ``for`` loop once before it, together with everything
    they depend on in the loop body (their backward slice): statements of tasklets in the body that compute the same
    values in every iteration (see :class:`_LoopBody`) and run at most once per iteration of the ``for`` loops around
    them in the body (they may run under ``if`` scopes, not in ``while`` loops) leave the loop if that saves work (see
    :func:`~dace.sdfg.analysis.schedule_tree.passes.tasklet_splitting.operation_cost`, divisions and calls count
    most). A copy of the slice runs before the loop and stores their values in new transient arrays, indexed by the
    variables of the loops around each statement that its value depends on (values that depend on none are left to
    the compiler); the loop loads them instead. The copy keeps the scopes (loops, ``while`` loops, branches) that
    control the slice and writes new containers instead of those the slice writes, so the loop runs as before
    otherwise.

    Only loops that run at least ``min_trip_count`` times (a known constant) and contain loops are considered, outer
    loops first; work in innermost loops is left to the compiler.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param min_trip_count: Only hoist out of loops that run at least this many times.
    :param min_saving: Only hoist statements of a tasklet whose removal from the loop saves at least this much per
                       execution (by default, about a division; see :func:`operation_cost`).
    :return: The number of loops whose invariant slices were hoisted.
    """
    root = stree.get_root()
    repository = repository_of(root)
    loops = [
        n
        for n in stree.preorder_traversal()
        if type(n) is tn.ForScope
        and n.loop.loop_variable
        and any(isinstance(c, (tn.LoopScope, tn.MapScope)) for c in n.children)
    ]
    hoisted = 0
    reaching = ReachingAnalysis(root)
    for loop in loops:
        if loop.parent is None:
            continue
        spaces = iteration_spaces(loop, repository)
        trips = trip_count(spaces[0][2]) if len(spaces) == 1 and spaces[0][2] is not None else None
        if trips is None or trips < max(min_trip_count, 1):
            continue
        if _hoist_slice(loop, trips, reaching, repository, min_saving):
            hoisted += 1
            reaching = ReachingAnalysis(root)
    return hoisted


def _hoist_slice(loop: tn.ForScope, trips: int, reaching: ReachingAnalysis, repository, min_saving: int) -> bool:
    body = _LoopBody(loop, reaching, repository)
    if not body.supported or body.escapes or not body.codes:
        return False
    body.analyze()
    stored = _stored_tasklets(body, min_saving)
    if not stored:
        return False
    cut = _backward_slice(body, stored)
    if cut is None:
        return False
    saving = sum(entry.saving for entry in stored)
    if trips * saving <= cut.cost + sum(len(entry.values) for entry in stored):
        return False

    containers = body.containers
    # What the copy writes goes to new containers (only those it reads back are written, see _copy_slice)
    renamed: dict[str, str] = {}
    for name in sorted(cut.written & cut.read):
        desc = copy.deepcopy(containers[name])
        desc.transient = True
        desc.lifetime = dtypes.AllocationLifetime.Scope
        renamed[name] = data.find_new_name(f"{name}_slice", containers)
        containers[renamed[name]] = desc
    for entry in stored:
        entry.memlets = {}
        for value in entry.values:
            name = data.find_new_name(f"{entry.node.node.label}_{value}", containers)
            containers[name] = data.Array(entry.types[value], [n for _, n, _ in entry.space[value]], transient=True)
            entry.memlets[value] = Memlet(f"{name}[{', '.join(i for _, _, i in entry.space[value])}]")

    copied = _copy_slice(loop.children[: cut.last + 1], body, cut, {id(entry.node): entry for entry in stored}, renamed)
    for entry in stored:
        code = entry.code
        kept = [k for k in range(len(code.statements)) if k not in entry.removed]
        outputs = [(o, m) for o, m in entry.node.out_memlets.items() if o in code.outputs]
        loaded = [(v, entry.memlets[v]) for v in entry.values]
        rest = tasklet_part(entry.node, code, kept, outputs, loaded, "rest", entry.types)
        parent = entry.node.parent
        children = [rest if c is entry.node else c for c in parent.children]
        parent.children = []
        parent.add_children(children)
    parent = loop.parent
    children = list(parent.children)
    position = next(k for k, c in enumerate(children) if c is loop)
    parent.children = []
    parent.add_children(children[:position] + copied + children[position:])
    return True
