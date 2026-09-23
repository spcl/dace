# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Assortment of passes for schedule trees.
"""

import ast
import copy
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import sympy

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion


def remove_unused_and_duplicate_labels(stree: tn.ScheduleTreeScope):
    """
    Removes unused and duplicate labels from the schedule tree.

    :param stree: The schedule tree to remove labels from.
    """

    class FindGotos(tn.ScheduleNodeVisitor):

        def __init__(self):
            self.gotos: Set[str] = set()

        def visit_GotoNode(self, node: tn.GotoNode):
            if node.target is not None:
                self.gotos.add(node.target)

    class RemoveLabels(tn.ScheduleNodeTransformer):

        def __init__(self, labels_to_keep: Set[str]) -> None:
            self.labels_to_keep = labels_to_keep
            self.labels_seen = set()

        def visit_StateLabel(self, node: tn.StateLabel):
            if node.state.name not in self.labels_to_keep:
                return None
            if node.state.name in self.labels_seen:
                return None
            self.labels_seen.add(node.state.name)
            return node

    fg = FindGotos()
    fg.visit(stree)
    return RemoveLabels(fg.gotos).visit(stree)


def remove_empty_scopes(stree: tn.ScheduleTreeScope):
    """
    Removes empty scopes from the schedule tree.

    :warning: This pass is not safe to use for for-loops, as it will remove indices that may be used after the loop.
    """

    class RemoveEmptyScopes(tn.ScheduleNodeTransformer):

        def visit_scope(self, node: tn.ScheduleTreeScope):
            if len(node.children) == 0:
                return None

            return self.generic_visit(node)

    return RemoveEmptyScopes().visit(stree)


# ----------------------------------------------------------------------------------------------------------------------
# Loop range reduction
# ----------------------------------------------------------------------------------------------------------------------


def _memlets(node: tn.ScheduleTreeNode, attr: str) -> list:
    memlets = getattr(node, attr, None)
    if memlets is None:
        return []
    if isinstance(memlets, Memlet):
        return [memlets]
    return list(memlets.values()) if isinstance(memlets, dict) else list(memlets)


def _names_read(node: tn.ScheduleTreeNode) -> Set[str]:
    """Symbol and container names a single tree node reads (not those of its children)."""
    read: Set[str] = set()
    for memlet in _memlets(node, 'in_memlets') + _memlets(node, 'memlet'):
        read |= memlet.free_symbols | {memlet.data}
    for memlet in _memlets(node, 'out_memlets'):
        read |= memlet.free_symbols
    for code in (getattr(node, 'condition', None), getattr(node, 'value', None)):
        if isinstance(code, CodeBlock):
            read |= code.get_free_symbols()
    if isinstance(node, tn.LoopScope):
        read |= {s for code in node.loop.get_meta_codeblocks() for s in code.get_free_symbols()}
    if getattr(node, 'node', None) is not None:  # Tasklets, library nodes, map entries
        read |= set(node.node.free_symbols)
    return read


def _names_written(node: tn.ScheduleTreeNode) -> Set[str]:
    """Symbol and container names a single tree node assigns (not those of its children)."""
    written = {memlet.data for memlet in _memlets(node, 'out_memlets')}
    for attr in ('name', 'target'):
        if isinstance(getattr(node, attr, None), str):
            written.add(getattr(node, attr))
    if isinstance(node, tn.LoopScope) and node.loop.loop_variable:
        written.add(node.loop.loop_variable)
    if isinstance(node, tn.MapScope):
        written |= set(node.node.map.params)
    return written


def _in_subtrees(nodes: Iterable[tn.ScheduleTreeNode], names_of) -> Set[str]:
    return set().union(*(names_of(n) for node in nodes for n in node.preorder_traversal()))


def _clone(node: tn.ScheduleTreeNode) -> tn.ScheduleTreeNode:
    """A copy of a subtree that owns its SDFG-side nodes and memlets (which the SDFG conversion inserts into the
    graph) but shares the descriptors, states and loop headers it merely references."""
    new = copy.copy(node)
    new.parent = None
    if isinstance(node, tn.ScheduleTreeScope):
        new.children = []
        new.add_children([_clone(c) for c in node.children])
    for attr in ('node', 'in_memlets', 'out_memlets', 'memlet', 'edge', 'value', 'condition'):
        if getattr(new, attr, None) is not None:
            setattr(new, attr, copy.deepcopy(getattr(new, attr)))
    return new


def _bound_names(node: tn.ScheduleTreeNode) -> Set[str]:
    """Iteration variables a loop or map binds for its own scope."""
    if isinstance(node, tn.MapScope):
        return set(node.node.map.params)
    if isinstance(node, tn.ForScope) and node.loop.loop_variable:
        return {node.loop.loop_variable}
    return set()


class _AccessIndex:
    """How many nodes of a tree read, and read or write, each name freely, i.e., other than as the iteration variable
    of an enclosing loop or map that binds it. Answers whether a name is used outside a scope in time proportional to
    the scope rather than the tree. Counts are only ever added to as the tree changes (removing nodes leaves stale
    counts), so the answers err on the side of "used"."""

    def __init__(self, root: tn.ScheduleTreeScope):
        self.reads: Dict[str, int] = {}
        self.accesses: Dict[str, int] = {}
        self._names: Dict[int, tuple] = {}  # Holds the node, so that ids are not reused
        self._inside: Dict[int, tuple] = {}
        self.add(root)

    def names(self, node: tn.ScheduleTreeNode) -> Tuple[Set[str], Set[str]]:
        """The names ``node`` itself reads, and reads or writes (computed once per node)."""
        entry = self._names.get(id(node))
        if entry is None:
            read = _names_read(node)
            entry = self._names[id(node)] = (node, read, read | _names_written(node))
        return entry[1], entry[2]

    def _count(self, node: tn.ScheduleTreeNode, reads: Dict[str, int], accesses: Dict[str, int],
               bound: Set[str]) -> None:
        bound = bound | _bound_names(node)
        read, accessed = self.names(node)
        for name in read - bound:
            reads[name] = reads.get(name, 0) + 1
        for name in accessed - bound:
            accesses[name] = accesses.get(name, 0) + 1
        for child in getattr(node, 'children', ()):
            self._count(child, reads, accesses, bound)

    def add(self, node: tn.ScheduleTreeNode, parent: Optional[tn.ScheduleTreeScope] = None) -> None:
        """Account for a new subtree, placed under ``parent`` (by default, its current parent)."""
        bound, ancestor = set(), node.parent if parent is None else parent
        while ancestor is not None:
            bound |= _bound_names(ancestor)
            ancestor = ancestor.parent
        self._count(node, self.reads, self.accesses, bound)
        self._inside.clear()

    def used_outside(self, scope: tn.ScheduleTreeScope, name: str, reads_only: bool = False) -> bool:
        """Whether ``name`` may be accessed (or only: read) outside ``scope``."""
        ancestor = scope.parent
        while ancestor is not None:
            if name in _bound_names(ancestor):
                return True  # Accesses in the enclosing binding scope are not counted; assume they exist
            ancestor = ancestor.parent
        entry = self._inside.get(id(scope))
        if entry is None:
            reads, accesses = {}, {}
            self._count(scope, reads, accesses, set())
            entry = self._inside[id(scope)] = (scope, reads, accesses)
        if reads_only:
            return self.reads.get(name, 0) > entry[1].get(name, 0)
        return self.accesses.get(name, 0) > entry[2].get(name, 0)

    def changed(self) -> None:
        """Forget the per-scope counts after the tree changed."""
        self._inside.clear()

    def read_outside(self, scope: tn.ScheduleTreeScope, name: str) -> bool:
        return self.used_outside(scope, name, reads_only=True)


def _access(memlet: Memlet, containers: Dict[str, data.Data]) -> ast.expr:
    """How a condition reads the data a memlet points to: scalars by name, other containers by subscript."""
    if isinstance(containers.get(memlet.data, None), data.Scalar):
        return ast.Name(id=memlet.data, ctx=ast.Load())
    return ast.parse(f'{memlet.data}[{memlet.subset}]', mode='eval').body


def _assignment(node: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> Optional[Tuple[str, ast.expr]]:
    """A node as ``(target, value)`` if it assigns a single value that a condition may use in its place: a symbol
    assignment, a single-element copy, or a Python tasklet ``out = <expression of single-element inputs>``."""
    from dace.frontend.python import astutils  # Avoid import loops
    if isinstance(node, tn.AssignNode):
        code = node.value.code[0]
        return node.name, astutils.copy_tree(getattr(code, 'value', code))
    if isinstance(node, tn.CopyNode) and node.memlet.subset is not None and node.memlet.subset.num_elements() == 1:
        return node.target, _access(node.memlet, containers)
    if isinstance(node, tn.TaskletNode) and node.node.language == dtypes.Language.Python:
        code, outputs = node.node.code.code, list(node.out_memlets.items())
        if (len(outputs) == 1 and len(code) == 1 and isinstance(code[0], ast.Assign) and len(code[0].targets) == 1
                and isinstance(code[0].targets[0], ast.Name) and code[0].targets[0].id == outputs[0][0]
                and all(m.subset.num_elements() == 1 for _, m in outputs + list(node.in_memlets.items()))):
            value = astutils.copy_tree(code[0].value)
            value = astutils.ASTFindReplace({
                c: _access(m, containers)
                for c, m in node.in_memlets.items()
            }).visit(value)
            return outputs[0][1].data, value
    return None


def _make_scope(scope: tn.ScheduleTreeScope, dim: Optional[int], space, interval, body: list,
                k: int) -> tn.ScheduleTreeScope:
    """A copy of the loop or map ``scope`` whose iteration space ``space`` (dimension ``dim`` of a map) is restricted
    to ``interval`` (``None`` for unchanged), with the given body. ``k`` numbers the copies of a loop."""
    lrr = _lrr()
    if isinstance(scope, tn.MapScope):
        entry = copy.deepcopy(scope.node)
        if interval is not None:
            entry.map.range = lrr.reduced_map_range(entry.map.range, dim, space, interval)
        return tn.MapScope(node=entry, children=body, state=scope.state)
    init, condition = lrr.reduced_loop_header(space, interval) if interval is not None else (None, None)
    old = scope.loop
    header = LoopRegion(old.label if k == 0 else f'{old.label}_{k}',
                        condition_expr=CodeBlock(condition) if condition else copy.deepcopy(old.loop_condition),
                        loop_var=old.loop_variable,
                        initialize_expr=CodeBlock(init) if init else copy.deepcopy(old.init_statement),
                        update_expr=copy.deepcopy(old.update_statement),
                        unroll=old.unroll,
                        unroll_factor=old.unroll_factor)
    return tn.ForScope(loop=header, children=body)


def _ordered(intervals: list) -> Optional[List[int]]:
    """Indices of ``intervals`` in iteration order, or ``None`` if some pair cannot be ordered."""
    lrr = _lrr()
    if any(not (lrr.provably_before(a, b) or lrr.provably_before(b, a)) for i, a in enumerate(intervals)
           for b in intervals[i + 1:]):
        return None
    return sorted(range(len(intervals)), key=lambda i: sum(lrr.provably_before(b, intervals[i]) for b in intervals))


def _clone_body(scope: tn.ScheduleTreeScope, body: list, used_outside: Callable[[tn.ScheduleTreeScope, str],
                                                                                bool]) -> list:
    """Copies of ``body`` (children of ``scope``). The copies of a map body own the transients local to the map (as
    ``replicate_scope`` does for SDFG scopes), so that sibling copies of a map do not share temporaries. Loop bodies
    keep their names: the iterations of a loop run in order, so a value one of them writes may be read by a later one,
    which could be in another copy.

    :param used_outside: Whether a name is used outside a scope; transients that are keep their names.
    """
    root = scope.get_root()
    clones = [_clone(n) for n in body]
    renames = {}
    for name in (_in_subtrees(body, _names_written) if isinstance(scope, tn.MapScope) else ()):
        desc = root.containers.get(name, None)
        if desc is not None and desc.transient and not used_outside(scope, name):
            renames[name] = data.find_new_name(name, root.containers)
            root.containers[renames[name]] = copy.deepcopy(desc)
    if not renames:
        return clones
    from dace.frontend.python import astutils  # Avoid import loops
    for node in (n for clone in clones for n in clone.preorder_traversal()):
        for memlet in _memlets(node, 'in_memlets') + _memlets(node, 'out_memlets') + _memlets(node, 'memlet'):
            memlet.data = renames.get(memlet.data, memlet.data)
        for attr in ('name', 'target'):
            if getattr(node, attr, None) in renames:
                setattr(node, attr, renames[getattr(node, attr)])
        for attr in ('value', 'condition'):
            code = getattr(node, attr, None)
            if isinstance(code, CodeBlock) and code.language == dtypes.Language.Python:
                astutils.ASTFindReplace(dict(renames)).visit(code.code[0])
        if isinstance(node, tn.AssignNode):
            node.edge.replace_dict(renames)
    return clones


def reduce_loop_ranges(stree: tn.ScheduleTreeScope, max_ranges: int = 32, max_enumeration: int = 1 << 20) -> int:
    """
    Split and shrink loops and maps according to the conditionals in their bodies, and hoist invariant guards.

    The body of a loop (or map) is read as a sequence of plain statements and guard groups: an ``if``, an
    ``if/else``, or the sibling pair ``if c`` / ``if not c``. For every group whose condition restricts the iteration
    variable, symbolically (``1 <= i < M``) or through compile-time constant data (``sdfg.constants``, e.g.
    ``cst[i] > 0``), the iteration range is partitioned into runs on which every such condition is decided, and the
    scope is replaced by one copy per run holding the branches that apply:

    * ``for i in range(N): if 1 <= i < M: A`` becomes ``for i in range(1, min(N, M)): A``;
    * ``for k in range(8): S; if cst[k] > 0: A`` with ``cst = [0,0,0,1,1,0,0,2]`` becomes ``for k in range(3): S``,
      ``for k in range(3, 5): S; A``, ``for k in range(5, 7): S``, ``for k in range(7, 8): S; A``.

    A guard that makes up the whole body and does not depend on the iteration (nor on anything the body writes) is
    hoisted above the scope instead (``for i: if cst[k] == 0: A else: B`` becomes ``if cst[k] == 0: for i: A else:
    for i: B``), exposing it to the enclosing scopes. Guards computed by preceding symbol assignments, single-element
    copies or single-assignment tasklets (``t = cst[i]; mask = (t > 0); if mask``) are analyzed through the values
    they compute, and such assignments are dropped once nothing reads them. Atoms that cannot be analyzed remain as
    residual conditions. See :class:`~dace.transformation.passes.loop_range_reduction.LoopRangeReduction` for the
    range analysis itself.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_ranges: Do not split a scope into more than this many copies.
    :param max_enumeration: Upper bound on the iterates evaluated for a guard over compile-time constant data.
    :return: The number of scopes rewritten.
    """
    # Avoid import loops
    from dace.frontend.python import astutils
    from dace.transformation.passes import loop_range_reduction as lrr

    root = stree.get_root()
    constants = {k: v for k, (_, v) in root.constants.items()}
    repository = SimpleNamespace(constants=constants, arrays=root.containers, symbols=root.symbols)

    def expr(code: CodeBlock) -> ast.expr:
        node = code.code[0]
        return astutils.copy_tree(getattr(node, 'value', node))

    def conj(atoms: List[ast.expr]) -> ast.expr:
        return atoms[0] if len(atoms) == 1 else ast.BoolOp(op=ast.And(), values=list(atoms))

    def code_block(node: ast.expr) -> CodeBlock:
        return CodeBlock([ast.fix_missing_locations(ast.Expr(value=astutils.copy_tree(node)))])

    def negated(node: ast.expr) -> ast.expr:
        return astutils.negate_expr(node).value

    def assignment(node: tn.ScheduleTreeNode) -> Optional[Tuple[str, ast.expr]]:
        return _assignment(node, root.containers)

    def temporary(node: tn.ScheduleTreeNode) -> Optional[str]:
        """The target of ``node`` if it assigns a symbol or a transient scalar (a value that may be dropped when
        nothing reads it), else ``None``."""
        assigned = assignment(node)
        if assigned is None:
            return None
        desc = root.containers.get(assigned[0], None)
        if isinstance(node, tn.AssignNode) or (isinstance(desc, data.Scalar) and desc.transient):
            return assigned[0]
        return None

    class Group:
        """A guard group: exclusive branches ``(condition, body)`` (one, or two for if/else) and the original nodes."""

        def __init__(self, nodes: List[tn.ScheduleTreeNode], branches: list):
            self.nodes, self.branches = nodes, branches

    def parse(scope: tn.ScheduleTreeScope) -> Optional[list]:
        """The scope body as a list of plain nodes and ``Group`` objects, or ``None`` if there is no group or an
        unsupported elif chain. Conditions are expressed in the values at the start of the iteration by substituting
        the assignments that precede them (unless something in between rewrites what they depend on)."""
        items, children, k = [], list(scope.children), 0
        while k < len(children):
            node = children[k]
            following = children[k + 1] if k + 1 < len(children) else None
            if isinstance(node, (tn.ElifScope, tn.ElseScope)):
                return None
            if not isinstance(node, tn.IfScope):
                items.append(node)
            else:
                condition = expr(node.condition)
                if isinstance(following, tn.ElseScope):
                    items.append(
                        Group([node, following], [(condition, node.children),
                                                  (negated(condition), following.children)]))
                    k += 1
                elif _complementary(node, following, root.containers):
                    items.append(
                        Group([node, following], [(condition, node.children),
                                                  (expr(following.condition), following.children)]))
                    k += 1
                else:
                    items.append(Group([node], [(condition, node.children)]))
            k += 1
        if not any(isinstance(item, Group) for item in items):
            return None
        for p in reversed(range(len(items))):  # Last assignment first, so chains of assignments compose
            item = items[p]
            assigned = None if isinstance(item, Group) else assignment(item)
            if assigned is None:
                continue
            name, value = assigned
            depends = {name} | set(symbolic.symbols_in_ast(value))
            for later in items[p + 1:]:
                if isinstance(later, Group):
                    later.branches = [(astutils.ASTFindReplace({
                        name: value
                    }).visit(c), body) for c, body in later.branches]
                    written = _in_subtrees(later.nodes, _names_written)
                else:
                    written = _in_subtrees([later], _names_written)
                if written & depends:
                    break
        return items

    def spaces(scope: tn.ScheduleTreeScope) -> Iterable:
        if isinstance(scope, tn.MapScope):
            for dim in range(len(scope.node.map.params)):
                yield dim, lrr.map_iteration_space(scope.node.map, dim, repository)
        else:
            defined = _in_subtrees(scope.children, _names_written)
            yield None, lrr.loop_iteration_space(scope.loop, repository, defined)

    def cells(space, items) -> Optional[List[Tuple[lrr.Interval, Dict[int, tuple]]]]:
        """Partition of the iteration range into ``(interval, {group index: (holds, residual atoms)})`` such that
        every group whose condition restricts the iteration variable is decided on each interval, or ``None``."""
        result = [(space.iteration_range, {})]
        for g, item in enumerate(items):
            if not isinstance(item, Group):
                continue
            ranges = lrr.reduced_ranges(item.branches[0][0], space, None, max_ranges, max_enumeration)
            if ranges is None:
                continue
            ascending = [r.interval for r in (ranges if space.ascending else ranges[::-1])]
            parts = [(iv, (True, r.residual)) for iv, r in zip(ascending, ranges if space.ascending else ranges[::-1])]
            # Between (and around) the ranges the analyzable part of the condition is false.
            edges = [None] + ascending + [None]
            for before, after in zip(edges, edges[1:]):
                lo = None if before is None else before.hi
                hi = None if after is None else after.lo
                if (before is None or lo is not None) and (after is None or hi is not None):
                    parts.append((lrr.Interval(None if lo is None else lo + 1,
                                               None if hi is None else hi - 1), (False, None)))
            result = [(iv, {
                **choice, g: verdict
            }) for cell, choice in result for part, verdict in parts for iv in [lrr.intersect(cell, part)]
                      if not lrr.provably_empty(iv)]
            if len(result) > max_ranges:
                return None
        return result if any(choice for _, choice in result) else None

    def assemble(items, choice: Dict[int, tuple]) -> list:
        """The body of one cell: plain items as they are, decided groups reduced to the branch that holds."""
        body = []
        for g, item in enumerate(items):
            if not isinstance(item, Group):
                body.append(item)
            elif g not in choice:
                body += item.nodes
            else:
                holds, residual = choice[g]
                if holds and residual:
                    body.append(tn.IfScope(condition=code_block(conj(residual)), children=list(item.branches[0][1])))
                    if len(item.branches) > 1:
                        body.append(tn.ElseScope(children=list(item.branches[1][1])))
                elif holds:
                    body += item.branches[0][1]
                elif len(item.branches) > 1:
                    body += item.branches[1][1]
        return body

    def prune(scope, body: list, needed: Set[str]) -> Optional[list]:
        """``body`` without the temporaries nothing later reads (nor anything outside the scope)."""
        kept = []
        for node in reversed(body):
            target = temporary(node)
            if target is not None and target not in needed and not index.read_outside(scope, target):
                continue
            needed = needed | _in_subtrees([node], _names_read)
            kept.insert(0, node)
        return kept

    class Reduce(tn.ScheduleNodeTransformer):
        rewritten = 0

        def visit_scope(self, scope: tn.ScheduleTreeScope):
            self.generic_visit(scope)  # Innermost scopes first
            if not isinstance(scope, (tn.ForScope, tn.MapScope)):
                return scope
            items = parse(scope)
            if items is None:
                return scope
            groups = [g for g, item in enumerate(items) if isinstance(item, Group)]
            for dim, space in spaces(scope):
                if space is None or (isinstance(scope, tn.ForScope) and index.read_outside(scope, space.itervar)):
                    continue  # Dropping iterations (or the loop) would leave a different final value.
                result = self.split(scope, dim, space, items)
                if result is None and len(groups) == 1:
                    result = self.hoist(scope, dim, space, items, groups[0])
                if result is not None:
                    self.rewritten += 1
                    for node in result:  # New scopes and copies (counting originals again errs on the safe side)
                        index.add(node, scope.parent)
                    return result or None  # Nothing left to run
            return scope

        def split(self, scope, dim, space, items) -> Optional[list]:
            partition = cells(space, items)
            if partition is None:
                return None
            bodies = [prune(scope, assemble(items, choice), set()) for _, choice in partition]
            live = [k for k, body in enumerate(bodies) if body]  # A copy without any effect is dropped
            order = _ordered([partition[k][0] for k in live])
            if order is None:
                return None
            if len(order) > 1 and any(isinstance(n, tn.BreakNode) for n in scope.preorder_traversal()):
                return None  # A break would also have to skip the remaining copies
            copies = [None] * len(order)
            for position in reversed(range(len(order))):  # Last to first: the first copy keeps the original nodes
                k = live[order[position]]
                body = bodies[k] if position == 0 else _clone_body(scope, bodies[k], index.used_outside)
                copies[position] = _make_scope(scope, dim, space, partition[k][0], body, position)
            return copies

        def hoist(self, scope, dim, space, items, g) -> Optional[list]:
            """Move the atoms of the (only) guard that do not depend on the iteration out of the scope."""
            varying = _names_written(scope) | {space.itervar} | _in_subtrees(scope.children, _names_written)
            if any(isinstance(n, (tn.ViewNode, tn.RefSetNode)) for n in scope.preorder_traversal()):
                varying |= set(root.containers)  # Aliasing: no data read is known to be invariant
            group = items[g]
            atoms = lrr.guard_atoms(group.branches[0][0])
            invariant = [a for a in atoms if not set(symbolic.symbols_in_ast(a)) & varying]
            variant = [a for a in atoms if a not in invariant]
            if not invariant or (len(group.branches) > 1 and variant):
                return None
            others = [item for item in items if not isinstance(item, Group)]
            needed = _in_subtrees([n for _, body in group.branches for n in body], _names_read)
            others = prune(scope, others, needed | set().union(*(set(symbolic.symbols_in_ast(a)) for a in variant)))
            if any(temporary(n) is None or index.read_outside(scope, temporary(n)) for n in others):
                return None  # Anything else in the body would have to be split off the guard
            first = others + ([tn.IfScope(condition=code_block(conj(variant)), children=list(group.branches[0][1]))]
                              if variant else list(group.branches[0][1]))
            result = [
                tn.IfScope(condition=code_block(conj(invariant)),
                           children=[_make_scope(scope, dim, space, None, first, 0)])
            ]
            if len(group.branches) > 1:
                second = _clone_body(scope, others, index.used_outside) + list(group.branches[1][1])
                result.append(tn.ElseScope(children=[_make_scope(scope, dim, space, None, second, 1)]))
            return result

    total = 0
    while True:  # Repeat so scopes newly exposed (by hoisting, or a map's other parameters) are handled too
        index = _AccessIndex(root)
        visitor = Reduce()
        visitor.visit(stree)
        total += visitor.rewritten
        if visitor.rewritten == 0:
            return total


# ----------------------------------------------------------------------------------------------------------------------
# Range facts, guard folding and index-set splitting
# ----------------------------------------------------------------------------------------------------------------------


def _lrr():
    # Avoid import loops
    from dace.transformation.passes import loop_range_reduction
    return loop_range_reduction


def _repository(root: tn.ScheduleTreeRoot) -> SimpleNamespace:
    """Where the range analysis resolves names: the compile-time constants, containers and symbols of the tree."""
    constants = {k: v for k, (_, v) in root.constants.items()}
    return SimpleNamespace(constants=constants, arrays=root.containers, symbols=root.symbols)


def _iteration_spaces(scope: tn.ScheduleTreeScope, repository) -> List[Tuple[Optional[int], str, object]]:
    """``(map dimension or None, iteration variable, iteration space or None)`` for each variable a loop or map
    binds. The iteration space is ``None`` if it cannot be analyzed (e.g., the body assigns the variable)."""
    lrr = _lrr()
    if isinstance(scope, tn.MapScope):
        written = _in_subtrees(scope.children, _names_written)
        result = []
        for dim, param in enumerate(scope.node.map.params):
            space = lrr.map_iteration_space(scope.node.map, dim, repository)
            if space is not None:  # Symbols the body assigns vary between iterations too
                space = space._replace(body_defined=space.body_defined | written)
            result.append((dim, param, space))
        return result
    if isinstance(scope, tn.ForScope) and scope.loop.loop_variable:
        written = _in_subtrees(scope.children, _names_written)
        return [(None, scope.loop.loop_variable, lrr.loop_iteration_space(scope.loop, repository, written))]
    return []


def _condition(scope: tn.ScheduleTreeScope) -> Optional[ast.expr]:
    """The condition of an ``if``/``elif`` scope as a Python expression, or ``None`` if it has none that can be
    analyzed (state-transition conditions of general blocks are left alone)."""
    if not isinstance(scope, (tn.IfScope, tn.ElifScope)) or isinstance(scope, tn.StateIfScope):
        return None
    code = scope.condition
    if code.language != dtypes.Language.Python or len(code.code) != 1 or not isinstance(code.code[0], ast.Expr):
        return None
    return code.code[0].value


def _boolean_leaves(node: ast.expr) -> int:
    """Number of non-boolean operands in a boolean expression."""
    if isinstance(node, ast.BoolOp):
        return sum(_boolean_leaves(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _boolean_leaves(node.operand)
    return 1


class _RangeFacts:
    """What is known about iteration variables at a point of the tree: for each variable of an enclosing loop or
    map, its iteration space and an interval containing its value (the iteration range, narrowed by the enclosing
    conditions). Valid because an analyzable iteration space guarantees that neither the variable nor the symbols of
    its bounds change within the scope."""

    def __init__(self, repository, max_enumeration: int):
        self.repository = repository
        self.max_enumeration = max_enumeration
        self.known: Dict[str, tuple] = {}

    def _with(self, known: Dict[str, tuple]) -> '_RangeFacts':
        result = copy.copy(self)
        result.known = known
        return result

    def within(self, child: tn.ScheduleTreeScope, previous: Optional[tn.ScheduleTreeNode]) -> '_RangeFacts':
        """The facts inside ``child`` (a scope whose preceding sibling is ``previous``)."""
        lrr = _lrr()
        if isinstance(child, (tn.ForScope, tn.MapScope)):
            known = dict(self.known)
            for _, var, space in _iteration_spaces(child, self.repository):
                if space is None:
                    known.pop(var, None)
                else:
                    known[var] = (space, space.iteration_range)
            return self._with(known)
        if isinstance(child, tn.ElseScope) and isinstance(previous, tn.IfScope):
            condition = _condition(previous)
            clauses = None if condition is None else lrr.disjunctive_normal_form(condition, negate=True)
        else:
            condition = _condition(child) if isinstance(child, tn.IfScope) else None
            clauses = None if condition is None else lrr.disjunctive_normal_form(condition)
        if not clauses or len(clauses) != 1:
            return self
        return self.assuming(clauses[0])

    def assuming(self, atoms: List[ast.expr]) -> '_RangeFacts':
        """The facts where all of ``atoms`` hold (narrowing the known intervals where an atom is a single one)."""
        lrr = _lrr()
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
        lrr = _lrr()
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
        lrr = _lrr()
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


def _fold_chains(children: List[tn.ScheduleTreeNode], facts: _RangeFacts) -> Tuple[List[tn.ScheduleTreeNode], int]:
    """``children`` with every if/elif/else chain among them folded under ``facts`` (including the chains that
    folding exposes), and the number of conditions folded or simplified. Does not descend into other scopes."""
    result, folded, k = [], 0, 0
    while k < len(children):
        node = children[k]
        k += 1
        if _condition(node) is None or not isinstance(node, tn.IfScope):
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
            condition = _condition(branch)
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


def _fold_scope(scope: tn.ScheduleTreeScope, facts: _RangeFacts) -> int:
    """Fold the conditions in the body of ``scope`` (whose own facts are ``facts``), recursively."""
    children, folded = _fold_chains(scope.children, facts)
    scope.children = []
    scope.add_children(children)
    previous = None
    for child in scope.children:
        if isinstance(child, tn.ScheduleTreeScope):
            folded += _fold_scope(child, facts.within(child, previous))
        previous = child
    return folded


def _facts_at(scope: tn.ScheduleTreeScope, max_enumeration: int) -> _RangeFacts:
    """The facts inside ``scope``, from all of its ancestors."""
    path = [scope]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    facts = _RangeFacts(_repository(path[-1]), max_enumeration)
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
    return _fold_scope(stree, _facts_at(stree, max_enumeration))


def _breaks_out_of(scope: tn.ScheduleTreeScope) -> bool:
    """Whether a ``break`` in the body of ``scope`` leaves ``scope`` itself (rather than a nested loop)."""
    for child in scope.children:
        if isinstance(child, tn.BreakNode):
            return True
        if isinstance(child, tn.ScheduleTreeScope) and not isinstance(child, tn.LoopScope) and _breaks_out_of(child):
            return True
    return False


def _guard_atoms_on(scope: tn.ScheduleTreeScope, var: str) -> List[ast.expr]:
    """The distinct atoms that mention ``var`` in the conditions within ``scope`` (outside nested scopes that bind
    ``var`` anew)."""
    lrr = _lrr()
    atoms, seen = [], set()

    def walk(node: tn.ScheduleTreeNode):
        if node is not scope and var in _bound_names(node):
            return
        condition = _condition(node) if isinstance(node, tn.ScheduleTreeScope) else None
        clauses = None if condition is None else lrr.disjunctive_normal_form(condition)
        for atom in (a for clause in clauses or [] for a in clause):
            key = ast.dump(atom)
            if key not in seen and var in symbolic.symbols_in_ast(atom):
                seen.add(key)
                atoms.append(atom)
        for child in getattr(node, 'children', ()):
            walk(child)

    walk(scope)
    return atoms


def _index_set_cells(space, atoms: List[ast.expr], max_ranges: int, max_enumeration: int) -> Optional[list]:
    """The iteration range of ``space`` partitioned into intervals, in iteration order, on each of which every
    analyzable atom of ``atoms`` is decided; adjacent intervals with the same outcomes are merged. ``None`` if the
    partition exceeds ``max_ranges`` intervals or its intervals cannot be ordered."""
    lrr = _lrr()
    cells = [(space.iteration_range, ())]
    for atom in atoms:
        intervals = lrr.atom_intervals(atom, space, max_enumeration)
        if intervals is None:
            continue
        parts = [(iv, True) for iv in intervals]
        # Between (and around) the intervals the atom does not hold.
        edges = [None] + intervals + [None]
        for before, after in zip(edges, edges[1:]):
            if (before is not None and before.hi is None) or (after is not None and after.lo is None):
                continue
            parts.append((lrr.Interval(None if before is None else before.hi + 1,
                                       None if after is None else after.lo - 1), False))
        cells = [(iv, outcomes + (holds, )) for cell, outcomes in cells for part, holds in parts
                 for iv in [lrr.intersect(cell, part)] if not lrr.provably_empty(iv)]
        if len(cells) > 4 * max_ranges:
            return None
    order = _ordered([iv for iv, _ in cells])
    if order is None:
        return None
    merged = []
    for iv, outcomes in (cells[k] for k in order):
        if merged and merged[-1][1] == outcomes:
            merged[-1] = (lrr.Interval(merged[-1][0].lo, iv.hi), outcomes)
        else:
            merged.append((iv, outcomes))
    if len(merged) > max_ranges:
        return None
    return merged if space.ascending else merged[::-1]


def _prune_empty(scope: tn.ScheduleTreeScope, index: _AccessIndex) -> None:
    """Remove the scopes in ``scope`` that have become empty and whose removal has no effect: maps, ``for`` loops whose
    variable is not used elsewhere, and ``if``/``else`` scopes not followed by further branches."""
    kept = []
    for k, child in enumerate(scope.children):
        if isinstance(child, tn.ScheduleTreeScope):
            _prune_empty(child, index)
        following = scope.children[k + 1] if k + 1 < len(scope.children) else None
        if isinstance(child, tn.ScheduleTreeScope) and not child.children and (
                isinstance(child, (tn.MapScope, tn.ElseScope)) or
            (isinstance(child, tn.ForScope) and not index.used_outside(child, child.loop.loop_variable)) or
            (isinstance(child, tn.IfScope) and not isinstance(child, tn.StateIfScope)
             and not isinstance(following, (tn.ElifScope, tn.ElseScope)))):
            continue
        kept.append(child)
    if len(kept) < len(scope.children):
        scope.children = []
        scope.add_children(kept)


def _ancestors(node: tn.ScheduleTreeNode) -> List[tn.ScheduleTreeScope]:
    result = []
    while node.parent is not None:
        node = node.parent
        result.append(node)
    return result


def _trip_count(space) -> Optional[int]:
    """The number of iterations of an iteration space, if it is a known constant."""
    count = _lrr()._num(sympy.floor((space.end - space.start) / space.stride) + 1)
    return None if count is None else max(int(count), 0)


def split_iteration_spaces(stree: tn.ScheduleTreeScope,
                           max_ranges: int = 32,
                           max_enumeration: int = 1 << 20,
                           min_trip_count: int = 8) -> int:
    """
    Split loops and maps at the points where the conditions in their bodies change outcome (index-set splitting),
    and fold those conditions in each part.

    For a loop or map over ``i``, every condition atom within its body (at any depth) that restricts ``i``
    symbolically (``i < M``) or through compile-time constant data (``cst[i] > 0``) partitions the iteration range
    into intervals on which it is decided. The scope is replaced by one copy per interval of the common refinement of
    these partitions, in iteration order, and each copy is folded with :func:`fold_guards`; copies whose body folds
    away entirely are dropped. For example, ``for j in range(-1, 26): for i in range(31): if j < 0: A; if j >= 25: B``
    becomes ``for j in range(-1, 0): for i: A``, ``for j in range(0, 25): for i: pass`` (dropped) and ``for j in
    range(25, 26): for i: B``. Scopes are processed outermost first, so that guards decided by an outer split do not
    split inner scopes.

    A loop is only split if its iteration variable is not used after it and nothing breaks out of it; a map is only
    split along one dimension at a time (the parts are split along the others in turn).

    Splitting every level multiplies the parts: a boundary row ``for j in range(0, 1)`` would carry its own copy of
    every part of the inner ``i`` loop. Code that runs within fewer than ``min_trip_count`` iterations of an enclosing
    loop (or of a map dimension already split) is therefore not split further; its conditions are still folded where
    the enclosing ranges decide them. Iteration spaces of unknown size count as large.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_ranges: Do not split a scope into more than this many copies.
    :param max_enumeration: Upper bound on the iterates evaluated for an atom over compile-time constant data.
    :param min_trip_count: Do not split within loops (or split map dimensions) of fewer iterations than this.
    :return: The number of scopes split.
    """
    lrr = _lrr()
    index = _AccessIndex(stree.get_root())
    splits = 0

    split_on: Dict[int, tuple] = {}  # Variables each part was already split along (a part is never split again

    # along the same variable, even if folding could not prove some outcome of its cell)

    def thin(scope: tn.ScheduleTreeScope, facts: _RangeFacts, variables=None) -> bool:
        """Whether ``scope`` (or, if given, only its ``variables``) iterates fewer than ``min_trip_count`` times."""
        for _, var, space in _iteration_spaces(scope, facts.repository):
            if space is not None and (variables is None or var in variables):
                count = _trip_count(space)
                if count is not None and count < min_trip_count:
                    return True
        return False

    def split(scope: tn.ScheduleTreeScope, facts: _RangeFacts) -> Optional[List[tn.ScheduleTreeNode]]:
        """The parts that replace ``scope`` (a loop or map within the facts ``facts``), or ``None`` to keep it."""
        done = split_on.get(id(scope), (None, set()))[1]
        if done and thin(scope, facts, done):
            return None  # A thin part of a map: its other dimensions are not split
        for dim, var, space in _iteration_spaces(scope, facts.repository):
            if space is None or var in done:
                continue
            if isinstance(scope, tn.ForScope) and (_breaks_out_of(scope) or index.used_outside(scope, var)):
                continue  # Dropping iterations (or the loop) would leave a different final value
            cells = _index_set_cells(space, _guard_atoms_on(scope, var), max_ranges, max_enumeration)
            if cells is None or len(cells) < 2:
                continue
            bodies = [list(scope.children)]
            bodies += [_clone_body(scope, scope.children, index.used_outside) for _ in cells[1:]]
            index.changed()
            parts = []
            for k, ((interval, _), body) in enumerate(zip(cells, bodies)):
                part = _make_scope(scope, dim, space, interval, body, k)
                part.parent = scope.parent  # Until it replaces ``scope`` there
                _fold_scope(part, facts.within(part, None))
                _prune_empty(part, index)
                if part.children:
                    if k > 0:
                        index.add(part)
                    split_on[id(part)] = (part, done | {var})  # Holding the part keeps its id unique
                    parts.append(part)
            return parts
        return None

    def process(node: tn.ScheduleTreeNode, facts: _RangeFacts, previous, cold: bool) -> List[tn.ScheduleTreeNode]:
        """``node`` (inside a scope with the facts ``facts``) with its loops and maps split, as a list of nodes.
        Nothing is split in ``cold`` code (within a thin loop)."""
        nonlocal splits
        if isinstance(node, (tn.ForScope, tn.MapScope)) and not cold:
            parts = split(node, facts)
            if parts is not None:
                splits += 1
                return [n for part in parts for n in process(part, facts, None, cold)]
        if isinstance(node, tn.ScheduleTreeScope):
            inner = facts.within(node, previous)
            cold = cold or (isinstance(node, (tn.ForScope, tn.MapScope)) and thin(node, facts))
            children, node.children, before = node.children, [], None
            for child in children:
                node.add_children(process(child, inner, before, cold))
                before = node.children[-1] if node.children else None
        return [node]

    facts = _facts_at(stree, max_enumeration)
    # The root scope itself is not replaced; its facts are those inside it
    cold = any(isinstance(a, (tn.ForScope, tn.MapScope)) and thin(a, facts) for a in _ancestors(stree))
    children, stree.children, before = stree.children, [], None
    for child in children:
        stree.add_children(process(child, facts, before, cold))
        before = stree.children[-1] if stree.children else None
    return splits


# ----------------------------------------------------------------------------------------------------------------------
# Forward substitution into conditions
# ----------------------------------------------------------------------------------------------------------------------

# Calls a value may contain and still be substituted into a condition (or dropped when unused): conversions and pure
# builtins only
_PURE_CALLS = {'float', 'int', 'bool', 'abs', 'min', 'max', 'round'}


def _pure(value: ast.expr) -> bool:
    return all(
        isinstance(n.func, ast.Name) and n.func.id in _PURE_CALLS for n in ast.walk(value) if isinstance(n, ast.Call))


def _substitutable(target: str, value: ast.expr, node: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> bool:
    """Whether a condition may read ``value`` instead of ``target`` (as assigned by ``node``) with the same result:
    symbols hold values as they are computed, boolean containers hold truth values, and literals are exact in the
    type of their container."""
    if not _pure(value):
        return False
    if isinstance(node, tn.AssignNode):
        return True
    desc = containers.get(target, None)
    if desc is None or desc.total_size != 1:
        return False
    if desc.dtype == dtypes.bool_:
        return True
    literal = value.operand if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub) else value
    if isinstance(literal, ast.Constant) and isinstance(literal.value, (bool, int, float)):
        try:
            return desc.dtype.type(literal.value) == literal.value
        except (TypeError, ValueError, OverflowError):
            return False
    return False


class _ReplaceReads(ast.NodeTransformer):
    """Replaces reads of a single-element container (``t`` or ``t[0]``) with an expression."""

    def __init__(self, target: str, value: ast.expr):
        self.target, self.value, self.replaced = target, value, 0

    def _replacement(self) -> ast.expr:
        from dace.frontend.python import astutils  # Avoid import loops
        self.replaced += 1
        return astutils.copy_tree(self.value)

    def visit_Name(self, node: ast.Name):
        return self._replacement() if node.id == self.target else node

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == self.target:
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if all(isinstance(i, ast.Constant) and i.value == 0 for i in index):
                return self._replacement()
            return node  # Some other element: leave alone
        return self.generic_visit(node)


def forward_substitute_conditions(stree: tn.ScheduleTreeScope) -> int:
    """
    Replace the values a condition reads by the expressions that compute them.

    A condition that reads a symbol or a single-element container (``if mask:``) is rewritten in terms of the value
    assigned to it (``mask = (nord[k] == 0)`` makes it ``if nord[k] == 0:``) when the assignment reaches the
    condition unchanged: it is the last write of that name before the condition, in the same scope or an enclosing
    one, and nothing in between writes the name or anything the value reads. Loops and maps between the assignment
    and the condition count as "in between" in their entirety (their later iterations run before the condition is
    evaluated again), and must not rebind a name the value reads. Assignments are symbol assignments, single-element
    copies, and Python tasklets ``out = <expression>`` over single-element inputs; a container's value is only used if
    the substitution cannot change the outcome (a boolean container, or a literal exact in the container's type).
    The assignments themselves are kept (see :func:`remove_dead_assignments`).

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of substitutions made.
    """
    root = stree.get_root()
    containers = root.containers
    written_in: Dict[int, tuple] = {}  # Subtree writes, by node (the node is held so that ids are not reused)

    def writes(node: tn.ScheduleTreeNode) -> Set[str]:
        entry = written_in.get(id(node))
        if entry is None:
            entry = written_in[id(node)] = (node, _in_subtrees([node], _names_written))
        return entry[1]

    def reaching(branch: tn.ScheduleTreeScope, name: str) -> Optional[ast.expr]:
        """The value assigned to ``name`` that reaches the condition of ``branch``, or ``None``."""
        between: Set[str] = set()
        rebound: Set[str] = set()
        current = branch
        if isinstance(branch, tn.ElifScope):
            # Evaluated where its chain starts: the conditions before it have no effects, and their bodies do not run
            siblings = branch.parent.children
            position = next(k for k, c in enumerate(siblings) if c is branch)
            while not isinstance(siblings[position], tn.IfScope):
                position -= 1
            current = siblings[position]
        while current.parent is not None:
            parent = current.parent
            if isinstance(parent, tn.GBlock):
                return None  # Unstructured control flow: the preceding siblings need not run first
            siblings = parent.children
            position = next(k for k, c in enumerate(siblings) if c is current)
            for sibling in reversed(siblings[:position]):
                assigned = _assignment(sibling, containers)
                if assigned is not None and assigned[0] == name:
                    value = assigned[1]
                    depends = set(symbolic.symbols_in_ast(value))
                    if depends & (between | rebound) or not _substitutable(name, value, sibling, containers):
                        return None
                    return value
                if name in writes(sibling):
                    return None  # Written in a way that is not analyzed
                between |= writes(sibling)
            if isinstance(parent, (tn.LoopScope, tn.MapScope)):
                between |= writes(parent)
                rebound |= _bound_names(parent)
                if name in between:
                    return None
            current = parent
        return None

    substitutions = 0
    for node in list(stree.preorder_traversal()):
        condition = _condition(node) if isinstance(node, tn.ScheduleTreeScope) else None
        if condition is None:
            continue
        changed = True
        while changed:  # Substituted values may read further substitutable names
            changed = False
            for name in sorted(set(symbolic.symbols_in_ast(condition))):
                value = reaching(node, name)
                if value is None:
                    continue
                replacer = _ReplaceReads(name, value)
                condition = replacer.visit(condition)
                if replacer.replaced:
                    substitutions += replacer.replaced
                    changed = True
        node.condition = CodeBlock([ast.fix_missing_locations(ast.Expr(value=condition))])
    return substitutions


def remove_dead_assignments(stree: tn.ScheduleTreeScope) -> int:
    """
    Remove assignments whose result nothing reads: symbol assignments, and single-element copies or pure Python
    tasklets writing a transient scalar (as :func:`forward_substitute_conditions` leaves behind).

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of assignments removed.
    """
    root = stree.get_root()
    containers = root.containers
    in_descriptors = set().union(*(map(str, desc.free_symbols) for desc in containers.values()))
    removed = 0
    while True:
        index = _AccessIndex(root)

        def dead(node: tn.ScheduleTreeNode) -> bool:
            assigned = _assignment(node, containers)
            if assigned is None or index.reads.get(assigned[0], 0) > 0 or not _pure(assigned[1]):
                return False
            if isinstance(node, tn.AssignNode):
                return assigned[0] not in in_descriptors
            desc = containers.get(assigned[0], None)
            return desc is not None and desc.transient and desc.total_size == 1

        count = 0
        for scope in [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]:
            kept = [c for c in scope.children if not dead(c)]
            if len(kept) < len(scope.children):
                count += len(scope.children) - len(kept)
                scope.children = []
                scope.add_children(kept)
        removed += count
        if count == 0:
            return removed


# ----------------------------------------------------------------------------------------------------------------------
# Pairing of complementary guards
# ----------------------------------------------------------------------------------------------------------------------


def _negation_of(condition: ast.expr, other: ast.expr) -> bool:
    """Whether ``other`` is syntactically the negation of ``condition``, up to normalizing both (negations pushed into
    comparisons, disjunctive normal form)."""
    from dace.frontend.python import astutils  # Avoid import loops
    if ast.dump(astutils.negate_expr(condition).value) == ast.dump(other):
        return True
    lrr = _lrr()

    def normalized(clauses) -> Optional[frozenset]:
        return None if clauses is None else frozenset(
            frozenset(ast.dump(atom) for atom in clause) for clause in clauses)

    negated = normalized(lrr.disjunctive_normal_form(condition, negate=True))
    return negated is not None and negated == normalized(lrr.disjunctive_normal_form(other))


def _complementary(first: tn.ScheduleTreeNode, second: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> bool:
    """Whether ``if c: A`` (``first``) directly followed by ``if not c: B`` (``second``) runs exactly one of the two
    bodies, i.e., may become ``if c: A else: B``. The conditions must be pure and negations of each other, and ``A``
    must not change what ``c`` reads: it writes none of its names, creates no aliases (views, references) and does not
    jump away (a ``goto`` could re-enter before ``second``)."""
    if not all(isinstance(n, tn.IfScope) and not isinstance(n, tn.StateIfScope) for n in (first, second)):
        return False
    condition, other = _condition(first), _condition(second)
    if condition is None or other is None or not (_pure(condition) and _pure(other)):
        return False
    if not _negation_of(condition, other):
        return False
    read = set(symbolic.symbols_in_ast(condition))
    written = _in_subtrees([first], _names_written)
    if read & written:
        return False
    for name in read | written:
        if isinstance(containers.get(name, None), (data.View, data.Reference)):
            return False
    for node in first.preorder_traversal():
        if isinstance(node, (tn.ViewNode, tn.RefSetNode, tn.GotoNode)):
            return False
        if isinstance(node, (tn.TaskletNode, tn.LibraryCall)) and getattr(node.node, 'side_effects', False):
            return False  # E.g., a callback that may write anything
    return True


def pair_complementary_guards(stree: tn.ScheduleTreeScope) -> int:
    """
    Turn a guard followed directly by its negation (``if c: A`` then ``if not c: B``) into ``if c: A else: B``.

    The two are only exclusive if running ``A`` cannot change the outcome of ``c``: ``A`` may not write anything
    ``c`` reads, nor create aliases or jump away (see :func:`_complementary`). The conditions are compared after
    normalization, so ``if i < 4`` / ``if i >= 4`` and ``if mask`` / ``if not mask`` are both recognized. The
    resulting ``else`` lets later passes use the condition's negation as a fact without re-deriving it.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of pairs formed.
    """
    containers = stree.get_root().containers
    paired = 0
    for scope in [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]:
        children, result, k, before = scope.children, [], 0, paired
        while k < len(children):
            node = children[k]
            following = children[k + 1] if k + 1 < len(children) else None
            after = children[k + 2] if k + 2 < len(children) else None
            # ``node`` must end its chain (``following`` is an ``if``), and ``following`` must be a chain of its own
            if (following is not None and not isinstance(after, (tn.ElifScope, tn.ElseScope))
                    and _complementary(node, following, containers)):
                result += [node, tn.ElseScope(children=following.children)]
                paired += 1
                k += 2
            else:
                result.append(node)
                k += 1
        if paired > before:
            scope.children = []
            scope.add_children(result)
    return paired


# ----------------------------------------------------------------------------------------------------------------------
# Merging of contiguous loops
# ----------------------------------------------------------------------------------------------------------------------


def _memlet_key(memlet: Memlet, renames: Dict[str, str]) -> tuple:
    return (renames.get(memlet.data,
                        memlet.data), str(memlet.subset), str(memlet.other_subset), str(memlet.wcr), memlet.dynamic)


def _code_key(code: CodeBlock, renames: Dict[str, str]) -> str:
    if not renames or code.language != dtypes.Language.Python:
        return code.as_string
    from dace.frontend.python import astutils  # Avoid import loops
    return '\n'.join(
        ast.unparse(astutils.ASTFindReplace(dict(renames)).visit(astutils.copy_tree(stmt))) for stmt in code.code)


def _equivalent(x: tn.ScheduleTreeNode, y: tn.ScheduleTreeNode, renames: Dict[str, str]) -> bool:
    """Whether ``y`` does exactly what ``x`` does, reading the names of ``y`` through ``renames``. Node kinds whose
    equivalence is not checked here compare unequal."""
    if type(x) is not type(y):
        return False
    if isinstance(x, tn.TaskletNode):
        if (x.node.language != y.node.language or x.node.code.as_string != y.node.code.as_string
                or getattr(x.node, 'side_effects', False) or getattr(y.node, 'side_effects', False)):
            return False
        for attr in ('in_memlets', 'out_memlets'):
            mx, my = getattr(x, attr), getattr(y, attr)
            if not isinstance(mx, dict) or not isinstance(my, dict) or mx.keys() != my.keys():
                return False
            if any(_memlet_key(mx[c], {}) != _memlet_key(my[c], renames) for c in mx):
                return False
    elif isinstance(x, tn.CopyNode):
        if x.target != renames.get(y.target, y.target) or _memlet_key(x.memlet, {}) != _memlet_key(y.memlet, renames):
            return False
    elif isinstance(x, tn.AssignNode):
        if x.name != y.name or x.value.as_string != _code_key(y.value, renames):
            return False
    elif isinstance(x, (tn.IfScope, tn.ElifScope)):
        if x.condition.as_string != _code_key(y.condition, renames):
            return False
    elif isinstance(x, tn.ForScope):
        if (x.loop.loop_variable != y.loop.loop_variable or any(
                getattr(x.loop, a).as_string != getattr(y.loop, a).as_string
                for a in ('init_statement', 'loop_condition', 'update_statement'))):
            return False
    elif isinstance(x, tn.MapScope):
        mx, my = x.node.map, y.node.map
        if mx.params != my.params or str(mx.range) != str(my.range) or mx.schedule != my.schedule:
            return False
    elif not isinstance(x, (tn.ElseScope, tn.BreakNode, tn.ContinueNode)):
        return False
    if isinstance(x, tn.ScheduleTreeScope):
        return len(x.children) == len(y.children) and all(
            _equivalent(a, b, renames) for a, b in zip(x.children, y.children))
    return True


def _local_renames(a: tn.MapScope, b: tn.MapScope, index: _AccessIndex) -> Optional[Dict[str, str]]:
    """The renaming of ``b``'s map-local transients to ``a``'s under which the bodies may be equivalent (as
    :func:`_clone_body` privatizes them in copies of a map), or ``None`` if the names do not correspond."""
    containers = a.get_root().containers

    def names(scope):
        result = []
        for node in scope.preorder_traversal():
            for memlet in (_memlets(node, 'in_memlets') + _memlets(node, 'out_memlets') + _memlets(node, 'memlet')):
                result.append(memlet.data)
            if isinstance(node, tn.CopyNode):
                result.append(node.target)
        return result

    xs, ys = names(a), names(b)
    if len(xs) != len(ys):
        return None
    renames: Dict[str, str] = {}
    for x, y in zip(xs, ys):
        if x == y and y not in renames:
            continue
        if renames.setdefault(y, x) != x:
            return None
    for y, x in renames.items():
        if x in renames or not all(
                getattr(containers.get(n, None), 'transient', False) and not index.used_outside(s, n)
                for n, s in ((x, a), (y, b))):
            return None
    return renames if len(set(renames.values())) == len(renames) else None


def _nonnegative_length(space) -> bool:
    lrr = _lrr()
    return (lrr.provably_ge(space.end +
                            1, space.start) if space.ascending else lrr.provably_ge(space.start + 1, space.end))


def _concatenate(a: tn.ScheduleTreeNode, b: tn.ScheduleTreeNode, repository,
                 index: _AccessIndex) -> Optional[tn.ScheduleTreeScope]:
    """``a`` and ``b`` as one loop or map, if ``b`` continues the iteration range of ``a`` with the same body."""
    lrr = _lrr()
    if isinstance(a, tn.ForScope) and isinstance(b, tn.ForScope):
        (_, var, sa), = _iteration_spaces(a, repository) or [(None, None, None)]
        (_, _, sb), = _iteration_spaces(b, repository) or [(None, None, None)]
        if (sa is None or sb is None or b.loop.loop_variable != var or abs(sa.stride) != 1 or sa.stride != sb.stride
                or sa.op is not sb.op):
            return None
        if lrr._num(sb.start - (sa.end + sa.stride)) != 0 or not (_nonnegative_length(sa) and _nonnegative_length(sb)):
            return None
        # The merged loop evaluates the bound of ``b`` while the body of ``a`` runs
        bound = {str(s) for s in sb.end.free_symbols} if hasattr(sb.end, 'free_symbols') else set()
        if bound & _in_subtrees(a.children, _names_written) or not all(
                _equivalent(x, y, {}) for x, y in zip(a.children, b.children)) or len(a.children) != len(b.children):
            return None
        old = a.loop
        header = LoopRegion(old.label,
                            condition_expr=copy.deepcopy(b.loop.loop_condition),
                            loop_var=var,
                            initialize_expr=copy.deepcopy(old.init_statement),
                            update_expr=copy.deepcopy(old.update_statement),
                            unroll=old.unroll,
                            unroll_factor=old.unroll_factor)
        return tn.ForScope(loop=header, children=a.children)
    if isinstance(a, tn.MapScope) and isinstance(b, tn.MapScope):
        ma, mb = a.node.map, b.node.map
        if ma.params != mb.params or ma.schedule != mb.schedule or len(ma.range) != len(mb.range):
            return None
        differing = [d for d in range(len(ma.range)) if str(ma.range[d]) != str(mb.range[d])]
        if len(differing) != 1:
            return None
        dim = differing[0]
        (ba, ea, sa), (bb, eb, sb) = ma.range[dim], mb.range[dim]
        if sa != 1 or sb != 1 or lrr._num(bb - (ea + 1)) != 0:
            return None
        if not (lrr.provably_ge(ea + 1, ba) and lrr.provably_ge(eb + 1, bb)):
            return None
        renames = {} if all(_equivalent(x, y, {})
                            for x, y in zip(a.children, b.children)) else _local_renames(a, b, index)
        if renames is None or len(a.children) != len(
                b.children) or not all(_equivalent(x, y, renames) for x, y in zip(a.children, b.children)):
            return None
        entry = copy.deepcopy(a.node)
        ranges = [r + (t, ) for r, t in zip(ma.range.ranges, ma.range.tile_sizes)]
        ranges[dim] = (ba, eb, 1, ranges[dim][3])
        entry.map.range = subsets.Range(ranges)
        return tn.MapScope(node=entry, children=a.children, state=a.state)
    return None


def merge_contiguous_loops(stree: tn.ScheduleTreeScope) -> int:
    """
    Merge adjacent loops (or maps) whose iteration ranges are contiguous and whose bodies are identical:
    ``for i in range(a, b): S`` followed by ``for i in range(b, c): S`` becomes ``for i in range(a, c): S``. This
    runs the same iterations in the same order, so it is always valid; it undoes splits that turned out not to
    change the body (e.g., parts on which the outcome of some atom differs but no condition does). Loops need a unit
    stride and ranges of provably non-negative length (so that the union is a range); maps may differ in the names of
    the transients their copies privatized. Inner scopes are merged first, which can make outer bodies identical.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of merges.
    """
    root = stree.get_root()
    repository = _repository(root)
    index = _AccessIndex(root)
    merged = 0

    def visit(scope: tn.ScheduleTreeScope):
        nonlocal merged
        for child in scope.children:
            if isinstance(child, tn.ScheduleTreeScope):
                visit(child)
        result, before = [], merged
        for child in scope.children:
            combined = _concatenate(result[-1], child, repository, index) if result else None
            if combined is not None:
                result[-1] = combined
                merged += 1
            else:
                result.append(child)
        if merged > before:
            scope.children = []
            scope.add_children(result)

    visit(stree)
    return merged
