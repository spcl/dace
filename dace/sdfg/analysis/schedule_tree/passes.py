# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Assortment of passes for schedule trees.
"""

import ast
import copy
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Set, Tuple

from dace import data, dtypes, symbolic
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


def _read_outside(scope: tn.ScheduleTreeScope, name: str) -> bool:
    """Whether ``name`` may be read anywhere outside ``scope`` (loops make lexical and execution order differ, so
    everything outside counts), except inside scopes that rebind it as their own iteration variable."""
    inside = {id(n) for n in scope.preorder_traversal()}
    skip: Set[int] = set()
    for node in scope.get_root().preorder_traversal():
        if id(node) in inside or id(node) in skip:
            continue
        if ((isinstance(node, tn.MapScope) and name in node.node.map.params)
                or (isinstance(node, tn.ForScope) and node.loop.loop_variable == name)):
            skip.update(id(n) for n in node.preorder_traversal())
        elif name in _names_read(node):
            return True
    return False


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

    def access(memlet) -> ast.expr:
        if isinstance(root.containers.get(memlet.data, None), data.Scalar):
            return ast.Name(id=memlet.data, ctx=ast.Load())  # Scalars are read by name in conditions
        return ast.parse(f'{memlet.data}[{memlet.subset}]', mode='eval').body

    def assignment(node: tn.ScheduleTreeNode) -> Optional[Tuple[str, ast.expr]]:
        """A node as ``(target, value)`` if it assigns a value that a guard may be computed from."""
        if isinstance(node, tn.AssignNode):
            return node.name, expr(node.value)
        if isinstance(node, tn.CopyNode) and node.memlet.subset is not None and node.memlet.subset.num_elements() == 1:
            return node.target, access(node.memlet)
        if isinstance(node, tn.TaskletNode) and node.node.language == dtypes.Language.Python:
            code, outputs = node.node.code.code, list(node.out_memlets.items())
            if (len(outputs) == 1 and len(code) == 1 and isinstance(code[0], ast.Assign) and len(code[0].targets) == 1
                    and isinstance(code[0].targets[0], ast.Name) and code[0].targets[0].id == outputs[0][0]
                    and all(m.subset.num_elements() == 1 for _, m in outputs + list(node.in_memlets.items()))):
                value = astutils.copy_tree(code[0].value)
                value = astutils.ASTFindReplace({c: access(m) for c, m in node.in_memlets.items()}).visit(value)
                return outputs[0][1].data, value
        return None

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
                elif (isinstance(following, tn.IfScope)
                      and ast.dump(negated(condition)) == ast.dump(expr(following.condition))):
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
            if target is not None and target not in needed and not _read_outside(scope, target):
                continue
            needed = needed | _in_subtrees([node], _names_read)
            kept.insert(0, node)
        return kept

    def make_scope(scope, dim, space, interval, body: list, k: int):
        """A copy of ``scope`` restricted to ``interval`` (``None`` for unchanged) with the given body."""
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

    def ordered(intervals: List[lrr.Interval]) -> Optional[List[int]]:
        """Indices of ``intervals`` in iteration order, or ``None`` if some pair cannot be ordered."""
        if any(not (lrr.provably_before(a, b) or lrr.provably_before(b, a)) for i, a in enumerate(intervals)
               for b in intervals[i + 1:]):
            return None
        return sorted(range(len(intervals)), key=lambda i: sum(lrr.provably_before(b, intervals[i]) for b in intervals))

    def clone_body(scope, body: list) -> list:
        """Copies of ``body`` that own their scope-local transients (as ``replicate_scope`` does for SDFG scopes),
        so that sibling copies of a map do not share temporaries."""
        clones = [_clone(n) for n in body]
        renames = {}
        for name in _in_subtrees(body, _names_written):
            desc = root.containers.get(name, None)
            if desc is not None and desc.transient and not _read_outside(scope, name):
                renames[name] = data.find_new_name(name, root.containers)
                root.containers[renames[name]] = copy.deepcopy(desc)
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
                if space is None or (isinstance(scope, tn.ForScope) and _read_outside(scope, space.itervar)):
                    continue  # Dropping iterations (or the loop) would leave a different final value.
                result = self.split(scope, dim, space, items)
                if result is None and len(groups) == 1:
                    result = self.hoist(scope, dim, space, items, groups[0])
                if result is not None:
                    self.rewritten += 1
                    return result or None  # Nothing left to run
            return scope

        def split(self, scope, dim, space, items) -> Optional[list]:
            partition = cells(space, items)
            if partition is None:
                return None
            bodies = [prune(scope, assemble(items, choice), set()) for _, choice in partition]
            live = [k for k, body in enumerate(bodies) if body]  # A copy without any effect is dropped
            order = ordered([partition[k][0] for k in live])
            if order is None:
                return None
            if len(order) > 1 and any(isinstance(n, tn.BreakNode) for n in scope.preorder_traversal()):
                return None  # A break would also have to skip the remaining copies
            copies = [None] * len(order)
            for position in reversed(range(len(order))):  # Last to first: the first copy keeps the original nodes
                k = live[order[position]]
                body = bodies[k] if position == 0 else clone_body(scope, bodies[k])
                copies[position] = make_scope(scope, dim, space, partition[k][0], body, position)
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
            if any(temporary(n) is None or _read_outside(scope, temporary(n)) for n in others):
                return None  # Anything else in the body would have to be split off the guard
            first = others + ([tn.IfScope(condition=code_block(conj(variant)), children=list(group.branches[0][1]))]
                              if variant else list(group.branches[0][1]))
            result = [
                tn.IfScope(condition=code_block(conj(invariant)),
                           children=[make_scope(scope, dim, space, None, first, 0)])
            ]
            if len(group.branches) > 1:
                second = clone_body(scope, others) + list(group.branches[1][1])
                result.append(tn.ElseScope(children=[make_scope(scope, dim, space, None, second, 1)]))
            return result

    total = 0
    while True:  # Repeat so scopes newly exposed (by hoisting, or a map's other parameters) are handled too
        visitor = Reduce()
        visitor.visit(stree)
        total += visitor.rewritten
        if visitor.rewritten == 0:
            return total
