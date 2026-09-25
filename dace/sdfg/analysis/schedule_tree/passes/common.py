# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers shared by the schedule-tree passes: what nodes read and write, cloning, loop headers."""
import ast
import copy
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import sympy

from dace import data, dtypes
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion


def memlets_of(node: tn.ScheduleTreeNode, attr: str) -> list:
    memlets = getattr(node, attr, None)
    if memlets is None:
        return []
    if isinstance(memlets, Memlet):
        return [memlets]
    return list(memlets.values()) if isinstance(memlets, dict) else list(memlets)


def names_read(node: tn.ScheduleTreeNode) -> Set[str]:
    """Symbol and container names a single tree node reads (not those of its children)."""
    read: Set[str] = set()
    for memlet in memlets_of(node, 'in_memlets') + memlets_of(node, 'memlet'):
        read |= memlet.free_symbols | {memlet.data}
    for memlet in memlets_of(node, 'out_memlets'):
        read |= memlet.free_symbols
    for code in (getattr(node, 'condition', None), getattr(node, 'value', None)):
        if isinstance(code, CodeBlock):
            read |= code.get_free_symbols()
    if isinstance(node, tn.LoopScope):
        read |= {s for code in node.loop.get_meta_codeblocks() for s in code.get_free_symbols()}
    if getattr(node, 'node', None) is not None:  # Tasklets, library nodes, map entries
        read |= set(node.node.free_symbols)
    return read


def names_written(node: tn.ScheduleTreeNode) -> Set[str]:
    """Symbol and container names a single tree node assigns (not those of its children)."""
    written = {memlet.data for memlet in memlets_of(node, 'out_memlets')}
    for attr in ('name', 'target'):
        if isinstance(getattr(node, attr, None), str):
            written.add(getattr(node, attr))
    if isinstance(node, tn.LoopScope) and node.loop.loop_variable:
        written.add(node.loop.loop_variable)
    if isinstance(node, tn.MapScope):
        written |= set(node.node.map.params)
    return written


def names_in_subtrees(nodes: Iterable[tn.ScheduleTreeNode], names_of) -> Set[str]:
    return set().union(*(names_of(n) for node in nodes for n in node.preorder_traversal()))


def clone_subtree(node: tn.ScheduleTreeNode) -> tn.ScheduleTreeNode:
    """A copy of a subtree that owns its SDFG-side nodes and memlets (which the SDFG conversion inserts into the
    graph) but shares the descriptors, states and loop headers it merely references."""
    new = copy.copy(node)
    new.parent = None
    if isinstance(node, tn.ScheduleTreeScope):
        new.children = []
        new.add_children([clone_subtree(c) for c in node.children])
    for attr in ('node', 'in_memlets', 'out_memlets', 'memlet', 'edge', 'value', 'condition'):
        if getattr(new, attr, None) is not None:
            setattr(new, attr, copy.deepcopy(getattr(new, attr)))
    return new


def bound_names(node: tn.ScheduleTreeNode) -> Set[str]:
    """Iteration variables a loop or map binds for its own scope."""
    if isinstance(node, tn.MapScope):
        return set(node.node.map.params)
    if isinstance(node, tn.ForScope) and node.loop.loop_variable:
        return {node.loop.loop_variable}
    return set()


class AccessIndex:
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
            read = names_read(node)
            entry = self._names[id(node)] = (node, read, read | names_written(node))
        return entry[1], entry[2]

    def _count(self, node: tn.ScheduleTreeNode, reads: Dict[str, int], accesses: Dict[str, int],
               bound: Set[str]) -> None:
        bound = bound | bound_names(node)
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
            bound |= bound_names(ancestor)
            ancestor = ancestor.parent
        self._count(node, self.reads, self.accesses, bound)
        self._inside.clear()

    def used_outside(self, scope: tn.ScheduleTreeScope, name: str, reads_only: bool = False) -> bool:
        """Whether ``name`` may be accessed (or only: read) outside ``scope``."""
        ancestor = scope.parent
        while ancestor is not None:
            if name in bound_names(ancestor):
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


def range_analysis():
    # Avoid import loops
    from dace.transformation.passes import loop_range_reduction
    return loop_range_reduction


def repository_of(root: tn.ScheduleTreeRoot) -> SimpleNamespace:
    """Where the range analysis resolves names: the compile-time constants, containers and symbols of the tree."""
    constants = {k: v for k, (_, v) in root.constants.items()}
    return SimpleNamespace(constants=constants, arrays=root.containers, symbols=root.symbols)


def iteration_spaces(scope: tn.ScheduleTreeScope, repository) -> List[Tuple[Optional[int], str, object]]:
    """``(map dimension or None, iteration variable, iteration space or None)`` for each variable a loop or map
    binds. The iteration space is ``None`` if it cannot be analyzed (e.g., the body assigns the variable)."""
    lrr = range_analysis()
    if isinstance(scope, tn.MapScope):
        written = names_in_subtrees(scope.children, names_written)
        result = []
        for dim, param in enumerate(scope.node.map.params):
            space = lrr.map_iteration_space(scope.node.map, dim, repository)
            if space is not None:  # Symbols the body assigns vary between iterations too
                space = space._replace(body_defined=space.body_defined | written)
            result.append((dim, param, space))
        return result
    if isinstance(scope, tn.ForScope) and scope.loop.loop_variable:
        written = names_in_subtrees(scope.children, names_written)
        return [(None, scope.loop.loop_variable, lrr.loop_iteration_space(scope.loop, repository, written))]
    return []


def condition_of(scope: tn.ScheduleTreeScope) -> Optional[ast.expr]:
    """The condition of an ``if``/``elif`` scope as a Python expression, or ``None`` if it has none that can be
    analyzed (state-transition conditions of general blocks are left alone)."""
    if not isinstance(scope, (tn.IfScope, tn.ElifScope)) or isinstance(scope, tn.StateIfScope):
        return None
    code = scope.condition
    if code.language != dtypes.Language.Python or len(code.code) != 1 or not isinstance(code.code[0], ast.Expr):
        return None
    return code.code[0].value


# Calls a value may contain and still be substituted into a condition (or dropped when unused): conversions and pure
# builtins only
_PURE_CALLS = {'float', 'int', 'bool', 'abs', 'min', 'max', 'round'}


def is_pure(value: ast.expr) -> bool:
    return all(
        isinstance(n.func, ast.Name) and n.func.id in _PURE_CALLS for n in ast.walk(value) if isinstance(n, ast.Call))


def make_scope(scope: tn.ScheduleTreeScope, dim: Optional[int], space, interval, body: list,
               k: int) -> tn.ScheduleTreeScope:
    """A copy of the loop or map ``scope`` whose iteration space ``space`` (dimension ``dim`` of a map) is restricted
    to ``interval`` (``None`` for unchanged), with the given body. ``k`` numbers the copies of a loop."""
    lrr = range_analysis()
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


def ordered_intervals(intervals: list) -> Optional[List[int]]:
    """Indices of ``intervals`` in iteration order, or ``None`` if some pair cannot be ordered."""
    lrr = range_analysis()
    if any(not (lrr.provably_before(a, b) or lrr.provably_before(b, a)) for i, a in enumerate(intervals)
           for b in intervals[i + 1:]):
        return None
    return sorted(range(len(intervals)), key=lambda i: sum(lrr.provably_before(b, intervals[i]) for b in intervals))


def clone_body(scope: tn.ScheduleTreeScope, body: list, used_outside: Callable[[tn.ScheduleTreeScope, str],
                                                                               bool]) -> list:
    """Copies of ``body`` (children of ``scope``). The copies of a map body own the transients local to the map (as
    ``replicate_scope`` does for SDFG scopes), so that sibling copies of a map do not share temporaries. Loop bodies
    keep their names: the iterations of a loop run in order, so a value one of them writes may be read by a later one,
    which could be in another copy.

    :param used_outside: Whether a name is used outside a scope; transients that are keep their names.
    """
    root = scope.get_root()
    clones = [clone_subtree(n) for n in body]
    renames = {}
    for name in (names_in_subtrees(body, names_written) if isinstance(scope, tn.MapScope) else ()):
        desc = root.containers.get(name, None)
        if desc is not None and desc.transient and not used_outside(scope, name):
            renames[name] = data.find_new_name(name, root.containers)
            root.containers[renames[name]] = copy.deepcopy(desc)
    if not renames:
        return clones
    from dace.frontend.python import astutils  # Avoid import loops
    for node in (n for clone in clones for n in clone.preorder_traversal()):
        for memlet in memlets_of(node, 'in_memlets') + memlets_of(node, 'out_memlets') + memlets_of(node, 'memlet'):
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


def ancestors(node: tn.ScheduleTreeNode) -> List[tn.ScheduleTreeScope]:
    result = []
    while node.parent is not None:
        node = node.parent
        result.append(node)
    return result


def trip_count(space) -> Optional[int]:
    """The number of iterations of an iteration space, if it is a known constant."""
    count = range_analysis()._num(sympy.floor((space.end - space.start) / space.stride) + 1)
    return None if count is None else max(int(count), 0)


def prune_empty(scope: tn.ScheduleTreeScope, index: AccessIndex) -> None:
    """Remove the scopes in ``scope`` that have become empty and whose removal has no effect: maps, ``for`` loops whose
    variable is not used elsewhere, and ``if``/``else`` scopes not followed by further branches."""
    kept = []
    for k, child in enumerate(scope.children):
        if isinstance(child, tn.ScheduleTreeScope):
            prune_empty(child, index)
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
