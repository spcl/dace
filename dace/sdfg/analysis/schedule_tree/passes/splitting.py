# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Index-set splitting of loops and maps by the conditions in their bodies, and merging contiguous loops back."""
import ast
import copy
from typing import Dict, List, Optional

from dace import dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, ancestors, bound_names, clone_body,
                                                            condition_of, iteration_spaces, make_scope, memlets_of,
                                                            names_in_subtrees, names_written, ordered_intervals,
                                                            prune_empty, range_analysis, repository_of, trip_count)
from dace.sdfg.analysis.schedule_tree.passes.folding import (RangeFacts, facts_at, fold_scope)


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
    lrr = range_analysis()
    atoms, seen = [], set()

    def walk(node: tn.ScheduleTreeNode):
        if node is not scope and var in bound_names(node):
            return
        condition = condition_of(node) if isinstance(node, tn.ScheduleTreeScope) else None
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
    lrr = range_analysis()
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
    order = ordered_intervals([iv for iv, _ in cells])
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


def split_iteration_spaces(stree: tn.ScheduleTreeScope,
                           max_ranges: int = 32,
                           max_enumeration: int = 1 << 20,
                           min_trip_count: int = 1) -> int:
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

    Splitting every level multiplies the parts: a boundary row ``for j in range(0, 1)`` carries its own copy of every
    part of the inner ``i`` loop. By default everything is split (full specialization, the fastest code measured).
    To trade some speed for code size, code that runs within fewer than ``min_trip_count`` iterations of an enclosing
    loop (or of a map dimension already split) is not split further; its conditions are still folded where the
    enclosing ranges decide them, and the rest remain as runtime guards. Iteration spaces of unknown size count as
    large.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_ranges: Do not split a scope into more than this many copies.
    :param max_enumeration: Upper bound on the iterates evaluated for an atom over compile-time constant data.
    :param min_trip_count: Do not split within loops (or split map dimensions) of fewer iterations than this; ``1``
                           splits everything, e.g. ``4`` roughly halves the code of stencils with boundary cases.
    :return: The number of scopes split.
    """
    index = AccessIndex(stree.get_root())
    splits = 0

    split_on: Dict[int, tuple] = {}  # Variables each part was already split along (a part is never split again

    # along the same variable, even if folding could not prove some outcome of its cell)

    def thin(scope: tn.ScheduleTreeScope, facts: RangeFacts, variables=None) -> bool:
        """Whether ``scope`` (or, if given, only its ``variables``) iterates fewer than ``min_trip_count`` times."""
        for _, var, space in iteration_spaces(scope, facts.repository):
            if space is not None and (variables is None or var in variables):
                count = trip_count(space)
                if count is not None and count < min_trip_count:
                    return True
        return False

    def split(scope: tn.ScheduleTreeScope, facts: RangeFacts) -> Optional[List[tn.ScheduleTreeNode]]:
        """The parts that replace ``scope`` (a loop or map within the facts ``facts``), or ``None`` to keep it."""
        done = split_on.get(id(scope), (None, set()))[1]
        if done and thin(scope, facts, done):
            return None  # A thin part of a map: its other dimensions are not split
        for dim, var, space in iteration_spaces(scope, facts.repository):
            if space is None or var in done:
                continue
            if isinstance(scope, tn.ForScope) and (_breaks_out_of(scope) or index.used_outside(scope, var)):
                continue  # Dropping iterations (or the loop) would leave a different final value
            cells = _index_set_cells(space, _guard_atoms_on(scope, var), max_ranges, max_enumeration)
            if cells is None or len(cells) < 2:
                continue
            bodies = [list(scope.children)]
            bodies += [clone_body(scope, scope.children, index.used_outside) for _ in cells[1:]]
            index.changed()
            parts = []
            for k, ((interval, _), body) in enumerate(zip(cells, bodies)):
                part = make_scope(scope, dim, space, interval, body, k)
                part.parent = scope.parent  # Until it replaces ``scope`` there
                fold_scope(part, facts.within(part, None))
                prune_empty(part, index)
                if part.children:
                    if k > 0:
                        index.add(part)
                    split_on[id(part)] = (part, done | {var})  # Holding the part keeps its id unique
                    parts.append(part)
            return parts
        return None

    def process(node: tn.ScheduleTreeNode, facts: RangeFacts, previous, cold: bool) -> List[tn.ScheduleTreeNode]:
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

    facts = facts_at(stree, max_enumeration)
    # The root scope itself is not replaced; its facts are those inside it
    cold = any(isinstance(a, (tn.ForScope, tn.MapScope)) and thin(a, facts) for a in ancestors(stree))
    children, stree.children, before = stree.children, [], None
    for child in children:
        stree.add_children(process(child, facts, before, cold))
        before = stree.children[-1] if stree.children else None
    return splits


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


def _local_renames(a: tn.MapScope, b: tn.MapScope, index: AccessIndex) -> Optional[Dict[str, str]]:
    """The renaming of ``b``'s map-local transients to ``a``'s under which the bodies may be equivalent (as
    :func:`clone_body` privatizes them in copies of a map), or ``None`` if the names do not correspond."""
    containers = a.get_root().containers

    def names(scope):
        result = []
        for node in scope.preorder_traversal():
            for memlet in (memlets_of(node, 'in_memlets') + memlets_of(node, 'out_memlets') +
                           memlets_of(node, 'memlet')):
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
    lrr = range_analysis()
    return (lrr.provably_ge(space.end +
                            1, space.start) if space.ascending else lrr.provably_ge(space.start + 1, space.end))


def _concatenate(a: tn.ScheduleTreeNode, b: tn.ScheduleTreeNode, repository,
                 index: AccessIndex) -> Optional[tn.ScheduleTreeScope]:
    """``a`` and ``b`` as one loop or map, if ``b`` continues the iteration range of ``a`` with the same body."""
    lrr = range_analysis()
    if isinstance(a, tn.ForScope) and isinstance(b, tn.ForScope):
        (_, var, sa), = iteration_spaces(a, repository) or [(None, None, None)]
        (_, _, sb), = iteration_spaces(b, repository) or [(None, None, None)]
        if (sa is None or sb is None or b.loop.loop_variable != var or abs(sa.stride) != 1 or sa.stride != sb.stride
                or sa.op is not sb.op):
            return None
        if lrr._num(sb.start - (sa.end + sa.stride)) != 0 or not (_nonnegative_length(sa) and _nonnegative_length(sb)):
            return None
        # The merged loop evaluates the bound of ``b`` while the body of ``a`` runs
        bound = {str(s) for s in sb.end.free_symbols} if hasattr(sb.end, 'free_symbols') else set()
        if bound & names_in_subtrees(a.children, names_written) or not all(
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
    repository = repository_of(root)
    index = AccessIndex(root)
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
