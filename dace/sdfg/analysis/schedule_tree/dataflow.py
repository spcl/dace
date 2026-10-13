# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Dataflow between the statements of a schedule tree: which writes certainly happen whenever a subtree runs, and whether
a value one statement writes may be the one another statement reads.

``ScheduleTreeNode.input_memlets`` drops reads covered by earlier writes of the same scope, counting every write,
including conditional ones (in ``if`` scopes, or in loops that may not run). That answers which data a subtree
touches, but not which reads may see values written before a subtree: for that, only writes that certainly happen
(*must-writes*) may hide earlier values. This module computes must-writes and answers reachability queries with them.

Containers are identified by name: accesses through views or of containers that may alias are the caller's concern.
"""

import sympy

from dace import symbolic
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn

# A region of a container: ``(first, last)`` per dimension, inclusive, or ``None`` for a dimension whose extent is not
# known (only in regions of accesses that may happen, never in those of must-writes)
Box = list[tuple[sympy.Expr, sympy.Expr] | None]

# An iteration range: ``(first, last, stride)``, inclusive, in iteration order
Space = tuple[sympy.Expr, sympy.Expr, int]

# Nodes that end an iteration or a loop early
_EARLY_EXITS = (tn.BreakNode, tn.ContinueNode)


def _expr(value) -> sympy.Expr:
    return sympy.sympify(symbolic.pystr_to_symbolic(value) if isinstance(value, str) else value)


def _number(value: sympy.Expr) -> int | None:
    value = sympy.sympify(value)
    return int(value) if value.is_Integer else None


def _symbol(expr: sympy.Expr, name: str) -> sympy.Symbol | None:
    """The symbol called ``name`` in ``expr``, if any."""
    return next((s for s in expr.free_symbols if str(s) == name), None)


def _slope(expr: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr | None:
    """The coefficient of ``variable`` in ``expr`` if ``expr`` is affine in it with a constant coefficient, else
    ``None``."""
    if variable not in expr.free_symbols:
        return sympy.Integer(0)
    if not expr.is_polynomial(variable):
        return None
    slope = sympy.diff(expr, variable)
    return slope if slope.is_number else None


def _bound(expr: sympy.Expr, ranges: dict[str, tuple[sympy.Expr, sympy.Expr] | None], upper: bool) -> sympy.Expr | None:
    """The lowest (or highest) value of ``expr`` over the variables of ``ranges`` (by name; ``None`` for a variable
    without known range), or ``None`` if it cannot be bounded."""
    expr = sympy.expand(expr)
    for variable in [s for s in expr.free_symbols if str(s) in ranges]:
        bounds = ranges[str(variable)]
        slope = _slope(expr, variable)
        if bounds is None or slope is None:
            return None
        expr = sympy.expand(expr.subs(variable, bounds[1] if (slope > 0) == upper else bounds[0]))
    return expr


def disjoint(a: Box, b: Box) -> bool:
    """Whether two regions provably share no element."""
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        for low, high in ((x[0], y[1]), (y[0], x[1])):
            difference = sympy.expand(high - low)
            if difference.is_number and difference < 0:
                return True
    return False


def _subtract(box: list[tuple[int, int]], other: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """``box`` minus ``other`` (integer boxes), as a list of disjoint boxes."""
    if any(d < a or c > b for (a, b), (c, d) in zip(box, other)):
        return [box]
    result, rest = [], list(box)
    for dim, ((a, b), (c, d)) in enumerate(zip(box, other)):
        if a < c:
            result.append(rest[:dim] + [(a, c - 1)] + rest[dim + 1 :])
        if d < b:
            result.append(rest[:dim] + [(d + 1, b)] + rest[dim + 1 :])
        rest[dim] = (max(a, c), min(b, d))
    return result


def covered(region: Box, writes: list[Box]) -> bool:
    """
    Whether the union of ``writes`` provably contains ``region``. Bounds may be symbolic: in a dimension whose extent
    in ``region`` is a constant, a write takes part if its bounds differ from the region's by constants; in other
    dimensions, if its bounds are the region's.

    :param region: The region to cover.
    :param writes: The regions written.
    :return: True if every element of ``region`` is in some region of ``writes``.
    """
    if any(d is None for d in region):
        return False
    extents = [_number(sympy.expand(high - low)) for low, high in region]
    if any(e is not None and e < 0 for e in extents):
        return True  # Empty
    fixed = [d for d, e in enumerate(extents) if e is not None]
    remaining = [[(0, extents[d]) for d in fixed]]  # Offsets relative to the region's lower bounds
    for write in writes:
        if len(write) != len(region) or any(d is None for d in write):
            continue
        if any(
            sympy.expand(write[d][0] - region[d][0]) != 0 or sympy.expand(write[d][1] - region[d][1]) != 0
            for d, e in enumerate(extents)
            if e is None
        ):
            continue
        offsets = [
            (_number(sympy.expand(write[d][0] - region[d][0])), _number(sympy.expand(write[d][1] - region[d][0])))
            for d in fixed
        ]
        if any(lo is None or hi is None for lo, hi in offsets):
            continue
        remaining = [piece for box in remaining for piece in _subtract(box, offsets)]
        if not remaining:
            return True
    return not remaining


def _index(scope: tn.ScheduleTreeScope, child: tn.ScheduleTreeNode) -> int:
    return next(k for k, c in enumerate(scope.children) if c is child)


def _path(node: tn.ScheduleTreeNode) -> list[tn.ScheduleTreeNode]:
    """``node`` and its ancestors, outermost first."""
    path = [node]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path[::-1]


def _assigned(scope: tn.ScheduleTreeScope) -> set:
    """Names that statements in a scope may assign (symbols, containers, and the variables of inner loops)."""
    names = set()
    for node in scope.preorder_traversal():
        if node is scope:
            continue
        if isinstance(node, tn.AssignNode):
            names.add(node.name)
        elif isinstance(node, (tn.TaskletNode, tn.LibraryCall)) and isinstance(node.out_memlets, dict):
            names |= {m.data for m in node.out_memlets.values()}
        elif isinstance(node, (tn.CopyNode, tn.DynScopeCopyNode)):
            names.add(node.target)
        elif isinstance(node, tn.ForScope) and node.loop.loop_variable:
            names.add(node.loop.loop_variable)
        elif isinstance(node, tn.MapScope):
            names |= set(node.node.map.params)
    return names


class ReachingAnalysis:
    """
    Must-writes of subtrees and reachability of values between the statements of a schedule tree. Results are cached
    and describe the tree when the analysis was created: create a new analysis after changing the tree.

    :param root: The schedule tree.
    """

    def __init__(self, root: tn.ScheduleTreeRoot):
        self.root = root
        self._ranges: dict[int, dict[str, Space | None]] = {}
        self._must: dict[int, dict[str, list[Box]]] = {}
        self._exits: dict[int, bool] = {}
        # Jumps between arbitrary places may skip any write: then no write is certain
        self._gotos = any(isinstance(n, (tn.GotoNode, tn.StateLabel)) for n in root.preorder_traversal())

    # ------------------------------------------------------------------------------------------------------------------
    # Iteration ranges

    def ranges(self, scope: tn.ScheduleTreeNode) -> dict[str, Space | None]:
        """
        The ``(first, last, stride)`` of each variable a ``for`` loop or map binds (inclusive, in iteration order), or
        ``None`` for a variable whose range cannot be analyzed (e.g., the body assigns it or its bounds). Empty for
        other nodes.

        :param scope: The node.
        :return: A dictionary from variable names to ranges.
        """
        key = id(scope)
        if key in self._ranges:
            return self._ranges[key]
        result = {}
        spaces = []
        if isinstance(scope, tn.ForScope) and scope.loop.loop_variable:
            spaces = [(scope.loop.loop_variable, tn.loop_range(scope.loop))]
        elif isinstance(scope, tn.MapScope):
            spaces = list(zip(scope.node.map.params, scope.node.map.range.ndrange()))
        if spaces:
            assigned = _assigned(scope)
            for variable, space in spaces:
                if space is None:
                    result[variable] = None
                    continue
                first, last, stride = (_expr(v) for v in space)
                names = {str(s) for s in first.free_symbols | last.free_symbols}
                stride = _number(stride)
                unknown = stride in (None, 0) or (names | {variable}) & assigned
                result[variable] = None if unknown else (first, last, stride)
        self._ranges[key] = result
        return result

    def _extent(self, scope: tn.ScheduleTreeNode) -> dict[str, tuple[sympy.Expr, sympy.Expr] | None]:
        """The lowest and highest value of each variable of a loop or map (``None`` if unknown)."""
        result = {}
        for variable, space in self.ranges(scope).items():
            if space is None:
                result[variable] = None
            else:
                first, last, stride = space
                result[variable] = (first, last) if stride > 0 else (last, first)
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Regions of accesses

    def region(self, node: tn.ScheduleTreeNode, memlet: Memlet, context: tn.ScheduleTreeNode | None) -> Box:
        """
        The region an access may touch over the iterations of the loops and maps between it and ``context`` (the
        variables of ``context`` and of the scopes around it stay symbolic).

        :param node: The node that accesses the container (if it is a loop, its own variables are not bounded).
        :param memlet: The access.
        :param context: An ancestor of ``node``, or ``None`` to bound the variables of all loops around it.
        :return: The region, with ``None`` for dimensions that cannot be bounded.
        """
        ranges: dict[str, tuple[sympy.Expr, sympy.Expr] | None] = {}
        scope = node.parent
        while scope is not None and scope is not context:
            for variable, extent in self._extent(scope).items():
                ranges.setdefault(variable, extent)  # Inner loops shadow outer ones of the same name
            scope = scope.parent
        box: Box = []
        for start, end, _ in memlet.subset.ndrange():
            low, high = _bound(_expr(start), ranges, False), _bound(_expr(end), ranges, True)
            box.append(None if low is None or high is None else (low, high))
        return box

    # ------------------------------------------------------------------------------------------------------------------
    # Must-writes

    def _exits_early(self, node: tn.ScheduleTreeNode) -> bool:
        key = id(node)
        if key not in self._exits:
            self._exits[key] = any(isinstance(n, _EARLY_EXITS) for n in node.preorder_traversal())
        return self._exits[key]

    def must_writes(self, node: tn.ScheduleTreeNode) -> dict[str, list[Box]]:
        """
        The regions that a node certainly writes whenever it runs, relative to one execution of its parent (the
        variables of the scopes around the node stay symbolic).

        Statements write their outputs (except dynamic and write-conflict-resolved ones, and strided ranges); ``for``
        loops and maps write the union over their iterations of what their bodies certainly write, if they run at least
        once and each region is indexed by each variable with unit stride and coefficient one (or minus one) in at most
        one dimension; an ``if`` followed by an ``else`` writes what both branches write. Nothing else is certain:
        ``if`` scopes alone, ``while`` loops, other scopes, subtrees that may exit early (``break``, ``continue``),
        and anything in trees with ``goto``.

        :param node: The node.
        :return: A dictionary from container names to the regions written.
        """
        key = id(node)
        if key in self._must:
            return self._must[key]
        result: dict[str, list[Box]] = {}
        if self._gotos:
            pass
        elif isinstance(node, tn.TaskletNode):
            for memlet in node.out_memlets.values():
                if memlet.wcr is not None or memlet.dynamic:
                    continue
                box = []
                for start, end, stride in memlet.subset.ndrange():
                    start, end = _expr(start), _expr(end)
                    if _number(sympy.expand(end - start)) != 0 and _number(_expr(stride)) != 1:
                        box = None  # Strided: not every element in the range
                        break
                    box.append((start, end))
                if box is not None:
                    result.setdefault(memlet.data, []).append(box)
        elif isinstance(node, (tn.ForScope, tn.MapScope)) and not self._exits_early(node):
            result = self.sequence_writes(node.children)
            for variable, space in self.ranges(node).items():
                result = self._over_iterations(result, variable, space)
        self._must[key] = result
        return result

    def sequence_writes(self, children: list[tn.ScheduleTreeNode]) -> dict[str, list[Box]]:
        """
        What a sequence of siblings certainly writes when it runs from its first to its last statement, relative to
        one execution of their parent (nothing if one of them may exit early).

        :param children: The siblings, in order.
        :return: A dictionary from container names to the regions written.
        """
        if self._gotos or any(self._exits_early(c) for c in children):
            return {}
        result: dict[str, list[Box]] = {}
        for k, child in enumerate(children):
            if type(child) is tn.IfScope:
                following = children[k + 1] if k + 1 < len(children) else None
                if type(following) is not tn.ElseScope:
                    continue
                then, otherwise = self.sequence_writes(child.children), self.sequence_writes(following.children)
                writes = {
                    name: [b for b in boxes if covered(b, otherwise.get(name, []))] for name, boxes in then.items()
                }
            elif isinstance(child, tn.ScheduleTreeScope) and not isinstance(child, (tn.ForScope, tn.MapScope)):
                continue
            else:
                writes = self.must_writes(child)
            for name, boxes in writes.items():
                if boxes:
                    result.setdefault(name, []).extend(boxes)
        return result

    @staticmethod
    def _over_iterations(writes: dict[str, list[Box]], variable: str, space: Space | None) -> dict[str, list[Box]]:
        """The union over the iterations of one variable of the regions written in each (exact, or dropped)."""
        if space is None:
            return {}  # May run no iteration
        first, last, stride = space
        trips = sympy.expand((last - first) * (1 if stride > 0 else -1))
        if abs(stride) != 1 or not ((trips.is_number and trips >= 0) or trips.is_nonnegative):
            return {}  # Strided, or may run no iteration
        low, high = (first, last) if stride > 0 else (last, first)
        result: dict[str, list[Box]] = {}
        for name, boxes in writes.items():
            for box in boxes:
                dims = [d for d, (lo, hi) in enumerate(box) if _symbol(lo, variable) or _symbol(hi, variable)]
                if not dims:
                    result.setdefault(name, []).append(box)
                    continue
                if len(dims) > 1:
                    continue
                lo, hi = box[dims[0]]
                symbol = _symbol(lo, variable) or _symbol(hi, variable)
                slopes = (_slope(lo, symbol), _slope(hi, symbol))
                if slopes == (1, 1):
                    span = (lo.subs(symbol, low), hi.subs(symbol, high))
                elif slopes == (-1, -1):
                    span = (lo.subs(symbol, high), hi.subs(symbol, low))
                else:
                    continue
                result.setdefault(name, []).append(
                    box[: dims[0]] + [tuple(map(sympy.expand, span))] + box[dims[0] + 1 :]
                )
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Reachability

    def _covered_inside(self, node: tn.ScheduleTreeNode, memlet: Memlet, top: tn.ScheduleTreeNode) -> bool:
        """Whether, in some scope between an access and ``top`` (an ancestor or the node itself), the siblings before
        the access certainly write what it reads."""
        current = node
        while current is not top:
            scope = current.parent
            before = scope.children[: _index(scope, current)]
            if before and covered(self.region(node, memlet, scope), self.sequence_writes(before).get(memlet.data, [])):
                return True
            current = scope
        return False

    def may_reach(
        self, write: tn.ScheduleTreeNode, write_memlet: Memlet, read: tn.ScheduleTreeNode, read_memlet: Memlet
    ) -> bool:
        """
        Whether a value that ``write`` stores through ``write_memlet`` may be read by ``read`` through ``read_memlet``.

        It cannot if the two accesses touch different elements, if everything read is certainly written in between,
        or if the read always comes first. With ``C`` the lowest common ancestor of the two nodes and ``A`` and ``B``
        its children that contain the write and the read, what is written in between is:

        * what the siblings before the read write, in some scope between the read and ``B``;
        * what the children of ``C`` between ``A`` and ``B`` write, if ``A`` comes first;
        * otherwise, what the children of ``C`` before ``B`` write (the value comes from an earlier iteration of a
          loop or map around ``C``; if there is none, the read always comes first, as when ``A`` is ``B``: a statement
          reads before it writes, and a scope reads its condition before its body runs).

        :param write: The node writing.
        :param write_memlet: The access of ``write``.
        :param read: The node reading (a statement, or e.g. an ``if`` scope reading in its condition).
        :param read_memlet: The access of ``read``.
        :return: False if the value cannot be read there, True if it may.
        """
        if write_memlet.data != read_memlet.data:
            return False
        if disjoint(self.region(write, write_memlet, None), self.region(read, read_memlet, None)):
            return False
        write_path, read_path = _path(write), _path(read)
        depth = 0  # Length of the common prefix
        while depth < min(len(write_path), len(read_path)) and write_path[depth] is read_path[depth]:
            depth += 1
        if depth == min(len(write_path), len(read_path)):  # One contains the other, or they are the same
            first = second = write_path[depth - 1]
            common = first.parent
            forward = False
        else:
            common = write_path[depth - 1]
            first, second = write_path[depth], read_path[depth]
            forward = _index(common, first) < _index(common, second)
        if common is None:
            return False  # The root reads before it writes
        if self._covered_inside(read, read_memlet, second):
            return False
        position_first, position_second = _index(common, first), _index(common, second)
        if forward:
            between = common.children[position_first + 1 : position_second]
        else:
            # The value can only come from an earlier iteration of a loop or map around both. Only the siblings before
            # the read run in the same iteration (the variables of the loops around them have the read's values)
            if not any(isinstance(n, (tn.LoopScope, tn.MapScope)) for n in _path(common)):
                return False
            between = common.children[:position_second]
        if not between:
            return True
        written = self.sequence_writes(between).get(read_memlet.data, [])
        return not covered(self.region(read, read_memlet, common), written)

    def may_reach_later(
        self,
        write: tn.ScheduleTreeNode,
        write_memlet: Memlet,
        read: tn.ScheduleTreeNode,
        read_memlet: Memlet,
        scope: tn.ScheduleTreeScope,
    ) -> bool:
        """
        Whether a value that ``write`` stores may be read by ``read`` in a later execution of ``scope`` (a scope that
        contains both, e.g., the next iteration of a loop, or of a loop around ``scope``) than the one that wrote it.

        A later execution runs the children of ``scope`` from the first one: the value cannot be read there if the
        children before the one that contains the read certainly write what it reads, or if siblings before the read
        do within that child (see :meth:`may_reach`).

        :param write: The node writing.
        :param write_memlet: The access of ``write``.
        :param read: The node reading.
        :param read_memlet: The access of ``read``.
        :param scope: A scope that contains both nodes.
        :return: False if the value cannot be read in a later execution of ``scope``, True if it may.
        """
        if write_memlet.data != read_memlet.data:
            return False
        if not any(isinstance(n, (tn.LoopScope, tn.MapScope)) for n in _path(scope)):
            return False  # Runs once
        if disjoint(self.region(write, write_memlet, None), self.region(read, read_memlet, None)):
            return False
        read_path = _path(read)
        child = read_path[next(k for k, n in enumerate(read_path) if n is scope) + 1]
        if self._covered_inside(read, read_memlet, child):
            return False
        before = scope.children[: _index(scope, child)]
        written = self.sequence_writes(before).get(read_memlet.data, [])
        return not covered(self.region(read, read_memlet, scope), written)
