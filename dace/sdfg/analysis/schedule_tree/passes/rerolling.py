# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rolling unrolled statements back into loops, and merging the resulting loop nests."""
from typing import Callable, Dict, List, Optional, Tuple

import sympy

from dace import subsets, symbolic
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion
from dace.sdfg.analysis.schedule_tree.passes.common import (clone_subtree, iteration_spaces, names_in_subtrees,
                                                            names_read, names_written, range_analysis, repository_of,
                                                            trip_count)


def roll_key(node: tn.ScheduleTreeNode) -> Optional[Tuple[tuple, Tuple[int, ...], list]]:
    """``(signature, point, memlets)`` of a tasklet whose memlets each access a single element: statements with equal
    signatures differ only in the integer offsets of their indices, which form the point. ``None`` if the node
    cannot be rolled."""
    if not isinstance(node, tn.TaskletNode) or getattr(node.node, 'side_effects', False):
        return None
    if not isinstance(node.in_memlets, dict) or not isinstance(node.out_memlets, dict):
        return None
    memlets = [('in', c, m) for c, m in sorted(node.in_memlets.items())]
    memlets += [('out', c, m) for c, m in sorted(node.out_memlets.items())]
    signature = [node.node.language, node.node.code.as_string]
    point = []
    for direction, connector, memlet in memlets:
        if memlet.subset is None or memlet.other_subset is not None or memlet.subset.num_elements() != 1:
            return None
        parts = []
        for index in memlet.subset.min_element():
            offset, rest = sympy.sympify(index).as_coeff_Add()
            if not offset.is_Integer:
                offset, rest = sympy.Integer(0), sympy.sympify(index)
            parts.append(str(rest))
            point.append(int(offset))
        signature.append((direction, connector, memlet.data, str(memlet.wcr), memlet.dynamic, tuple(parts)))
    return tuple(signature), tuple(point), memlets


def _difference(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _rolled(nodes: List[tn.ScheduleTreeNode], points: List[Tuple[int, ...]], min_statements: int,
            fresh: Callable[[], str]) -> List[tn.ScheduleTreeNode]:
    """``nodes`` (consecutive statements with one signature and the given points) with runs of constant difference
    rolled into loops, and consecutive equal runs stacked into two-dimensional loop nests. Order is preserved."""
    # One-dimensional runs: (first index, step, length)
    runs, k = [], 0
    while k < len(points):
        n, step = 1, None
        while k + n < len(points):
            d = _difference(points[k + n], points[k + n - 1])
            if not any(d) or (step is not None and d != step):
                break
            step, n = d, n + 1
        runs.append((k, step, n))
        k += n
    # Stack consecutive runs of equal step and length whose starts advance by a constant vector
    result, r = [], 0
    while r < len(runs):
        first, step, n = runs[r]
        m, stride = 1, None
        while n > 1 and r + m < len(runs) and runs[r + m][1:] == (step, n):
            d = _difference(points[runs[r + m][0]], points[runs[r + m - 1][0]])
            if not any(d) or (stride is not None and d != stride):
                break
            stride, m = d, m + 1
        if n * m < min_statements:
            result += nodes[first:first + n]
            r += 1
            continue
        result.append(_roll(nodes[first], points[first], [(stride, m)] if m > 1 else [], (step, n), fresh))
        r += m
    return result


def _roll(node: tn.TaskletNode, point: Tuple[int, ...], outer: list, inner: tuple, fresh: Callable[[],
                                                                                                   str]) -> tn.ForScope:
    """A loop nest running ``node`` (at ``point``) over ``outer`` (``[(stride, count)]`` or ``[]``) and ``inner``
    (``(step, count)``) iterations, with its memlet offsets advanced by the strides."""
    dims = outer + [inner]
    variables = [fresh() for _ in dims]
    body = clone_subtree(node)
    offsets = iter(range(len(point)))
    for attr in ('in_memlets', 'out_memlets'):
        for connector in sorted(getattr(body, attr)):
            memlet = getattr(body, attr)[connector]
            indices = []
            for index in memlet.subset.min_element():
                position = next(offsets)
                shift = sum(step[position] * symbolic.symbol(var) for (step, _), var in zip(dims, variables))
                indices.append(sympy.sympify(index) + shift)
            memlet.subset = subsets.Range([(i, i, 1) for i in indices])
    scope: tn.ScheduleTreeNode = body
    for var, (_, count) in reversed(list(zip(variables, dims))):
        header = LoopRegion(f'rolled_{var}', f'{var} < {count}', var, f'{var} = 0', f'{var} = {var} + 1')
        scope = tn.ForScope(loop=header, children=[scope])
    return scope


def reroll_statements(stree: tn.ScheduleTreeScope, min_statements: int = 3) -> int:
    """
    Roll runs of unrolled statements back into loops.

    Consecutive tasklets with the same code, connectors and containers whose single-element accesses differ only in
    integer offsets are treated as points; a run of points with a constant difference becomes a loop, and consecutive
    runs of equal step and length whose starts advance by a constant vector become a two-dimensional loop nest. For
    example, the corner fill ``d[0, 1] = d[4, 0]; d[1, 1] = d[4, 1]; d[2, 1] = d[4, 2]; d[0, 2] = d[3, 0]; ...``
    becomes ``for y in range(2): for x in range(3): d[x, y + 1] = d[4 - y, x]``. The statements run in the same order
    as before, so this is always valid; it shrinks code and exposes loops to the vectorizer.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param min_statements: Do not roll fewer statements than this into a loop.
    :return: The number of loop nests created.
    """
    root = stree.get_root()
    taken = set(root.symbols) | set(root.containers)
    taken |= names_in_subtrees([stree], names_read) | names_in_subtrees([stree], names_written)
    counter = [0]

    def fresh() -> str:
        while f'__roll{counter[0]}' in taken:
            counter[0] += 1
        name = f'__roll{counter[0]}'
        taken.add(name)
        return name

    created = 0
    for scope in [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]:
        children, result, k, changed = scope.children, [], 0, False
        while k < len(children):
            key = roll_key(children[k])
            j = k + 1
            while key is not None and j < len(children):
                other = roll_key(children[j])
                if other is None or other[0] != key[0]:
                    break
                j += 1
            if key is None or j - k < min_statements:
                result += children[k:j]
                k = j
                continue
            segment = children[k:j]
            rolled = _rolled(segment, [roll_key(n)[1] for n in segment], min_statements, fresh)
            created += sum(isinstance(n, tn.ForScope) for n in rolled)
            changed |= len(rolled) < len(segment)
            result += rolled
            k = j
        if changed:
            scope.children = []
            scope.add_children(result)
    return created


def _constant_nest(node: tn.ScheduleTreeNode, repository) -> Optional[Tuple[List[Tuple[str, int]], list]]:
    """``(loops, statements)`` if ``node`` is a perfect nest of loops ``for v in range(0, n)`` with constant ``n``
    around single-element statements (as :func:`reroll_statements` creates), or ``([], [node])`` for a single such
    statement; ``None`` otherwise."""
    loops = []
    while isinstance(node, tn.ForScope) and len(node.children) >= 1:
        spaces = iteration_spaces(node, repository)
        space = spaces[0][2] if spaces else None
        count = None if space is None else trip_count(space)
        if space is None or count is None or space.stride != 1 or range_analysis()._num(space.start) != 0:
            return None
        loops.append((space.itervar, count))
        if len(node.children) == 1 and isinstance(node.children[0], tn.ForScope):
            node = node.children[0]
            continue
        statements = node.children
        break
    else:
        statements = [node]
    if not statements or any(roll_key(s) is None for s in statements):
        return None
    return loops, list(statements)


def _element_accesses(loops: List[Tuple[str, int]], statements: list, limit: int) -> Optional[Tuple[dict, dict]]:
    """``(reads, writes)`` of a constant nest: for each container, the elements accessed, as a map from the symbolic
    part of the indices to the set of integer offsets. ``None`` if the nest has more than ``limit`` iterations or an
    index is not affine in the loop variables with integer coefficients."""
    import itertools
    total = 1
    for _, count in loops:
        total *= count
    if total > limit:
        return None
    variables = [symbolic.symbol(v) for v, _ in loops]
    # Each index as (symbolic part, integer offset, integer coefficient per loop variable)
    patterns = []
    for statement in statements:
        for write, memlets in ((False, statement.in_memlets), (True, statement.out_memlets)):
            for memlet in memlets.values():
                indices = []
                for index in memlet.subset.min_element():
                    expr = sympy.expand(sympy.sympify(index))
                    coefficients = [expr.coeff(v) for v in variables]
                    if not all(c.is_Integer for c in coefficients):
                        return None
                    rest = expr - sum(c * v for c, v in zip(coefficients, variables))
                    if rest.free_symbols & set(variables):
                        return None
                    offset, part = rest.as_coeff_Add()
                    if not offset.is_Integer:
                        offset, part = sympy.Integer(0), rest
                    indices.append((str(part), int(offset), [int(c) for c in coefficients]))
                patterns.append((write, memlet.data, indices))
    reads: Dict[str, Dict[tuple, set]] = {}
    writes: Dict[str, Dict[tuple, set]] = {}
    for write, data, indices in patterns:
        rest = tuple(part for part, _, _ in indices)
        elements = (reads if not write else writes).setdefault(data, {}).setdefault(rest, set())
        for values in itertools.product(*(range(count) for _, count in loops)):
            elements.add(tuple(o + sum(c * x for c, x in zip(cs, values)) for _, o, cs in indices))
    return reads, writes


def _union_accesses(a: dict, b: dict) -> dict:
    result = {data: {rest: set(offsets) for rest, offsets in parts.items()} for data, parts in a.items()}
    for data, parts in b.items():
        for rest, offsets in parts.items():
            result.setdefault(data, {}).setdefault(rest, set()).update(offsets)
    return result


def _may_overlap(a: Dict[str, Dict[tuple, set]], b: Dict[str, Dict[tuple, set]]) -> bool:
    for data in a.keys() & b.keys():
        for rest_a, offsets_a in a[data].items():
            for rest_b, offsets_b in b[data].items():
                if rest_a != rest_b or offsets_a & offsets_b:
                    return True  # Different symbolic parts may still denote the same element
    return False


def fuse_rolled_loops(stree: tn.ScheduleTreeScope, max_iterations: int = 1 << 12) -> int:
    """
    Merge loop nests of equal shape within a block of straight-line code.

    A block is a run of consecutive single-element statements and perfect loop nests over constant ranges starting
    at zero (as :func:`reroll_statements` produces, e.g. one 3x3 nest per field for the corners of a cubed-sphere
    tile). Each nest is merged into the latest earlier nest of the same shape if it is independent of that nest and
    of every item in between: no common element that one of them writes, checked exactly by enumerating the
    iterations (which are few). The merged nest runs the statements of both in their original order in each
    iteration; everything else keeps its order.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_iterations: Do not analyze nests of more iterations than this.
    :return: The number of nests merged into others.
    """
    repository = repository_of(stree.get_root())
    merged = 0

    def conflict(a, b) -> bool:
        return _may_overlap(a[1], b[0]) or _may_overlap(a[0], b[1]) or _may_overlap(a[1], b[1])

    for scope in [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]:
        # Entries: [node or None, loops, statements (for merged nests), accesses or None]
        entries, changed = [], False
        for child in scope.children:
            nest = _constant_nest(child, repository)
            if nest is None:
                entries.append([child, None, None, None])
                continue
            loops, statements = nest
            accesses = _element_accesses(loops, statements, max_iterations)
            target = None
            if loops and accesses is not None:
                shape = tuple(count for _, count in loops)
                for position in reversed(range(len(entries))):
                    entry = entries[position]
                    if entry[3] is None:
                        break  # Nothing moves across an item that is not analyzed
                    if entry[1] and tuple(c for _, c in entry[1]) == shape:
                        if not conflict(entry[3], accesses):
                            target = position
                        break
                    if conflict(entry[3], accesses):
                        break
            if target is None:
                entries.append([child, loops, list(statements), accesses])
                continue
            entry = entries[target]
            renames = {symbolic.symbol(o): symbolic.symbol(l) for (o, _), (l, _) in zip(loops, entry[1]) if o != l}
            for statement in statements:
                if renames:
                    for memlet in list(statement.in_memlets.values()) + list(statement.out_memlets.values()):
                        memlet.subset = subsets.Range([(e, e, 1) for e in (sympy.sympify(i).subs(renames)
                                                                           for i in memlet.subset.min_element())])
                entry[2].append(statement)
            entry[0] = None  # Rebuilt below
            entry[3] = (_union_accesses(entry[3][0], accesses[0]), _union_accesses(entry[3][1], accesses[1]))
            merged += 1
            changed = True
        if not changed:
            continue
        result = []
        for node, loops, statements, _ in entries:
            if node is not None:
                result.append(node)
                continue
            nest = None
            for var, count in reversed(loops):
                header = LoopRegion(f'rolled_{var}', f'{var} < {count}', var, f'{var} = 0', f'{var} = {var} + 1')
                nest = tn.ForScope(loop=header, children=[nest] if nest is not None else statements)
            result.append(nest)
        scope.children = []
        scope.add_children(result)
    return merged
