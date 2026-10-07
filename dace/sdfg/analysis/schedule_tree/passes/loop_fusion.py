# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fusing coarse sibling loops so that the data they share is reused while it is still in cache."""
import ast
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import sympy

from dace import data, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, iteration_spaces, memlets_of, names_read,
                                                            names_written, repository_of, trip_count)

# Names under which containers that may share memory are analyzed together
_MAY_ALIAS = '__may_alias__'


@dataclass
class _Access:
    """One access of a container (or of the memory it aliases) within a loop, relative to the loop variable ``k``.

    ``kind`` is ``'affine'`` (dimension ``dim`` is indexed with ``k + offset``), ``'fixed'`` (no index depends on
    ``k``; ``region`` holds the constant ``(first, last)`` indices per dimension, or ``None`` where not constant) or
    ``'unknown'``."""
    key: str
    write: bool
    kind: str
    dim: int = -1
    offset: int = 0
    region: Optional[List[Optional[Tuple[int, int]]]] = None
    memlet: Optional[Memlet] = None
    node: Optional[tn.ScheduleTreeNode] = None
    element: Optional[str] = None  # The subset accessed, for fixed single-element accesses


@dataclass
class _Box:
    """The elements of a container accessed in one iteration of a loop: a bounding box of the dimensions not indexed
    by the loop variable (``None`` for a whole dimension), and the offsets of the loop variable in the one that is."""
    shape: List[int]
    element_bytes: int
    dims: List[Optional[Tuple[int, int]]]
    offsets: Set[int]
    kdim: int = -1

    def union(self, other: '_Box') -> '_Box':
        dims = [
            None if a is None or b is None else (min(a[0], b[0]), max(a[1], b[1]))
            for a, b in zip(self.dims, other.dims)
        ]
        kdim = self.kdim if self.kdim >= 0 else other.kdim
        return _Box(self.shape, self.element_bytes, dims, self.offsets | other.offsets, kdim)

    def bytes(self) -> int:
        elements = max(len(self.offsets), 1)
        for dim, extent in enumerate(self.dims):
            if dim == self.kdim:
                continue
            elements *= (extent[1] - extent[0] + 1) if extent is not None else self.shape[dim]
        return elements * self.element_bytes


@dataclass
class _LoopInfo:
    scope: tn.ForScope
    var: str
    first: int
    last: int
    accesses: List[_Access] = field(default_factory=list)
    assigned: Set[str] = field(default_factory=set)  # Symbols assigned in the body (other than loop variables)
    read: Set[str] = field(default_factory=set)  # Symbols and containers read in the body
    boxes: Dict[str, _Box] = field(default_factory=dict)  # Elements accessed per iteration, per key
    local: Set[str] = field(default_factory=set)  # Containers written before being read in every block that reads them
    opaque: bool = False  # Contains nodes whose accesses cannot be analyzed


def _integer(expr) -> Optional[int]:
    try:
        value = sympy.sympify(expr)
    except (sympy.SympifyError, TypeError):
        return None
    return int(value) if value.is_Integer else None


def _as_expr(value):
    return sympy.sympify(symbolic.pystr_to_symbolic(value) if isinstance(value, str) else value)


def _classify(key: str, memlet: Memlet, write: bool, var: str, node) -> _Access:
    k = symbolic.symbol(var)
    region, kdims = [], []
    for dim, (start, end, step) in enumerate(memlet.subset.ndrange()):
        start, end = _as_expr(start), _as_expr(end)
        if start.has(k) or end.has(k):
            kdims.append((dim, start, end))
            region.append(None)
        else:
            first, last = _integer(start), _integer(end)
            region.append((first, last) if first is not None and last is not None else None)
    if not kdims:
        element = str(memlet.subset) if memlet.subset.num_elements() == 1 else None
        return _Access(key, write, 'fixed', region=region, memlet=memlet, node=node, element=element)
    if len(kdims) == 1:
        dim, start, end = kdims[0]
        offset = _integer(sympy.expand(start - k))
        if sympy.expand(start - end) == 0 and offset is not None:
            return _Access(key, write, 'affine', dim=dim, offset=offset, region=region, memlet=memlet, node=node)
    return _Access(key, write, 'unknown', memlet=memlet, node=node)


def _interval(expr, ranges: Dict[sympy.Symbol, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Bounds of an expression affine in the variables of ``ranges`` (with integer coefficients)."""
    expr = sympy.expand(_as_expr(expr))
    lo = hi = expr
    for sym in expr.free_symbols:
        if sym not in ranges:
            continue
        coefficient = expr.coeff(sym)
        if not coefficient.is_number:
            return None
        a, b = ranges[sym]
        lo = lo.subs(sym, a if coefficient > 0 else b)
        hi = hi.subs(sym, b if coefficient > 0 else a)
    lo, hi = _integer(lo), _integer(hi)
    return None if lo is None or hi is None else (lo, hi)


class _Analysis:
    """Accesses, footprints and privatizable containers of the candidate loops of a tree."""

    def __init__(self, root: tn.ScheduleTreeRoot):
        self.root = root
        self.containers = root.containers
        self.repository = repository_of(root)
        self.views = {n.target: n.source for n in root.preorder_traversal() if isinstance(n, tn.ViewNode)}

    def key(self, name: str) -> Optional[str]:
        """The memory a container name refers to (views resolved), or ``None`` for symbols."""
        seen = set()
        while name in self.views and name not in seen:
            seen.add(name)
            name = self.views[name]
        desc = self.containers.get(name)
        if desc is None:
            return None
        if getattr(desc, 'may_alias', False):
            return _MAY_ALIAS
        return name

    def loop(self, scope: tn.ScheduleTreeNode, min_trip_count: int) -> Optional[_LoopInfo]:
        """Information on ``scope`` if it is a coarse candidate: a unit-stride ``for`` loop with constant bounds, at
        least ``min_trip_count`` iterations, and a body containing loop nests."""
        if not isinstance(scope, tn.ForScope):
            return None
        spaces = iteration_spaces(scope, self.repository)
        if len(spaces) != 1 or spaces[0][2] is None:
            return None
        _, var, space = spaces[0]
        first, last = _integer(space.start), _integer(space.end)
        if space.stride != 1 or first is None or last is None or var in space.body_defined:
            return None
        if trip_count(space) is None or trip_count(space) < min_trip_count:
            return None
        if not any(isinstance(n, (tn.ForScope, tn.MapScope)) for n in scope.preorder_traversal() if n is not scope):
            return None  # Not coarse: fusing innermost loops is the vectorizer's business
        info = _LoopInfo(scope, var, first, last)
        self._collect(info)
        return info

    def _collect(self, info: _LoopInfo):
        # The ranges of the loops enclosing each node within the loop (split loops reuse variable names)
        node_ranges: Dict[int, Dict[sympy.Symbol, Tuple[int, int]]] = {}

        def walk(node: tn.ScheduleTreeNode, ranges: Dict[sympy.Symbol, Tuple[int, int]]):
            node_ranges[id(node)] = ranges
            if node is not info.scope and isinstance(node, (tn.ForScope, tn.MapScope)):
                ranges = dict(ranges)
                for _, var, space in iteration_spaces(node, self.repository):
                    bounds = None if space is None else (_integer(space.start), _integer(space.end))
                    if bounds is not None and None not in bounds:
                        ranges[symbolic.symbol(var)] = (min(bounds), max(bounds))
                    else:
                        ranges.pop(symbolic.symbol(var), None)
            for child in getattr(node, 'children', ()):
                walk(child, ranges)

        walk(info.scope, {})
        loop_variables = {v for n in info.scope.preorder_traversal() for v in _bound(n)}
        for n in info.scope.preorder_traversal():
            if n is info.scope:
                continue
            if isinstance(n, tn.TaskletNode) and getattr(n.node, 'side_effects', False):
                info.opaque = True
            covered_reads, covered_writes = set(), set()
            for memlet in memlets_of(n, 'in_memlets') + memlets_of(n, 'memlet'):
                key = self.key(memlet.data)
                if key is not None:
                    info.accesses.append(self._access(info, key, memlet, False, n, node_ranges[id(n)]))
                    covered_reads.add(memlet.data)
            for memlet in memlets_of(n, 'out_memlets'):
                key = self.key(memlet.data)
                if key is not None:
                    if memlet.wcr is not None:
                        info.accesses.append(_Access(key, True, 'unknown', memlet=memlet, node=n))
                    else:
                        info.accesses.append(self._access(info, key, memlet, True, n, node_ranges[id(n)]))
                    covered_writes.add(memlet.data)
            for name in names_read(n) - covered_reads:
                key = self.key(name)
                if key is not None and isinstance(self.containers[name], data.Scalar):  # E.g., in a condition
                    info.accesses.append(_Access(key, False, 'fixed', region=[(0, 0)], node=n, element='0'))
                elif key is not None:  # E.g., an array read in a condition, at indices not analyzed
                    info.accesses.append(_Access(key, False, 'unknown', node=n))
                else:
                    info.read.add(name)
            for name in names_written(n) - covered_writes:
                key = self.key(name)
                if key is not None:
                    info.accesses.append(_Access(key, True, 'unknown', node=n))
                elif name not in loop_variables:
                    info.assigned.add(name)
        info.read |= {a.key for a in info.accesses}
        info.local = self._local_containers(info)

    def _access(self, info: _LoopInfo, key: str, memlet: Memlet, write: bool, node, ranges) -> _Access:
        access = _classify(key, memlet, write, info.var, node)
        if key != memlet.data:  # Through a view or an alias: the indices are not those of the memory accessed
            access = _Access(key, write, 'unknown', memlet=memlet, node=node)
        # Footprint per iteration of the loop: the extent of the access over the inner loops
        desc = self.containers[memlet.data]
        shape = [_integer(x) or 1 for x in desc.shape] if key != _MAY_ALIAS else [1 << 18]
        dims: List[Optional[Tuple[int, int]]] = [None] * len(shape)
        if access.kind != 'unknown' and key != _MAY_ALIAS:
            for dim, (start, end, _) in enumerate(memlet.subset.ndrange()):
                lo, hi = _interval(start, ranges), _interval(end, ranges)
                if dim != access.dim and lo is not None and hi is not None:
                    dims[dim] = (min(lo[0], hi[0]), max(lo[1], hi[1]))
        offsets = {access.offset} if access.kind == 'affine' else set()
        new = _Box(shape, desc.dtype.bytes, dims, offsets, access.dim if access.kind == 'affine' else -1)
        info.boxes[key] = info.boxes[key].union(new) if key in info.boxes else new
        return access

    def _local_containers(self, info: _LoopInfo) -> Set[str]:
        """Containers accessed at fixed elements that every read in the loop finds written earlier in its own block
        (the same element, by a statement before it or before the scope containing it), such as scalar temporaries:
        no value flows between iterations through them."""
        writes: Dict[Tuple[str, str], List[tn.ScheduleTreeNode]] = {}
        for access in info.accesses:
            if access.write and access.kind == 'fixed' and access.element is not None:
                writes.setdefault((access.key, access.element), []).append(access.node)
        local = {a.key for a in info.accesses if a.kind == 'fixed' and a.key != _MAY_ALIAS}
        for access in info.accesses:
            if access.key not in local or access.write:
                continue
            if access.kind != 'fixed' or access.element is None:
                local.discard(access.key)
                continue
            writers = writes.get((access.key, access.element), [])
            covered, current = False, access.node
            while current is not info.scope and current.parent is not None and not covered:
                siblings = current.parent.children
                position = next(k for k, c in enumerate(siblings) if c is current)
                covered = any(w.parent is current.parent and any(c is w for c in siblings[:position]) for w in writers)
                current = current.parent
            if not covered:
                local.discard(access.key)
        return local


def _bound(node: tn.ScheduleTreeNode) -> Set[str]:
    if isinstance(node, tn.MapScope):
        return set(node.node.map.params)
    if isinstance(node, tn.LoopScope) and node.loop.loop_variable:
        return {node.loop.loop_variable}
    return set()


def _values(access: _Access, dim: int) -> Optional[range]:
    """The indices a ``'fixed'`` access touches in dimension ``dim``, or ``None`` if not constant."""
    if access.region is None or dim >= len(access.region) or access.region[dim] is None:
        return None
    first, last = access.region[dim]
    return range(min(first, last), max(first, last) + 1)


def _violates(a: _Access, b: _Access, first: _LoopInfo, second: _LoopInfo) -> bool:
    """Whether fusing could run an iteration of ``second`` that accesses (with ``b``) an element before an iteration of
    ``first`` that accesses it (with ``a``) and comes later in the fused loop's order is the wrong way round: that
    is, whether some ``k1`` of ``first`` and ``k2 < k1`` of ``second`` access the same element."""
    r1, r2 = range(first.first, first.last + 1), range(second.first, second.last + 1)
    if a.kind == 'unknown' or b.kind == 'unknown':
        return True
    if a.kind == 'affine' and b.kind == 'affine':
        if a.dim != b.dim:
            return True
        # k1 + a.offset == k2 + b.offset with k2 < k1  <=>  k1 - k2 == b.offset - a.offset > 0
        distance = b.offset - a.offset
        return distance > 0 and any(k1 - distance in r2 for k1 in r1)
    if a.kind == 'affine':  # b touches the same indices in every iteration
        values = _values(b, a.dim)
        touching = [k1 for k1 in r1 if values is None or k1 + a.offset in values]
        return bool(touching) and max(touching) > min(r2)
    if b.kind == 'affine':
        values = _values(a, b.dim)
        touching = [k2 for k2 in r2 if values is None or k2 + b.offset in values]
        return bool(touching) and max(r1) > min(touching)
    # Both touch the same elements in every iteration (unless their regions are disjoint)
    for x, y in zip(a.region or [], b.region or []):
        if x is not None and y is not None and (max(x) < min(y) or max(y) < min(x)):
            return False
    return max(r1) > min(r2)


def _legal(first: _LoopInfo, second: _LoopInfo, used_elsewhere) -> Optional[str]:
    """``None`` if fusing ``second`` into ``first`` preserves every dependence, else the reason why not."""
    if first.opaque or second.opaque:
        return 'side effects'
    if first.assigned & (second.read | second.assigned) or second.assigned & first.read:
        return 'symbol assigned in one loop and used in the other'
    if second.var != first.var and first.var in second.read:
        return 'loop variable name used in the other loop'
    by_key: Dict[str, List[_Access]] = {}
    for access in second.accesses:
        by_key.setdefault(access.key, []).append(access)
    for a in first.accesses:
        for b in by_key.get(a.key, ()):
            if not (a.write or b.write):
                continue
            if a.kind == 'fixed' and b.kind == 'fixed' and a.key in first.local and a.key in second.local:
                # Temporaries written before being read in each block: no value flows between the loops, but the
                # value left after the fused loop must be that of the second loop
                if first.last <= second.last or not used_elsewhere(a.key):
                    continue
            if _violates(a, b, first, second):
                return f'{a.key}: {"write" if a.write else "read"} in the first loop and ' \
                       f'{"write" if b.write else "read"} in the second'
    return None


def _group_local(group: List[_LoopInfo]) -> Set[str]:
    """Containers local (see ``_Analysis._local_containers``) in every loop of ``group`` that accesses them."""
    keys = {a.key for g in group for a in g.accesses}
    return {k for k in keys if all(k in g.local for g in group if any(a.key == k for a in g.accesses))}


def _boxes(infos: List[_LoopInfo]) -> Dict[str, _Box]:
    total: Dict[str, _Box] = {}
    for info in infos:
        for key, box in info.boxes.items():
            total[key] = total[key].union(box) if key in total else box
    return total


def _footprint(infos: List[_LoopInfo]) -> Tuple[int, int]:
    """Bytes accessed per iteration by loops fused together, and the part of it in containers indexed by the loop
    variable (different in every iteration)."""
    boxes = _boxes(infos).values()
    return sum(b.bytes() for b in boxes), sum(b.bytes() for b in boxes if b.offsets)


def _profitable(group: List[_LoopInfo], second: _LoopInfo, cache_bytes: int) -> Optional[str]:
    """``None`` if fusing ``second`` into the loops of ``group`` shortens the reuse distance of shared data that would
    otherwise leave the cache, without the fused iterations overflowing it; else the reason why not."""
    first = _boxes(group)
    reused = sum(min(second.boxes[k].bytes(), first[k].bytes()) for k in set(second.boxes) & set(first))
    if not reused:
        return 'no shared data'
    per_iteration, indexed = _footprint(group)
    trips = max(g.last for g in group) - min(g.first for g in group) + 1
    if per_iteration + indexed * (trips - 1) <= cache_bytes:
        return 'shared data already stays in cache'
    fused, _ = _footprint(group + [second])
    if fused > cache_bytes:
        return f'fused iteration footprint {fused} B exceeds the cache ({cache_bytes} B)'
    return None


class _Rename(ast.NodeTransformer):

    def __init__(self, old: str, new: str):
        self.old, self.new = old, new

    def visit_Name(self, node: ast.Name):
        if node.id == self.old:
            node.id = self.new
        return node


def _rename_code(code: CodeBlock, old: str, new: str) -> CodeBlock:
    if code is None or code.language.name != 'Python':
        return code
    body = [_Rename(old, new).visit(copy.deepcopy(s)) for s in code.code]
    return CodeBlock(body, code.language)


def _rename_symbol(scope: tn.ScheduleTreeNode, old: str, new: str):
    """Rename a symbol (e.g., the loop variable) in a subtree."""
    repl = {old: new}
    for n in scope.preorder_traversal():
        for attr in ('in_memlets', 'out_memlets'):
            memlets = getattr(n, attr, None)
            if isinstance(memlets, dict):
                for connector, memlet in list(memlets.items()):
                    memlet = copy.deepcopy(memlet)
                    memlet.replace(repl)
                    memlets[connector] = memlet
        if isinstance(getattr(n, 'memlet', None), Memlet):
            n.memlet = copy.deepcopy(n.memlet)
            n.memlet.replace(repl)
        for attr in ('condition', 'value'):
            if isinstance(getattr(n, attr, None), CodeBlock):
                setattr(n, attr, _rename_code(getattr(n, attr), old, new))
        if isinstance(n, tn.LoopScope):
            loop = copy.deepcopy(n.loop)
            loop.init_statement = _rename_code(loop.init_statement, old, new)
            loop.loop_condition = _rename_code(loop.loop_condition, old, new)
            loop.update_statement = _rename_code(loop.update_statement, old, new)
            if loop.loop_variable == old:
                loop.loop_variable = new
            n.loop = loop
        if isinstance(n, tn.MapScope):
            n.node = copy.deepcopy(n.node)
            n.node.map.range.replace(repl)
            n.node.map.params = [new if p == old else p for p in n.node.map.params]
        if isinstance(n, tn.TaskletNode) and old in n.node.free_symbols:
            n.node = copy.deepcopy(n.node)
            n.node.code = _rename_code(n.node.code, old, new)


def _guarded(body: list, var: str, first: int, last: int, fused_first: int, fused_last: int) -> list:
    if first <= fused_first and last >= fused_last:
        return body
    return [tn.IfScope(condition=CodeBlock(f'({var} >= {first}) and ({var} <= {last})'), children=body)]


def fuse_loops_for_reuse(stree: tn.ScheduleTreeScope,
                         cache_bytes: int = 256 * 1024,
                         min_trip_count: int = 2,
                         log: Optional[list] = None) -> int:
    """
    Fuse adjacent coarse loops over the same range when that shortens the reuse distance of the data they share.

    Loops that each run a loop nest per iteration (e.g., vertical loops around horizontal planes) and share data
    (a field that one computes and the next reads, or that both read) are fused, so the shared data of an iteration is
    still in cache when the second body uses it: ``for k: A(k); for k: B(k)`` becomes ``for k: A(k); B(k)``. When the
    ranges differ, the fused loop runs over both and each body is guarded by its own range (the guards are later
    removed by index-set splitting, which keeps the bodies together).

    Fusion must preserve every dependence: for each element accessed by both loops (at least once by a write), an
    iteration of the first loop that accesses it must not come after an iteration of the second loop that does (e.g.,
    the second loop may read ``x[k - 1]`` or ``x[k]`` of what the first writes, but not ``x[k + 1]``). Views are
    resolved to the containers they view, containers that may alias are treated as one, and accesses whose index in
    the loop variable is not affine prevent fusion. Temporaries that each block writes before reading (e.g., scalars
    shared by both bodies) carry no values between the loops and do not prevent it.

    Fusion is applied only where it pays off (a cost model of the cache, rather than fusing everything):
    * the loops share data (bytes accessed by both);
    * without fusion that data would leave the cache: the first loop's footprint over all its iterations exceeds
      ``cache_bytes``;
    * the fused iteration still fits: its footprint is at most ``cache_bytes``.
    Innermost loops are never fused (they are what vectorizes), and loops fused together are not split apart by the
    other passes, so fusion and index-set splitting do not undo each other.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param cache_bytes: The capacity of the cache level to keep reused data in (e.g., the L2 cache).
    :param min_trip_count: Only fuse loops that run at least this many iterations.
    :param log: If given, a list to which the reason for every candidate pair that was not fused is appended.
    :return: The number of loops fused into others.
    """
    root = stree.get_root()
    analysis = _Analysis(root)
    index = AccessIndex(root)
    fused_count = 0

    def visit(scope: tn.ScheduleTreeScope):
        nonlocal fused_count
        result: list = []
        group: List[_LoopInfo] = []
        changed = False
        for child in scope.children:
            info = analysis.loop(child, min_trip_count)
            if info is None:
                if isinstance(child, tn.ScheduleTreeScope):
                    visit(child)
                result.append(child)
                group = []
                continue
            if group:
                inside = {}
                for g in group + [info]:
                    index._count(g.scope, inside, {}, set())

                def used_elsewhere(key: str) -> bool:
                    return index.reads.get(key, 0) > inside.get(key, 0)

                merged = _LoopInfo(group[0].scope, group[0].var, min(g.first for g in group),
                                   max(g.last for g in group), [a for g in group for a in g.accesses],
                                   set().union(*(g.assigned for g in group)),
                                   set().union(*(g.read for g in group)), {}, _group_local(group),
                                   any(g.opaque for g in group))
                reason = _legal(merged, info, used_elsewhere) or _profitable(group, info, cache_bytes)
                if reason is None:
                    _fuse(result[-1], group, info)
                    group.append(info)
                    fused_count += 1
                    changed = True
                    continue
                if log is not None:
                    log.append(f'{group[0].scope.loop.label} + {info.scope.loop.label}: {reason}')
            result.append(child)
            group = [info]
        if changed:
            scope.children = []
            scope.add_children(result)

    visit(stree)
    return fused_count


def _fuse(target: tn.ForScope, group: List[_LoopInfo], second: _LoopInfo):
    """Fuse the loop ``second`` into ``target``, the fused loop of ``group``."""
    first_var = group[0].var
    if second.var != first_var:
        _rename_symbol(second.scope, second.var, first_var)
    old_first, old_last = min(g.first for g in group), max(g.last for g in group)
    new_first, new_last = min(old_first, second.first), max(old_last, second.last)
    body = _guarded(list(target.children), first_var, old_first, old_last, new_first, new_last)
    body += _guarded(list(second.scope.children), first_var, second.first, second.last, new_first, new_last)
    if (new_first, new_last) != (old_first, old_last):
        loop = copy.deepcopy(target.loop)
        loop.init_statement = CodeBlock(f'{first_var} = {new_first}')
        loop.loop_condition = CodeBlock(f'{first_var} < {new_last + 1}')
        loop.update_statement = CodeBlock(f'{first_var} = {first_var} + 1')
        target.loop = loop
    target.children = []
    target.add_children(body)
