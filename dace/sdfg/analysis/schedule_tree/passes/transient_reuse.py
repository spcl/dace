# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reusing the memory of transients whose live ranges do not overlap, and moving small transients to the stack."""
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import sympy

from dace import data, dtypes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (iteration_spaces, memlets_of, names_read, names_written,
                                                            repository_of)

# A region: per dimension, (first, last) as integers, or as equal symbolic expressions for dimensions indexed by
# variables of loops outside the context (the same in every access of one iteration)
_Box = List[Tuple[object, object]]


def _integer(value) -> Optional[int]:
    value = sympy.sympify(value)
    return int(value) if value.is_Integer else None


def _subtract(box: list, other: list) -> list:
    """``box`` minus ``other`` (integer boxes as lists of ``(first, last)``), as a list of disjoint boxes."""
    for (a, b), (c, d) in zip(box, other):
        if d < a or c > b:
            return [box]
    result, rest = [], list(box)
    for dim, ((a, b), (c, d)) in enumerate(zip(box, other)):
        if a < c:
            result.append(rest[:dim] + [(a, c - 1)] + rest[dim + 1:])
        if d < b:
            result.append(rest[:dim] + [(d + 1, b)] + rest[dim + 1:])
        rest[dim] = (max(a, c), min(b, d))
    return result


def _covered(read: Optional[_Box], writes: List[_Box]) -> bool:
    """Whether the union of ``writes`` contains ``read``."""
    if read is None:
        return False
    symbolic_dims = [d for d, (lo, hi) in enumerate(read) if _integer(lo) is None or _integer(hi) is None]
    integer_dims = [d for d in range(len(read)) if d not in symbolic_dims]
    remaining = [[(_integer(read[d][0]), _integer(read[d][1])) for d in integer_dims]]
    for write in writes:
        if write is None or len(write) != len(read):
            continue
        if any(
                sympy.expand(sympy.sympify(write[d][0]) -
                             read[d][0]) != 0 or sympy.expand(sympy.sympify(write[d][1]) - read[d][1]) != 0
                for d in symbolic_dims):
            continue
        bounds = [(_integer(write[d][0]), _integer(write[d][1])) for d in integer_dims]
        if any(lo is None or hi is None for lo, hi in bounds):
            continue
        remaining = [piece for box in remaining for piece in _subtract(box, bounds)]
        if not remaining:
            return True
    return not remaining


@dataclass
class _Use:
    node: tn.ScheduleTreeNode
    memlets: dict
    connector: str
    write: bool


@dataclass
class _Event:
    """What a child of a context does with a container: its reads that need values from before it (as regions of the
    context), the regions it certainly writes, and its extent in program order."""
    start: int
    end: int
    reads: List[Optional[_Box]] = field(default_factory=list)
    writes: List[Optional[_Box]] = field(default_factory=list)
    inner: Optional[List[Tuple[int, int]]] = None  # Live segments within one iteration, if the child is a loop whose
    # iterations do not pass values of the container to each other
    writes_at_all: bool = False  # Whether the child may write the container (also conditionally)


class _Liveness:
    """Live segments of transients in program order (preorder positions of the tree), taking loops into account: a
    container whose values do not flow between the iterations of a loop is live only in parts of the loop body."""

    def __init__(self, root: tn.ScheduleTreeRoot, trust_reads: bool):
        self.root = root
        self.trust_reads = trust_reads
        self.repository = repository_of(root)
        self.position: Dict[int, int] = {}
        self.last: Dict[int, int] = {}
        self._number(root, 0)
        self._spaces: Dict[int, Optional[List[Tuple[str, int, int]]]] = {}
        self._boxes: Dict[tuple, Optional[_Box]] = {}

    def _ranges(self, scope: tn.ScheduleTreeScope) -> Optional[List[Tuple[str, int, int]]]:
        """``(variable, first, last)`` of each variable a loop or map binds (in iteration order), or ``None`` if some
        bound is not a constant (computed once per scope)."""
        key = id(scope)
        if key not in self._spaces:
            result = []
            for _, var, space in iteration_spaces(scope, self.repository):
                first, last = (None, None) if space is None else (_integer(space.start), _integer(space.end))
                if first is None or last is None:
                    result = None
                    break
                result.append((var, first, last))
                if (last - first) * (1 if space.stride > 0 else -1) < 0:
                    result.append((var, None, None))  # Runs no iteration
            self._spaces[key] = result
        return self._spaces[key]

    def _number(self, node: tn.ScheduleTreeNode, pos: int) -> int:
        self.position[id(node)] = pos
        end = pos
        for child in getattr(node, 'children', ()):
            end = self._number(child, end + 1)
        self.last[id(node)] = end
        return end

    def _path(self, node: tn.ScheduleTreeNode, context: tn.ScheduleTreeNode) -> List[tn.ScheduleTreeNode]:
        """The ancestors of ``node`` below ``context``, outermost first (``node`` included)."""
        path = [node]
        while path[-1].parent is not context:
            path.append(path[-1].parent)
        return path[::-1]

    def _box(self, use: _Use, context: tn.ScheduleTreeNode) -> Optional[_Box]:
        """The region of an access, over the iterations of the loops between it and ``context``."""
        ranges: Dict[str, Tuple[int, int]] = {}
        for scope in self._path(use.node, context)[:-1]:
            if isinstance(scope, (tn.ForScope, tn.MapScope)):
                spaces = self._ranges(scope)
                if spaces is None or any(first is None for _, first, _ in spaces):
                    return None
                for var, first, last in spaces:
                    ranges[var] = (min(first, last), max(first, last))
        subset = use.memlets[use.connector].subset
        key = (tuple(tuple(r) for r in subset.ndrange()), tuple(sorted(ranges.items())))
        if key not in self._boxes:
            self._boxes[key] = self._region(subset, ranges)
        return self._boxes[key]

    @staticmethod
    def _region(subset, ranges: Dict[str, Tuple[int, int]]) -> Optional[_Box]:
        box = []
        for start, end, _ in subset.ndrange():
            bounds = []
            for expr, pick in ((start, 0), (end, 1)):
                expr = sympy.expand(sympy.sympify(expr))
                value = expr
                for sym in expr.free_symbols:
                    if str(sym) not in ranges:
                        continue
                    coefficient = expr.coeff(sym)
                    if not coefficient.is_number:
                        return None
                    lo, hi = ranges[str(sym)]
                    low_end = (coefficient > 0) == (pick == 0)
                    value = value.subs(sym, lo if low_end else hi)
                if any(str(sym) in ranges for sym in value.free_symbols):
                    return None
                bounds.append(value)
            box.append(tuple(bounds))
        return box

    def _unconditional(self, use: _Use, context: tn.ScheduleTreeNode) -> bool:
        """Whether an access runs in every iteration of ``context`` (not in a branch, nor in a loop that may not run)."""
        for scope in self._path(use.node, context)[:-1]:
            if not isinstance(scope, (tn.ForScope, tn.MapScope)):
                return False
            spaces = self._ranges(scope)
            if spaces is None or any(first is None for _, first, _ in spaces):
                return False
        return True

    def segments(self, uses: List[_Use]) -> Tuple[List[Tuple[int, int]], bool]:
        """Live segments of a container, and whether it is read before being written (live on entry)."""
        return self._analyze(self.root, uses)

    def _analyze(self, context: tn.ScheduleTreeNode, uses: List[_Use]) -> Tuple[List[Tuple[int, int]], bool]:
        # Group the uses by the child of the context containing them, in program order
        groups: Dict[int, Tuple[tn.ScheduleTreeNode, List[_Use]]] = {}
        for use in uses:
            child = self._path(use.node, context)[0]
            groups.setdefault(id(child), (child, []))[1].append(use)
        events = []
        for child, child_uses in sorted(groups.values(), key=lambda g: self.position[id(g[0])]):
            start, end = self.position[id(child)], self.last[id(child)]
            if isinstance(child, (tn.ForScope, tn.MapScope)):
                inner, exposed = self._analyze(child, child_uses)
                reads = [self._box(u, context) for u in child_uses if not u.write] if exposed else []
                event = _Event(start, end, reads, inner=None if exposed else inner)
            elif isinstance(child, tn.ScheduleTreeScope):
                event = _Event(start, end, [self._box(u, context) for u in child_uses if not u.write])
            else:  # A statement reads its inputs before writing its outputs
                event = _Event(start, start, [self._box(u, context) for u in child_uses if not u.write])
            event.writes = [self._box(u, context) for u in child_uses if u.write and self._unconditional(u, context)]
            event.writes_at_all = any(u.write for u in child_uses)
            events.append(event)

        # Chains of events through which a value of the container stays live
        chains: List[List[_Event]] = []
        chain_writes: List[_Box] = []
        all_writes: List[_Box] = []
        chain_written = any_written = False  # For trusted reads: whether the chain or context wrote at all before
        exposed = False
        for event in events:
            needs_older = False
            for read in event.reads:
                if self.trust_reads:
                    covered_here, covered_before = chain_written, any_written
                else:
                    covered_here = bool(chains) and _covered(read, chain_writes)
                    covered_before = covered_here or _covered(read, all_writes)
                if covered_here:
                    continue
                needs_older = True
                if not covered_before:
                    exposed = True
            if needs_older and len(chains) > 1:  # The value may come from any earlier chain: merge them all
                chains = [[e for chain in chains for e in chain]]
                chain_writes, chain_written = list(all_writes), any_written
            if not event.reads and event.writes or not chains:
                chains.append([event])
                chain_writes, chain_written = [], False
            else:
                chains[-1].append(event)
            chain_writes += event.writes
            all_writes += event.writes
            chain_written = chain_written or event.writes_at_all
            any_written = any_written or event.writes_at_all
        result = []
        for chain in chains:
            if len(chain) == 1 and chain[0].inner is not None:
                result += chain[0].inner  # A loop whose iterations do not pass values to each other
            else:
                result.append((chain[0].start, chain[-1].end))
        if exposed and chains:  # Live on entry to the context: from its beginning
            first = result[0] if result else (self.position[id(context)], self.position[id(context)])
            result[0:1] = [(self.position[id(context)], first[1])]
        return result, exposed


def _overlaps(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> bool:
    return any(x0 <= y1 and y0 <= x1 for x0, x1 in a for y0, y1 in b)


def _fits(desc: data.Array, slot: data.Array) -> bool:
    """Whether every element of ``desc`` can be addressed in ``slot`` with the same indices."""
    if (desc.dtype != slot.dtype or len(desc.shape) != len(slot.shape) or desc.storage != slot.storage
            or any(_integer(o) != 0 for o in list(desc.offset) + list(slot.offset))):
        return False
    for extent, slot_extent, stride, slot_stride in zip(desc.shape, slot.shape, desc.strides, slot.strides):
        extent, slot_extent = _integer(extent), _integer(slot_extent)
        if extent is None or slot_extent is None or extent > slot_extent:
            return False
        if extent > 1 and sympy.sympify(stride) != sympy.sympify(slot_stride):
            return False
    return True


def _usable_accesses(root: tn.ScheduleTreeRoot) -> Tuple[Dict[str, List[_Use]], Set[str]]:
    """The memlet uses of every container, and the containers accessed in ways other than through the memlets of
    statements (e.g., by name in conditions, as copy targets or through views)."""
    uses: Dict[str, List[_Use]] = {}
    opaque: Set[str] = set()
    names = set(root.containers.keys())
    for node in root.preorder_traversal():
        covered = set()
        for attr, write in (('in_memlets', False), ('out_memlets', True)):
            memlets = getattr(node, attr, None)
            if isinstance(memlets, dict):
                for connector, memlet in memlets.items():
                    uses.setdefault(memlet.data, []).append(_Use(node, memlets, connector, write))
                    covered.add(memlet.data)
                    if memlet.wcr is not None or memlet.other_subset is not None:
                        opaque.add(memlet.data)
            elif memlets is not None:
                opaque |= {m.data for m in memlets_of(node, attr)}
        opaque |= {m.data for m in memlets_of(node, 'memlet')}
        if isinstance(node, tn.ViewNode):
            opaque |= {node.source, node.target}
        opaque |= (names_read(node) | names_written(node)) & names - covered
    return uses, opaque


def reuse_transients(stree: tn.ScheduleTreeScope, trust_reads: bool = False) -> int:
    """
    Let transient arrays whose live ranges do not overlap share memory.

    Live ranges are computed on the program order of the tree, loop by loop: a transient whose values do not flow
    between the iterations of a loop (every element an iteration reads was written earlier in that iteration, which
    is proven by comparing the regions the accesses cover over the inner loops) is live only in the part of the loop
    body between its writes and its reads, so the per-iteration temporaries of a loop body (e.g., planes of a vertical
    loop) can share memory; otherwise it is live throughout the loop. Transients are then packed into as few slots as
    possible (a greedy sweep over the live segments in program order), where a transient can use a slot of the same
    element type whose extents are at least its own and whose strides equal its own in every dimension it uses more
    than one index of, so that its accesses need not change beyond the name.

    :param stree: The schedule tree to transform in place.
    :param trust_reads: Assume that every read of a transient in a loop iteration reads an element written earlier in
                        that iteration if the iteration writes the transient before at all (as in stencil programs,
                        where a temporary is only read where it was computed), instead of proving it.
    :return: The number of transients that now share the memory of another.
    """
    root = stree.get_root()
    containers = root.containers
    uses, opaque = _usable_accesses(root)
    liveness = _Liveness(root, trust_reads)
    candidates = []
    for name, name_uses in uses.items():
        desc = containers.get(name)
        if (desc is None or not desc.transient or type(desc) is not data.Array or name in opaque
                or getattr(desc, 'may_alias', False) or _integer(desc.total_size) is None or desc.total_size == 1):
            continue
        if desc.lifetime not in (dtypes.AllocationLifetime.Scope, dtypes.AllocationLifetime.SDFG,
                                 dtypes.AllocationLifetime.State, dtypes.AllocationLifetime.Persistent):
            continue
        segments, exposed = liveness.segments(name_uses)
        if exposed or not segments:
            continue  # Read before written: its value may come from outside (or a previous call)
        candidates.append((min(s for s, _ in segments), name, segments))

    # Largest first within the sweep order, so that slots take the shape of their largest member
    candidates.sort(key=lambda c: (c[0], -int(containers[c[1]].total_size)))
    slots: List[Tuple[data.Array, List[str], List[Tuple[int, int]]]] = []
    for _, name, segments in candidates:
        desc = containers[name]
        for k, (slot, members, occupied) in enumerate(slots):
            if _overlaps(segments, occupied):
                continue
            if _fits(desc, slot):
                members.append(name)
                occupied += segments
                break
            if all(_fits(containers[m], desc) for m in members):  # The new member can host the others
                slots[k] = (desc, members + [name], occupied + segments)
                break
        else:
            slots.append((desc, [name], list(segments)))

    shared = 0
    for slot, members, _ in slots:
        if len(members) < 2:
            continue
        slot_desc = copy.deepcopy(slot)
        slot_desc.transient = True
        if any(containers[m].lifetime == dtypes.AllocationLifetime.Persistent for m in members):
            slot_desc.lifetime = dtypes.AllocationLifetime.Persistent
        slot_name = data.find_new_name('__reused', containers)
        containers[slot_name] = slot_desc
        for member in members:
            for use in uses[member]:
                old = use.memlets[use.connector]
                memlet = copy.deepcopy(old)
                memlet.data = slot_name
                use.memlets[use.connector] = memlet
            del containers[member]
        shared += len(members)
    return shared


def move_small_transients_to_stack(stree: tn.ScheduleTreeScope,
                                   max_array_bytes: int = 4096,
                                   max_total_bytes: int = 128 * 1024) -> int:
    """
    Allocate small transient arrays of constant size on the stack (``StorageType.Register``) rather than on the heap,
    as auto-optimization does for SDFGs: their accesses need no base pointers loaded from memory, and they cannot
    alias other data. Arrays are chosen by how often they are accessed, until the total reaches ``max_total_bytes``
    (keep it below the stack size of the threads that run the program, e.g., ``OMP_STACKSIZE``). Run after
    :func:`reuse_transients`, which reduces how much memory the transients need.

    :param stree: The schedule tree to transform in place.
    :param max_array_bytes: Only move arrays of at most this many bytes.
    :param max_total_bytes: Move arrays of at most this many bytes in total.
    :return: The number of arrays moved.
    """
    root = stree.get_root()
    containers = root.containers
    uses, opaque = _usable_accesses(root)
    candidates = []
    for name, desc in containers.items():
        if (not desc.transient or type(desc) is not data.Array or name in opaque
                or desc.storage not in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)):
            continue
        size = _integer(desc.total_size)
        if size is None or size * desc.dtype.bytes > max_array_bytes:
            continue
        candidates.append((-len(uses.get(name, ())), name, size * desc.dtype.bytes))
    moved, total = 0, 0
    for _, name, size in sorted(candidates):
        if total + size > max_total_bytes:
            continue
        desc = containers[name]
        desc.storage = dtypes.StorageType.Register
        desc.lifetime = dtypes.AllocationLifetime.Scope
        total += size
        moved += 1
    return moved
