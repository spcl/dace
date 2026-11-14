# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop a memset whose array the next map overwrites element for element before anyone reads it.

``s = np.zeros(N); for i in range(N): s[i] = r[i] - alpha * v[i]`` zeroes ``s`` only for the map
to replace every zero. Inside an iterative solver's loop (``bicgstab`` holds two of them) that is
an N-element store per iteration with no observer, and one the CPU transfer rule runs on a single
core because the loop re-enters it.

The memset is dead when, in its own state:

* it fills the whole of a transient array, and its access node passes no data on -- every edge out
  of it is an empty ordering edge;
* a later access node of the same array, reachable from it, takes a map's write that covers every
  element exactly once: the inner write is ``array[p0, p1, ...]`` with one distinct map parameter
  per axis, each ranging over that axis' whole extent, without WCR;
* the map does not read the array, and every other access node of the array in the state lies
  downstream of the overwrite.

A memset alone in its state (the access node it writes is the array's only one there) is dead too
when the control flow after it overwrites the array before anything reads it. The walk follows the
unconditional chain of blocks from the memset's state -- leaving a plain region or a branch at its
end, never a loop body -- skips blocks that do not touch the array, and stops at the first that does,
which must overwrite all of it first:

* a state, by the rule above;
* a loop ``for v in range(0, n)`` over an axis of extent ``n``, with no jump and no read of the array
  anywhere in it, whose body chain holds a state that writes the slice ``array[..., v, ...]`` whole;
* a conditional with an ``else`` whose every branch does, its conditions not reading the array.

A memset, or a top-level map writing a transient element for element (``np.zeros`` lowered as a map), shrinks to
what a copy in its state leaves when the copy overwrites all of it but a prefix or suffix of one axis before anything
reads it (``sink = np.zeros(n + 1); sink[0:n] = c`` zeroes row ``n`` only); it goes when the copy leaves nothing.

A fill ahead of a loop is dead when no iteration observes it, even when the array is read inside the loop. quatrex_rgf
allocates its per-energy workspaces with ``np.zeros`` ahead of the energy loop::

    xr_d = np.zeros((NB, BS, BS))
    for e in range(NE):
        xr_d[0] = ...
        for i in range(NB - 1):
            xr_d[i + 1] = f(xr_d[i])

Each iteration writes the whole array before it reads any of it, and the fill keeps ``LoopToMap`` from privatizing the
workspace. The fill must be constant and the array's only access outside the loop, the writes that tile the array must
run unconditionally in one iteration -- in the body or in unit-stride loops nested in it, outside conditionals and
nested SDFGs -- and no iteration may read the transient before writing it.

Outside a loop, the overwrite need not cover the whole array, only every subset of it that anything reads: an element
nobody reads never shows its fill. The write may index an axis by ``p + c`` (``al[i + 2]`` over ``0:n + 3``). A read
whose subset is not a range over symbols that no interstate edge, loop or scope rebinds (a control-flow read, a read
through a written View) counts as unknown and keeps the fill.

A memset of a program argument (``out[:] = 0; out[:] = a + 1``) goes or shrinks by the same rules, in the outermost
SDFG only and without the loop rule or the read coverage: the caller sees every element it leaves.

Anything else (a partial or indirect write, a conditional one without ``else``, a read in between, a
nested SDFG writing it) keeps the memset.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg.sdfg import interstate_bound_names
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis

#: Axes a write must fix to one index instead of covering: ``{axis: index}`` for the loop around it.
Fixed = dict[int, symbolic.SymbolicType]


@dataclass(slots=True)
class Reads:
    """Every subset of each array of ``sdfg`` that something reads, gathered in one walk on first use: out edges of its
    access nodes, WCR writes and writes out of a nested SDFG (which may read first). ``None`` marks an array read where
    no range says what: in control flow, through a written View, or at an index naming a ``bound`` symbol (one an
    interstate edge, a loop or a scope assigns)."""

    sdfg: SDFG
    boxes: dict[str, list[Range] | None] | None = None
    bound: set[str] = field(default_factory=set)

    def gather(self) -> dict[str, list[Range] | None]:
        self.bound = interstate_bound_names(self.sdfg)
        states = list(self.sdfg.states())
        for state in states:
            for entry in (n for n in state.nodes() if isinstance(n, nodes.MapEntry)):
                self.bound.update(entry.map.params)
                self.bound.update(c for c in entry.in_connectors if not c.startswith("IN_"))
        boxes: dict[str, list[Range] | None] = {}
        for region in self.sdfg.all_control_flow_regions():
            boxes.update(dict.fromkeys(control_reads(region)))
        for state in states:
            for node in state.data_nodes():
                for edge in state.all_edges(node):
                    if boxes.get(node.data, []) is None or edge.data.is_empty():
                        continue
                    if edge.src is node:
                        subset = edge.data.get_src_subset(edge, state)
                    elif isinstance(edge.src, nodes.AccessNode) and isinstance(edge.src.desc(self.sdfg), data.View):
                        subset = None
                    elif edge.data.wcr is not None or isinstance(edge.src, nodes.NestedSDFG):
                        subset = edge.data.get_dst_subset(edge, state)
                    else:
                        continue
                    if isinstance(subset, Range) and not subset.free_symbols & self.bound:
                        boxes.setdefault(node.data, []).append(subset)
                    else:
                        boxes[node.data] = None
        return boxes

    def covered_by(self, name: str, box: Range) -> bool:
        """Whether ``box``, over symbols nothing rebinds, holds every subset of ``name`` that is read."""
        if self.boxes is None:
            self.boxes = self.gather()
        reads = self.boxes.get(name, [])
        if reads is None or box.free_symbols & self.bound:
            return False
        facts = self.sdfg.facts(self.bound)
        return all(box.covers(read, facts) for read in reads)


def covers_every_element(
    state: SDFGState,
    written: nodes.AccessNode,
    desc: data.Array,
    fixed: Fixed | None = None,
    reads: Reads | None = None,
) -> bool:
    """Whether ``written``'s only writer is a map storing each element of ``desc`` exactly once, unread.

    With ``fixed``, each axis it names holds that one index instead, and the rest is covered once. Without, given
    ``reads``, every element read suffices.
    """
    fixed = fixed or {}
    reads = None if fixed else reads
    in_edges = state.in_edges(written)
    if len(in_edges) == 1 and isinstance(in_edges[0].src, nodes.AccessNode):
        return copy_covers(state, in_edges[0], desc, fixed, reads)
    if len(in_edges) != 1 or not isinstance(in_edges[0].src, nodes.MapExit):
        return False
    outer = in_edges[0]
    if outer.data.dynamic or outer.data.wcr is not None or outer.src_conn is None:
        return False
    exit_node = outer.src
    entry = state.entry_node(exit_node)
    inner = list(state.in_edges_by_connector(exit_node, "IN_" + outer.src_conn[len("OUT_") :]))
    if len(inner) != 1 or inner[0].data.dynamic or inner[0].data.wcr is not None or inner[0].data.subset is None:
        return False
    ranges = dict(zip(entry.map.params, entry.map.range.ranges))
    seen = set()
    box = []
    for axis, (begin, end, step) in enumerate(inner[0].data.subset.ranges):
        if axis in fixed:
            if begin != end or symbolic.simplify(begin - fixed[axis]) != 0:
                return False
            continue
        # ``p + c``: one parameter with unit coefficient, the offset free of every parameter.
        params = [s for s in symbolic.pystr_to_symbolic(begin).free_symbols if str(s) in ranges]
        if begin != end or len(params) != 1 or str(params[0]) in seen:
            return False
        offset = symbolic.simplify(begin - params[0])
        if any(str(s) in ranges for s in offset.free_symbols):
            return False
        seen.add(str(params[0]))
        low, high, stride = ranges[str(params[0])]
        if stride != 1:
            return False
        box.append((low + offset, high + offset, 1))
    # One parameter per axis and no other: a parameter left over repeats the stores, so it is still a map whose
    # iterations outnumber the elements -- kept, not reasoned about.
    if len(seen) != len(desc.shape) - len(fixed) or len(entry.map.params) != len(seen):
        return False
    if any(e.data.data == written.data for e in state.in_edges(entry)):
        return False
    extents = [extent for axis, extent in enumerate(desc.shape) if axis not in fixed]
    if all(symbolic.simplify(b) == 0 and symbolic.simplify(e + 1 - n) == 0 for (b, e, _), n in zip(box, extents)):
        return True
    return reads is not None and reads.covered_by(written.data, Range(box))


def copy_covers(state: SDFGState, edge, desc: data.Array, fixed: Fixed, reads: Reads | None = None) -> bool:
    """Whether the copy ``edge`` writes every element of ``desc`` (``fixed`` axes at that one index) or read."""
    if edge.data.dynamic or edge.data.wcr is not None or edge.src.data == edge.dst.data:
        return False
    subset = edge.data.get_dst_subset(edge, state)
    if subset is None or len(subset.ranges) != len(desc.shape):
        return False
    if reads is not None and isinstance(subset, Range) and reads.covered_by(edge.dst.data, subset):
        return True
    for axis, ((begin, end, step), extent) in enumerate(zip(subset.ranges, desc.shape)):
        if axis in fixed:
            if begin != end or symbolic.simplify(begin - fixed[axis]) != 0:
                return False
        elif symbolic.simplify(begin) != 0 or step != 1 or symbolic.simplify(end + 1 - extent) != 0:
            return False
    return True


def overwritten_in_state(
    state: SDFGState, name: str, desc: data.Array, fixed: Fixed | None = None, reads: Reads | None = None
) -> bool:
    """Whether a map in ``state`` overwrites ``name`` (``fixed`` axes at one index) before the state reads it."""
    accesses = [n for n in state.data_nodes() if n.data == name]
    for written in accesses:
        if not covers_every_element(state, written, desc, fixed, reads):
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in accesses):
            return True
    return False


def dead_fill(
    state: SDFGState, fill: FillLibraryNode, reads: Reads | None = None
) -> tuple[nodes.AccessNode, tuple] | None:
    """The access node ``fill`` writes and what of it is dead: :func:`copy_leaves` of the copy that overwrites all of
    it but a prefix or suffix of one axis, ``(-1, ())`` when all of it is; ``None`` when none of it is."""
    if any(not e.data.is_empty() for e in state.in_edges(fill)):
        return None
    out_edges = state.out_edges(fill)
    if len(out_edges) != 1 or not isinstance(out_edges[0].dst, nodes.AccessNode):
        return None
    zeroed = out_edges[0].dst
    sdfg = state.sdfg
    desc = sdfg.arrays[zeroed.data]
    if not isinstance(desc, data.Array) or isinstance(desc, data.View):
        return None
    if not desc.transient:
        # The caller sees every element of an argument: only a whole overwrite kills its fill, and only in the SDFG
        # the caller calls, where nothing outside it reads the argument in between.
        if sdfg.parent is not None:
            return None
        reads = None
    subset = out_edges[0].data.subset
    if subset is None or symbolic.simplify(subset.num_elements() - desc.total_size) != 0:
        return None
    if any(not e.data.is_empty() for e in state.out_edges(zeroed)) or len(state.in_edges(zeroed)) != 1:
        return None
    others = [n for n in state.data_nodes() if n.data == zeroed.data and n is not zeroed]
    reachable = set(state.bfs_nodes(zeroed))
    for written in others:
        if written not in reachable:
            continue
        whole = covers_every_element(state, written, desc, reads=reads)
        left = (-1, ()) if whole else copy_leaves(state, written, desc)
        if left is None:
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in others):
            return zeroed, left
    if not others and overwritten_after(state, zeroed.data, desc, reads):
        return zeroed, (-1, ())
    return None


def copy_leaves(state: SDFGState, written: nodes.AccessNode, desc: data.Array) -> tuple[int, tuple] | None:
    """What a copy into ``written`` leaves of ``desc``: ``(axis, (begin, end, 1))`` when it writes the whole array but a
    prefix or suffix of one axis, ``(-1, ())`` when it writes all of it; ``None`` for anything else."""
    in_edges = state.in_edges(written)
    if len(in_edges) != 1 or not isinstance(in_edges[0].src, nodes.AccessNode):
        return None
    edge = in_edges[0]
    if edge.data.dynamic or edge.data.wcr is not None or edge.src.data == written.data:
        return None
    subset = edge.data.get_dst_subset(edge, state)
    if subset is None or len(subset.ranges) != len(desc.shape):
        return None
    left: tuple[int, tuple] = (-1, ())
    for axis, ((begin, end, step), extent) in enumerate(zip(subset.ranges, desc.shape)):
        low, high = symbolic.simplify(begin) == 0, symbolic.simplify(end + 1 - extent) == 0
        if low and high and step == 1:
            continue
        if left[0] != -1 or step != 1 or low == high:
            return None
        left = (axis, (end + 1, extent - 1, 1)) if low else (axis, (0, begin - 1, 1))
    return left


def dead_part_of_fill_map(
    state: SDFGState, entry: nodes.MapEntry, reads: Reads | None = None
) -> tuple[nodes.AccessNode, tuple] | None:
    """The array a top-level map writes element for element, and what of it a copy in the same state overwrites before
    anything reads it: the :func:`copy_leaves` of that copy, or all of it when a map there or the control flow after
    overwrites it (:func:`covers_every_element`); ``None`` when nothing of the map's work is dead."""
    exit_node = state.exit_node(entry)
    out_edges = state.out_edges(exit_node)
    if len(out_edges) != 1 or not isinstance(out_edges[0].dst, nodes.AccessNode):
        return None
    zeroed = out_edges[0].dst
    desc = state.sdfg.arrays[zeroed.data]
    if not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, data.View):
        return None
    if any(not e.data.is_empty() for e in state.out_edges(zeroed)) or len(state.in_edges(zeroed)) != 1:
        return None
    if not covers_every_element(state, zeroed, desc):
        return None
    scope = state.scope_subgraph(entry).nodes()
    if any(isinstance(n, nodes.CodeNode) and n.has_side_effects(state.sdfg) for n in scope):
        return None
    others = [n for n in state.data_nodes() if n.data == zeroed.data and n is not zeroed]
    reachable = set(state.bfs_nodes(zeroed))
    for written in others:
        if written not in reachable:
            continue
        left = copy_leaves(state, written, desc)
        if left is None and covers_every_element(state, written, desc, reads=reads):
            left = (-1, ())
        if left is None:
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in others):
            return zeroed, left
    if not others and overwritten_after(state, zeroed.data, desc, reads):
        return zeroed, (-1, ())
    return None


def narrow_fill_map(state: SDFGState, entry: nodes.MapEntry, zeroed: nodes.AccessNode, left: tuple) -> None:
    """Restrict the map writing ``zeroed`` to the ``left`` part of one axis, or drop the map when nothing is left."""
    axis, kept = left
    if axis == -1:
        scope = state.scope_subgraph(entry).nodes()
        sources = {e.src for e in state.in_edges(entry)}
        state.keep_order_through([*scope, zeroed])
        state.remove_nodes_from([*scope, zeroed])
        state.remove_nodes_from([n for n in sources if isinstance(n, nodes.AccessNode) and state.degree(n) == 0])
        return
    exit_node = state.exit_node(entry)
    inner = next(iter(state.in_edges(exit_node)))
    # The write is ``p + c`` (:func:`covers_every_element`): ``kept`` is in array coordinates, ``p`` runs ``c`` lower.
    index = symbolic.pystr_to_symbolic(inner.data.subset.ranges[axis][0])
    param = next(s for s in index.free_symbols if str(s) in entry.map.params)
    offset = symbolic.simplify(index - param)
    entry.map.range[entry.map.params.index(str(param))] = (kept[0] - offset, kept[1] - offset, 1)
    outer = state.out_edges(exit_node)[0]
    outer.data.subset[axis] = kept


def drop_fill(state: SDFGState, fill: FillLibraryNode, zeroed: nodes.AccessNode) -> None:
    """Remove ``fill`` and the access node it writes, keeping any order that ran through them."""
    state.keep_order_through([fill, zeroed])
    state.remove_node(fill)
    state.remove_node(zeroed)


def blocks_within(block: ControlFlowBlock) -> Iterator[ControlFlowBlock]:
    """``block`` and every block nested in it, through regions, loops and branches."""
    yield block
    if isinstance(block, ConditionalBlock):
        for _, branch in block.branches:
            yield from blocks_within(branch)
    elif isinstance(block, AbstractControlFlowRegion):
        for inner in block.nodes():
            yield from blocks_within(inner)


def control_reads(block: ControlFlowBlock) -> set[str]:
    """The names the control flow of ``block`` itself reads: edge conditions and assignments, branch and loop code."""
    names: set[str] = set()
    if isinstance(block, ConditionalBlock):
        for condition, _ in block.branches:
            if condition is not None:
                names |= condition.get_free_symbols()
    if isinstance(block, LoopRegion):
        for code in (block.loop_condition, block.init_statement, block.update_statement):
            if code is not None:
                names |= code.get_free_symbols()
    if isinstance(block, AbstractControlFlowRegion) and not isinstance(block, ConditionalBlock):
        for edge in block.edges():
            names |= edge.data.read_symbols()
    return names


def touches(block: ControlFlowBlock, name: str, reads_only: bool = False) -> bool:
    """Whether anything in ``block`` reads ``name`` -- or, unless ``reads_only``, accesses it at all."""
    for inner in blocks_within(block):
        if name in control_reads(inner):
            return True
        if not isinstance(inner, SDFGState):
            continue
        for node in inner.data_nodes():
            if node.data != name:
                continue
            if not reads_only:
                return True
            if any(not e.data.is_empty() for e in inner.out_edges(node)):
                return True
            if any(e.data.wcr is not None for e in inner.in_edges(node)):
                return True
    return False


def unconditional_successor(graph: ControlFlowRegion, block: ControlFlowBlock, name: str) -> ControlFlowBlock | None:
    """The one block ``block`` always passes to in ``graph``, if its only out edge does not read ``name``."""
    out_edges = graph.out_edges(block)
    if len(out_edges) != 1 or not out_edges[0].data.is_unconditional() or name in out_edges[0].data.read_symbols():
        return None
    return out_edges[0].dst


def overwritten_by(
    block: ControlFlowBlock, name: str, desc: data.Array, fixed: Fixed, reads: Reads | None = None
) -> bool:
    """Whether executing ``block`` overwrites ``name`` whole (``fixed`` axes at one index) before it reads it."""
    if isinstance(block, SDFGState):
        return overwritten_in_state(block, name, desc, fixed, reads)
    if isinstance(block, ConditionalBlock):
        if name in control_reads(block) or not any(condition is None for condition, _ in block.branches):
            return False
        return all(
            overwritten_from(branch, branch.start_block, name, desc, fixed, reads) is True
            for _, branch in block.branches
        )
    if isinstance(block, LoopRegion):
        return not fixed and overwritten_by_loop(block, name, desc)
    if isinstance(block, ControlFlowRegion) and block.nodes():
        return overwritten_from(block, block.start_block, name, desc, fixed, reads) is True
    return False


def overwritten_by_loop(loop: LoopRegion, name: str, desc: data.Array) -> bool:
    """Whether ``loop`` sweeps one axis of ``name`` from 0 to its extent, each iteration overwriting that slice unread."""
    variable = loop.loop_variable
    if not variable or loop_analysis.loop_jumps(loop) or touches(loop, name, reads_only=True):
        return False
    if any(isinstance(b, ReturnBlock) for b in blocks_within(loop)):
        return False
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    if start is None or end is None or loop_analysis.get_loop_stride(loop) != 1 or symbolic.simplify(start) != 0:
        return False
    axes = [axis for axis, extent in enumerate(desc.shape) if symbolic.simplify(end + 1 - extent) == 0]
    if not axes or not loop.nodes():
        return False
    # The body's chain from its start: a block on it runs every iteration, since nothing jumps out.
    block = loop.start_block
    index = symbolic.pystr_to_symbolic(variable)
    while block is not None:
        if any(overwritten_by(block, name, desc, {axis: index}) for axis in axes):
            return True
        block = unconditional_successor(loop, block, name)
    return False


def overwritten_from(
    graph: ControlFlowRegion,
    block: ControlFlowBlock | None,
    name: str,
    desc: data.Array,
    fixed: Fixed,
    reads: Reads | None = None,
) -> bool | None:
    """Follow ``graph``'s unconditional chain from ``block`` to the first block touching ``name``.

    :returns: whether that block overwrites ``name`` before reading it; ``None`` if the chain ends or
              branches before any block touches it.
    """
    while block is not None:
        if touches(block, name):
            return overwritten_by(block, name, desc, fixed, reads)
        block = unconditional_successor(graph, block, name)
    return None


def overwritten_after(state: SDFGState, name: str, desc: data.Array, reads: Reads | None = None) -> bool:
    """Whether the control flow after ``state`` overwrites ``name`` whole before anything reads it."""
    block: ControlFlowBlock = state
    while True:
        graph = block.parent_graph
        following = unconditional_successor(graph, block, name)
        while following is not None:
            if touches(following, name):
                return overwritten_by(following, name, desc, {}, reads)
            block, following = following, unconditional_successor(graph, following, name)
        # The chain ran out untouched: at the end of a plain region or a branch, control passes on after it.
        if (
            graph.out_degree(block) != 0
            or isinstance(graph, (LoopRegion, SDFG))
            or isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock))
        ):
            return False
        block = graph.parent_graph if isinstance(graph.parent_graph, ConditionalBlock) else graph


def same(a: Any, b: Any) -> bool:
    """Whether two index expressions are provably one value (no sign assumptions: an index may be 0)."""
    return symbolic.equal(a, b, is_length=False) is True


def enclosing_loops(state: SDFGState, loop: LoopRegion) -> list[LoopRegion] | None:
    """The loops between ``state`` and ``loop`` (innermost first), or ``None`` when a conditional is in between."""
    loops = []
    region = state.parent_graph
    while region is not loop:
        if isinstance(region, ConditionalBlock) or region is None or isinstance(region, SDFG):
            return None
        if isinstance(region, LoopRegion):
            loops.append(region)
        region = region.parent_graph
    return loops


def swept_interval(begin, end, loops: list[LoopRegion]) -> tuple[Any, Any] | None:
    """The interval ``[begin, end]`` of one axis covers over the unit-stride ``loops``: a ``c + v`` index sweeps
    ``[c + start, c + last]``, a ``c - v`` one ``[c - last, c - start]``; an axis naming no loop variable is itself.
    Anything else is ``None``."""
    names = {loop.loop_variable: loop for loop in loops}
    used = {str(s) for s in symbolic.pystr_to_symbolic(begin).free_symbols} & names.keys()
    if not used:
        return begin, end
    if len(used) != 1 or not same(begin, end):
        return None
    loop = names[used.pop()]
    start, last = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    if start is None or last is None or not same(loop_analysis.get_loop_stride(loop), 1):
        return None
    index = symbolic.pystr_to_symbolic(begin)
    var = next(s for s in index.free_symbols if str(s) == loop.loop_variable)
    for sign in (1, -1):
        offset = index - sign * var
        if loop.loop_variable not in {str(s) for s in offset.free_symbols}:
            return (start + offset, last + offset) if sign == 1 else (offset - last, offset - start)
    return None


def tiles_every_iteration(loop: LoopRegion, name: str, desc: data.Data) -> bool:
    """Whether the unconditional writes of ``name`` in one iteration of ``loop`` cover all of it: every axis whole but
    one, along which some of the swept intervals chain from 0 to its extent."""
    intervals: dict[int, list[tuple[Any, Any]]] = {}
    for state in [block for block in loop.all_control_flow_blocks() if isinstance(block, SDFGState)]:
        accesses = [n for n in state.data_nodes() if n.data == name]
        if not accesses:
            continue
        loops = enclosing_loops(state, loop)
        scopes = state.scope_dict()
        for node in accesses:
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    continue
                if loops is None or scopes[node] is not None or edge.data.wcr is not None:
                    return False
                # A dynamic or nested write may store nothing.
                if any(e.data.dynamic or isinstance(e.src, nodes.NestedSDFG) for e in state.memlet_tree(edge).leaves()):
                    return False
                subset = edge.data.get_dst_subset(edge, state)
                if not isinstance(subset, Range) or len(subset.ranges) != len(desc.shape):
                    return False
                partial = [
                    axis
                    for axis, ((b, e, step), extent) in enumerate(zip(subset.ranges, desc.shape))
                    if not (same(b, 0) and same(e, extent - 1) and same(step, 1))
                ]
                if len(partial) > 1:
                    return False
                # A nested loop may not run: only the tiles, which shrink with its trips, cover anyway.
                if not partial and not loops:
                    return True
                if not partial:
                    return False
                (axis,) = partial
                begin, end, step = subset.ranges[axis]
                swept = swept_interval(begin, end, loops) if same(step, 1) else None
                if swept is None:
                    return False
                intervals.setdefault(axis, []).append(swept)
    if len(intervals) != 1:
        return False
    ((axis, spans),) = intervals.items()
    # Chain from 0: a span starting where the covered prefix ends extends it; rewrites of covered elements are spare.
    reach, pending = 0, list(spans)
    while not same(reach, desc.shape[axis]):
        following = next((span for span in pending if same(span[0], reach)), None)
        if following is None:
            return False
        pending.remove(following)
        reach = following[1] + 1
    return True


def dead_loop_fills(
    sdfg: SDFG, loop: LoopRegion, candidates: set[str]
) -> list[tuple[SDFGState, nodes.AccessNode, FillLibraryNode]]:
    """The fills outside ``loop`` of the ``candidates`` (filled transients ``loop`` accesses) whose values no
    iteration and nothing after the loop observes."""
    from dace.transformation.interstate.loop_to_map import (  # Avoid import loop
        carried_local_transients,
        control_flow_reads_outside,
    )

    inside = set(loop.all_control_flow_blocks())
    outside: dict[str, list[tuple[SDFGState, nodes.AccessNode]]] = {}
    for state in sdfg.states():
        if state in inside:
            continue
        for node in state.data_nodes():
            if node.data in candidates:
                outside.setdefault(node.data, []).append((state, node))
    # Every access outside the loop must be a lone constant fill: checked first, before the loop-wide analyses.
    filled: dict[str, list[tuple[SDFGState, nodes.AccessNode, FillLibraryNode]]] = {}
    for name in sorted(outside):
        if isinstance(sdfg.arrays[name], data.View):
            continue
        fills = []
        for state, node in outside[name]:
            producers = state.in_edges(node)
            if (
                state.out_degree(node) != 0
                or len(producers) != 1
                or not isinstance(producers[0].src, FillLibraryNode)
                or state.in_degree(producers[0].src) != 0
                or state.out_degree(producers[0].src) != 1
            ):
                break
            fills.append((state, node, producers[0].src))
        else:
            if fills:
                filled[name] = fills
    if not filled:
        return []
    _, written = loop.read_and_write_sets()
    flow_reads = control_flow_reads_outside(sdfg, loop)
    found = []
    for name, fills in filled.items():
        if name not in written or name in flow_reads:
            continue
        if tiles_every_iteration(loop, name, sdfg.arrays[name]) and not carried_local_transients(loop, {name}):
            found.extend(fills)
    return found


@properties.make_properties
class DeadFillElimination(ppl.Pass):
    """Remove every memset overwritten in full before any read: in its state, by the control flow after it, or in
    every iteration of a loop."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Drop the dead memsets of ``sdfg`` and its nested SDFGs.

        :returns: how many memsets were removed, or ``None`` if none were.
        """
        removed = 0
        for sd in sdfg.all_sdfgs_recursive():
            # Gathered once per SDFG: what a removal drops is a fill's write or a fill map's reads, so the reads stay
            # a superset.
            reads = Reads(sd)
            for state in sd.states():
                for fill in [n for n in state.nodes() if isinstance(n, FillLibraryNode)]:
                    dead = dead_fill(state, fill, reads)
                    if dead is None:
                        continue
                    zeroed, (axis, kept) = dead
                    if axis == -1:
                        drop_fill(state, fill, zeroed)
                    else:
                        written = state.in_edges(zeroed)[0].data
                        written.subset[axis] = kept
                        written.volume = written.subset.num_elements()
                    removed += 1
                for entry in [
                    n for n, scope in state.scope_dict().items() if isinstance(n, nodes.MapEntry) and scope is None
                ]:
                    dead = dead_part_of_fill_map(state, entry, reads)
                    if dead is not None:
                        narrow_fill_map(state, entry, *dead)
                        removed += 1
            # Only a filled transient can have a dead fill: the loop analyses run where one exists.
            filled = {
                edge.dst.data
                for state in sd.states()
                for node in state.nodes()
                if isinstance(node, FillLibraryNode)
                for edge in state.out_edges(node)
                if isinstance(edge.dst, nodes.AccessNode) and sd.arrays[edge.dst.data].transient
            }
            if not filled:
                continue
            for loop in [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                candidates = {
                    node.data
                    for block in loop.all_control_flow_blocks()
                    if isinstance(block, SDFGState)
                    for node in block.data_nodes()
                    if node.data in filled
                }
                if not candidates:
                    continue
                for state, node, fill in dead_loop_fills(sd, loop, candidates):
                    drop_fill(state, fill, node)
                    removed += 1
        return removed or None
