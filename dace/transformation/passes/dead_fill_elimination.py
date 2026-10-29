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

A top-level map writing a transient element for element (``np.zeros`` lowered as a map) shrinks to what a copy in
its state leaves when the copy overwrites all of it but a prefix or suffix of one axis before anything reads it
(``sink = np.zeros(n + 1); sink[0:n] = c`` zeroes row ``n`` only); it goes when the copy leaves nothing.

Anything else (a partial or indirect write, a conditional one without ``else``, a read in between, a
nested SDFG writing it) keeps the memset.
"""

from collections.abc import Iterator
from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    ConditionalBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis

#: Axes a write must fix to one index instead of covering: ``{axis: index}`` for the loop around it.
Fixed = dict[int, symbolic.SymbolicType]


def covers_every_element(
    state: SDFGState, written: nodes.AccessNode, desc: data.Array, fixed: Fixed | None = None
) -> bool:
    """Whether ``written``'s only writer is a map storing each element of ``desc`` exactly once, unread.

    With ``fixed``, each axis it names holds that one index instead, and the rest is covered once.
    """
    fixed = fixed or {}
    in_edges = state.in_edges(written)
    if len(in_edges) == 1 and isinstance(in_edges[0].src, nodes.AccessNode):
        return copy_covers(state, in_edges[0], desc, fixed)
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
    for axis, ((begin, end, step), extent) in enumerate(zip(inner[0].data.subset.ranges, desc.shape)):
        if axis in fixed:
            if begin != end or symbolic.simplify(begin - fixed[axis]) != 0:
                return False
            continue
        param = str(begin)
        if begin != end or param not in ranges or param in seen:
            return False
        seen.add(param)
        low, high, stride = ranges[param]
        if symbolic.simplify(low) != 0 or stride != 1 or symbolic.simplify(high + 1 - extent) != 0:
            return False
    # One parameter per axis and no other: a parameter left over repeats the stores, so it is still a map whose
    # iterations outnumber the elements -- kept, not reasoned about.
    if len(seen) != len(desc.shape) - len(fixed) or len(entry.map.params) != len(seen):
        return False
    return all(e.data.data != written.data for e in state.in_edges(entry))


def copy_covers(state: SDFGState, edge, desc: data.Array, fixed: Fixed) -> bool:
    """Whether the copy ``edge`` writes every element of ``desc`` (``fixed`` axes at that one index)."""
    if edge.data.dynamic or edge.data.wcr is not None or edge.src.data == edge.dst.data:
        return False
    subset = edge.data.get_dst_subset(edge, state)
    if subset is None or len(subset.ranges) != len(desc.shape):
        return False
    for axis, ((begin, end, step), extent) in enumerate(zip(subset.ranges, desc.shape)):
        if axis in fixed:
            if begin != end or symbolic.simplify(begin - fixed[axis]) != 0:
                return False
        elif symbolic.simplify(begin) != 0 or step != 1 or symbolic.simplify(end + 1 - extent) != 0:
            return False
    return True


def overwritten_in_state(state: SDFGState, name: str, desc: data.Array, fixed: Fixed | None = None) -> bool:
    """Whether a map in ``state`` overwrites ``name`` (``fixed`` axes at one index) before the state reads it."""
    accesses = [n for n in state.data_nodes() if n.data == name]
    for written in accesses:
        if not covers_every_element(state, written, desc, fixed):
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in accesses):
            return True
    return False


def dead_fill(state: SDFGState, fill: FillLibraryNode) -> nodes.AccessNode | None:
    """The access node ``fill`` writes, when the fill is dead; ``None`` otherwise."""
    if any(not e.data.is_empty() for e in state.in_edges(fill)):
        return None
    out_edges = state.out_edges(fill)
    if len(out_edges) != 1 or not isinstance(out_edges[0].dst, nodes.AccessNode):
        return None
    zeroed = out_edges[0].dst
    sdfg = state.sdfg
    desc = sdfg.arrays[zeroed.data]
    if not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, data.View):
        return None
    subset = out_edges[0].data.subset
    if subset is None or symbolic.simplify(subset.num_elements() - desc.total_size) != 0:
        return None
    if any(not e.data.is_empty() for e in state.out_edges(zeroed)) or len(state.in_edges(zeroed)) != 1:
        return None
    others = [n for n in state.data_nodes() if n.data == zeroed.data and n is not zeroed]
    reachable = set(state.bfs_nodes(zeroed))
    for written in others:
        if written not in reachable or not covers_every_element(state, written, desc):
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in others):
            return zeroed
    if not others and overwritten_after(state, zeroed.data, desc):
        return zeroed
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


def dead_part_of_fill_map(state: SDFGState, entry: nodes.MapEntry) -> tuple[nodes.AccessNode, tuple] | None:
    """The array a top-level map writes element for element, and what of it a copy in the same state overwrites before
    anything reads it: the :func:`copy_leaves` of that copy; ``None`` when nothing of the map's work is dead."""
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
        if left is None:
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in others):
            return zeroed, left
    return None


def narrow_fill_map(state: SDFGState, entry: nodes.MapEntry, zeroed: nodes.AccessNode, left: tuple) -> None:
    """Restrict the map writing ``zeroed`` to the ``left`` part of one axis, or drop the map when nothing is left."""
    axis, kept = left
    if axis == -1:
        scope = state.scope_subgraph(entry).nodes()
        sources = {e.src for e in state.in_edges(entry)}
        state.remove_nodes_from([*scope, zeroed])
        state.remove_nodes_from([n for n in sources if isinstance(n, nodes.AccessNode) and state.degree(n) == 0])
        return
    exit_node = state.exit_node(entry)
    inner = next(iter(state.in_edges(exit_node)))
    param = str(inner.data.subset.ranges[axis][0])
    entry.map.range.ranges[entry.map.params.index(param)] = kept
    outer = state.out_edges(exit_node)[0]
    outer.data.subset.ranges[axis] = kept


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


def unconditional_successor(graph: ControlFlowRegion, block: ControlFlowBlock) -> ControlFlowBlock | None:
    """The one block ``block`` always passes to in ``graph``, if its only out edge reads nothing it is not told."""
    out_edges = graph.out_edges(block)
    if len(out_edges) != 1 or not out_edges[0].data.is_unconditional():
        return None
    return out_edges[0].dst


def overwritten_by(block: ControlFlowBlock, name: str, desc: data.Array, fixed: Fixed) -> bool:
    """Whether executing ``block`` overwrites ``name`` whole (``fixed`` axes at one index) before it reads it."""
    if isinstance(block, SDFGState):
        return overwritten_in_state(block, name, desc, fixed)
    if isinstance(block, ConditionalBlock):
        if name in control_reads(block) or not any(condition is None for condition, _ in block.branches):
            return False
        return all(
            overwritten_from(branch, branch.start_block, name, desc, fixed) is True for _, branch in block.branches
        )
    if isinstance(block, LoopRegion):
        return not fixed and overwritten_by_loop(block, name, desc)
    if isinstance(block, ControlFlowRegion) and block.nodes():
        return overwritten_from(block, block.start_block, name, desc, fixed) is True
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
        block = unconditional_successor(loop, block)
    return False


def overwritten_from(
    graph: ControlFlowRegion, block: ControlFlowBlock | None, name: str, desc: data.Array, fixed: Fixed
) -> bool | None:
    """Follow ``graph``'s unconditional chain from ``block`` to the first block touching ``name``.

    :returns: whether that block overwrites ``name`` before reading it; ``None`` if the chain ends or
              branches before any block touches it.
    """
    while block is not None:
        if touches(block, name):
            return overwritten_by(block, name, desc, fixed)
        block = unconditional_successor(graph, block)
    return None


def overwritten_after(state: SDFGState, name: str, desc: data.Array) -> bool:
    """Whether the control flow after ``state`` overwrites ``name`` whole before anything reads it."""
    block: ControlFlowBlock = state
    while True:
        graph = block.parent_graph
        following = unconditional_successor(graph, block)
        while following is not None:
            if touches(following, name):
                return overwritten_by(following, name, desc, {})
            block, following = following, unconditional_successor(graph, following)
        # The chain ran out untouched: at the end of a plain region or a branch, control passes on after it.
        if graph.out_degree(block) != 0 or isinstance(graph, (LoopRegion, SDFG)):
            return False
        block = graph.parent_graph if isinstance(graph.parent_graph, ConditionalBlock) else graph


@properties.make_properties
class DeadFillElimination(ppl.Pass):
    """Remove every memset a map overwrites in full, unread, in the same state."""

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
            for state in sd.states():
                for fill in [n for n in state.nodes() if isinstance(n, FillLibraryNode)]:
                    zeroed = dead_fill(state, fill)
                    if zeroed is None:
                        continue
                    state.remove_node(fill)
                    state.remove_node(zeroed)
                    removed += 1
                for entry in [
                    n for n, scope in state.scope_dict().items() if isinstance(n, nodes.MapEntry) and scope is None
                ]:
                    dead = dead_part_of_fill_map(state, entry)
                    if dead is not None:
                        narrow_fill_map(state, entry, *dead)
                        removed += 1
        return removed or None
