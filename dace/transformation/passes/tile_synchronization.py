# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Place the thread-block barriers between the block-level tile nodes of the GPU kernels.

A ``WARP`` or ``BLOCK`` tile node spreads its tiles over the threads of a block (:func:`block_layout`), so a tile one
node writes is read by other threads in the next one unless both give each thread the same elements. Every write of a
tile leaves a token for the subset it wrote and the layout that wrote it, so the slots of a circular buffer each carry
their own; an access waits only for the tokens whose subsets it may overlap. Walking a kernel in program order, the first node that reads the
tile with another layout -- an ``mma`` reading its operands, a ``sum``, any other code -- is the latest point the token
can be waited for, and a barrier goes right before it. One barrier settles every pending token. Reads leave tokens too:
a node that overwrites a tile other threads still read waits for them. A loop body is walked a second time from the
tokens its end leaves, the loop variable moved back by one step in their subsets, which places the barrier its next
iteration needs before it overwrites a subset the previous one still reads.

A load is a token like any other write: the barrier before its first use is where an asynchronous copy is waited for.
A block-level tile node runs the pass when it expands (:meth:`~dace.libraries.tileops.nodes.tile_op.TileOp.expand`),
so the first node of a kernel to expand places the barriers of all of them; a kernel holding an expanded tile node is
left alone, and a barrier already in a kernel settles the tokens like a new one.
"""

import copy
import dataclasses
from typing import NamedTuple

from dace import SDFG, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes.barrier import Barrier, SyncScope
from dace.libraries.tileops.dispatch import TileGroup
from dace.libraries.tileops.expansions import TILE_THREADS_MAP, BlockLayout, block_layout
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.loop_analysis import get_loop_stride

#: The layout an access has: the tile layout of a block-level node, or ``None`` for an access by any thread.
Layout = BlockLayout | None

#: Where a barrier goes: before ``node`` in ``state``.
BarrierSite = tuple[SDFGState, nodes.Node]


class Token(NamedTuple):
    """An access no barrier has settled yet: ``subset`` of ``data`` (``None``: all of it) in ``layout``."""

    data: str
    subset: subsets.Subset | None
    layout: Layout

    def waits_for(self, other: "Token") -> bool:
        """Whether this access must wait for ``other``: another thread may touch an element both touch."""
        if self.data != other.data:
            return False
        if self.subset is None or other.subset is None:
            return True
        if subsets.intersects(self.subset, other.subset) is False:
            return False
        # The same elements in the same layout stay on the same threads
        return self.subset != other.subset or differ(self.layout, other.layout)

    def shifted(self, symbol: str, step: symbolic.SymbolicType | None) -> "Token":
        """The token as the next iteration of a loop over ``symbol`` sees it: its subset one ``step`` back."""
        if self.subset is None or symbol not in map(str, self.subset.free_symbols):
            return self
        if step is None:
            return self._replace(subset=None)
        subset = copy.deepcopy(self.subset)
        subset.replace({symbolic.symbol(symbol): symbolic.symbol(symbol) - step})
        return self._replace(subset=subset)


@dataclasses.dataclass
class Tokens:
    """The tile accesses no barrier has settled yet."""

    writes: list[Token] = dataclasses.field(default_factory=list)
    reads: list[Token] = dataclasses.field(default_factory=list)

    def copy(self) -> "Tokens":
        return Tokens(list(self.writes), list(self.reads))

    def merge(self, other: "Tokens") -> "Tokens":
        """The tokens either control-flow path may leave."""
        return Tokens(unique(self.writes + other.writes), unique(self.reads + other.reads))

    def shifted(self, loop: LoopRegion) -> "Tokens":
        step = get_loop_stride(loop)
        return Tokens(
            [token.shifted(loop.loop_variable, step) for token in self.writes],
            [token.shifted(loop.loop_variable, step) for token in self.reads],
        )

    def conflicts(self, reads: list[Token], writes: list[Token]) -> bool:
        """Whether the accesses wait for a token: another thread may have written an element read or written here,
        or may still read an element written here."""
        return any(access.waits_for(token) for access in (*reads, *writes) for token in self.writes) or any(
            access.waits_for(token) for access in writes for token in self.reads
        )

    def record(self, reads: list[Token], writes: list[Token]) -> None:
        self.reads = unique(self.reads + reads)
        self.writes = unique(self.writes + writes)


def unique(tokens: list[Token]) -> list[Token]:
    return list(dict.fromkeys(tokens))


def differ(first: Layout, second: Layout) -> bool:
    """Whether two accesses may touch an element on different threads."""
    return first is None or second is None or first != second


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertTileSync(ppl.Pass):
    """Insert a :class:`~dace.libraries.standard.nodes.barrier.Barrier` before the first node that needs a tile another thread wrote, or overwrites a
    tile another thread still reads, in every GPU kernel holding ``WARP`` or ``BLOCK`` tile nodes."""

    CATEGORY: str = "GPU"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Nodes

    def apply_pass(self, sdfg: SDFG, _) -> int | None:
        walk = Walk()
        for state in (state for sd in sdfg.all_sdfgs_recursive() for state in sd.states()):
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                    inner_nodes = [inner for inner, _ in kernel_nodes(state, node)]
                    groups = {inner.group for inner in inner_nodes if is_block_tile_node(inner)}
                    # A kernel with an expanded tile node was synchronized before that node expanded
                    if groups and not any(is_tile_expansion(inner) for inner in inner_nodes):
                        # Tiles that only warps share need only their warp to wait
                        walk.scope_of_kernel = SyncScope.WARP if groups == {TileGroup.WARP} else SyncScope.BLOCK
                        walk.scope(state, node, Tokens())
        #: The barriers this run inserted, with their states
        self.inserted: list[tuple[SDFGState, Barrier]] = []
        for (state, node), scope in walk.barriers.items():
            order = list(walk.order[state])
            position = order.index(node)
            self.inserted.append((state, insert_barrier(state, order[:position], order[position:], scope)))
        return len(self.inserted) or None


def is_tile_expansion(node: nodes.Node) -> bool:
    """A block-level tile node already expanded: the first one of a kernel to expand placed the barriers of all
    of them (:meth:`~dace.libraries.tileops.nodes.tile_op.TileOp.expand`)."""
    return isinstance(node, nodes.NestedSDFG) and node.sdfg.name.startswith(TILE_THREADS_MAP)


def is_block_tile_node(node: nodes.Node) -> bool:
    return isinstance(node, TileOp) and node.group is not TileGroup.THREAD


def kernel_nodes(state: SDFGState, kernel: nodes.MapEntry):
    """Every node in the kernel, nested SDFGs included."""
    for node in state.scope_subgraph(kernel, include_entry=False, include_exit=False).nodes():
        yield node, state
        if isinstance(node, nodes.NestedSDFG):
            yield from node.sdfg.all_nodes_recursive()


@dataclasses.dataclass
class Walk:
    """The program-order walk of the kernels, collecting the barriers they need and, per state, the kernel-level
    nodes that access tiles in the order the walk met them."""

    barriers: dict[BarrierSite, SyncScope] = dataclasses.field(default_factory=dict)
    order: dict[SDFGState, dict[nodes.Node, None]] = dataclasses.field(default_factory=dict)
    #: The threads the barriers of the kernel being walked wait for
    scope_of_kernel: SyncScope = SyncScope.BLOCK

    def scope(self, state: SDFGState, kernel: nodes.MapEntry, tokens: Tokens) -> Tokens:
        body = state.scope_subgraph(kernel, include_entry=False, include_exit=False)
        for node in sdutil.dfs_topological_sort(body):
            tokens = self.node(state, node, tokens)
        return tokens

    def region(self, region: AbstractControlFlowRegion, tokens: Tokens) -> Tokens:
        for block in sdutil.dfs_topological_sort(region):
            if isinstance(block, SDFGState):
                for node in sdutil.dfs_topological_sort(block):
                    tokens = self.node(block, node, tokens)
            elif isinstance(block, LoopRegion):
                # The second walk starts from what an iteration leaves for the next
                end = self.region(block, tokens.copy())
                tokens = tokens.merge(self.region(block, tokens.merge(end.shifted(block))))
            elif isinstance(block, ConditionalBlock):
                branches = [self.region(branch, tokens.copy()) for _, branch in block.branches]
                for branch in branches:
                    tokens = tokens.merge(branch)
            elif isinstance(block, ControlFlowRegion):
                tokens = self.region(block, tokens)
        return tokens

    def node(self, state: SDFGState, node: nodes.Node, tokens: Tokens) -> Tokens:
        if isinstance(node, Barrier):
            # A barrier already placed settles every token, so walking again places none twice
            self.order.setdefault(state, {})[kernel_level(state, node)] = None
            return Tokens()
        if not isinstance(node, nodes.CodeNode):
            return tokens
        reads, writes = accesses(state, node)
        if not (reads or writes or isinstance(node, nodes.NestedSDFG)):
            return tokens
        level = kernel_level(state, node)
        self.order.setdefault(state, {})[level] = None
        if tokens.conflicts(reads, writes):
            self.barriers[(state, level)] = self.scope_of_kernel
            tokens = Tokens()
        if isinstance(node, nodes.NestedSDFG):
            # Inside, the connectors may hold tiles written by any thread; what it leaves is any thread's
            inner = Tokens(
                [Token(name, None, None) for name in node.in_connectors if is_tile(node.sdfg.arrays[name], True)]
            )
            self.region(node.sdfg, inner)
        tokens.record(reads, writes)
        return tokens


def is_tile(desc: data.Data, connector: bool = False) -> bool:
    """Whether ``desc`` may be a tile the threads of a block share."""
    return (
        isinstance(desc, data.Array)
        and not isinstance(desc, data.View)
        and (desc.transient or connector)
        and desc.storage in (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.GPU_Shared)
    )


def accesses(state: SDFGState, node: nodes.CodeNode) -> tuple[list[Token], list[Token]]:
    """The tile subsets ``node`` reads and writes, each with the layout it accesses them in. A block-level node
    touches a subset of its output's size in its own layout through a lane-wise connector, anything else any thread."""
    block = is_block_tile_node(node)
    layout = block_layout(node, state) if block else None

    def of(edges, connector) -> list[Token]:
        found = []
        for edge in edges:
            name = edge.data.data
            if name is None or not is_tile(state.sdfg.arrays[name]):
                continue
            lane_wise = (
                block and node.reads_lane_wise(connector(edge)) and edge.data.subset.num_elements() == layout.elements
            )
            found.append(Token(name, edge.data.subset, layout if lane_wise else None))
        return found

    return of(state.in_edges(node), lambda edge: edge.dst_conn), of(state.out_edges(node), lambda edge: edge.src_conn)


def kernel_level(state: SDFGState, node: nodes.Node) -> nodes.Node:
    """``node``, or the outermost scope around it inside the kernel: a barrier inside a divergent scope deadlocks."""
    scope = state.entry_node(node)
    while scope is not None and scope.map.schedule != dtypes.ScheduleType.GPU_Device:
        node, scope = scope, state.entry_node(scope)
    return node


def insert_barrier(state: SDFGState, before: list[nodes.Node], after: list[nodes.Node], scope: SyncScope) -> Barrier:
    """A barrier of ``scope`` ordered after the nodes ``before`` and before the nodes ``after`` (kernel-level nodes of
    ``state`` in program order): a node merely independent of the one that waits must not slip past the barrier."""
    barrier = Barrier("tile_sync", scope)
    state.add_node(barrier)
    for node in dict.fromkeys(kernel_exit(state, node) for node in before):
        state.add_edge(node, None, barrier, None, Memlet())
    scope = state.entry_node(after[0])
    if scope is not None and state.in_degree(barrier) == 0:
        state.add_edge(scope, None, barrier, None, Memlet())
    for node in after:
        state.add_edge(barrier, None, node, None, Memlet())
    return barrier


def kernel_exit(state: SDFGState, node: nodes.Node) -> nodes.Node:
    """The exit of the outermost scope inside the kernel that ``node`` leaves through, or ``node`` itself."""
    level = kernel_level(state, node)
    return state.exit_node(level) if isinstance(level, nodes.EntryNode) else level
