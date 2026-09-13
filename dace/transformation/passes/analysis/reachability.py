# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Control flow block reachability as bitsets over each region's strongly connected components.

The reach sets are quadratic in the number of blocks, so building them item by item is quadratic
work even when a consumer asks a handful of membership questions. Here each set is one integer
bitset, filled by one OR per edge of the component condensation, and its items are laid out only
when a consumer needs more than ``in`` or ``len``. That layout replays the insertion order of the
item-by-item construction on a snapshot of the graph taken when the analysis ran, so neither answer
moves if the graph is edited afterwards.
"""
import functools
import itertools
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from dace.ordered import OrderedSet
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import (AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState)

Replay = Callable[[], Iterable[ControlFlowBlock]]


def span(start: int, stop: int) -> int:
    """The bitset holding positions ``start`` up to, not including, ``stop``."""
    return ((1 << (stop - start)) - 1) << start


def bits_at(positions: Iterable[int]) -> int:
    """The bitset holding ``positions``, built in one pass instead of one big-integer OR per position."""
    buffer = bytearray()
    for position in positions:
        byte = position >> 3
        if byte >= len(buffer):
            buffer.extend(bytes(byte + 1 - len(buffer)))
        buffer[byte] |= 1 << (position & 7)
    return int.from_bytes(buffer, 'little')


class BlockNumbering:
    """Bit positions of the control flow blocks of one SDFG, nested SDFGs excluded.

    Numbered depth first, so the blocks below a region directly follow it: the region together with
    everything it contains is the span ``[positions[region], ends[region])``.
    """

    __slots__ = ('positions', 'ends', 'states')

    def __init__(self, sdfg: SDFG) -> None:
        self.positions: Dict[ControlFlowBlock, int] = {}
        self.ends: Dict[ControlFlowBlock, int] = {}
        state_positions: List[int] = []
        self.number(sdfg, state_positions)
        self.states: int = bits_at(state_positions)

    def number(self, region: AbstractControlFlowRegion, state_positions: List[int]) -> None:
        for block in region.nodes():
            position = len(self.positions)
            self.positions[block] = position
            if isinstance(block, SDFGState):
                state_positions.append(position)
            elif isinstance(block, AbstractControlFlowRegion):
                self.number(block, state_positions)
            self.ends[block] = len(self.positions)

    def expanded(self, block: ControlFlowBlock) -> int:
        """``block`` together with every block below it."""
        return span(self.positions[block], self.ends[block])

    def below(self, block: ControlFlowBlock) -> int:
        """Every block below ``block``, which is ``region.all_control_flow_blocks()`` for a region."""
        return span(self.positions[block] + 1, self.ends[block])


class ReachSet(OrderedSet):
    """An ``OrderedSet`` of blocks answering ``in`` and ``len`` from a bitset until anything else is asked.

    A deferred set carries a bitset over a :class:`BlockNumbering` and a replay yielding its items in
    insertion order, duplicates allowed. The first access to ``items`` or ``map`` lays the items out and
    drops the replay; from then on it is a plain ``OrderedSet``. A set built with the constructor starts
    out laid out.
    """

    def __init__(self, initial: Optional[Iterable[ControlFlowBlock]] = None) -> None:
        self.replay: Optional[Replay] = None
        self.bits = 0
        self.numbering: Optional[BlockNumbering] = None
        super().__init__(initial)

    @classmethod
    def deferred(cls, bits: int, numbering: BlockNumbering, replay: Replay) -> 'ReachSet':
        reach = cls()
        reach.bits = bits
        reach.numbering = numbering
        reach.replay = replay
        return reach

    def lay_out(self) -> None:
        items = list(dict.fromkeys(self.replay()))
        self.laid_out_items = items
        self.laid_out_map = dict(zip(items, range(len(items))))
        self.replay = None
        self.numbering = None

    @property
    def items(self) -> List[ControlFlowBlock]:
        if self.replay is not None:
            self.lay_out()
        return self.laid_out_items

    @items.setter
    def items(self, items: List[ControlFlowBlock]) -> None:
        self.laid_out_items = items

    @property
    def map(self) -> Dict[ControlFlowBlock, int]:
        if self.replay is not None:
            self.lay_out()
        return self.laid_out_map

    @map.setter
    def map(self, positions: Dict[ControlFlowBlock, int]) -> None:
        self.laid_out_map = positions

    def __contains__(self, key: object) -> bool:
        if self.replay is None:
            return key in self.laid_out_map
        position = self.numbering.positions.get(key)
        return position is not None and (self.bits >> position) & 1 == 1

    def __len__(self) -> int:
        if self.replay is None:
            return len(self.laid_out_items)
        return self.bits.bit_count()


def breadth_first_order(successors: Dict[ControlFlowBlock, List[ControlFlowBlock]],
                        source: ControlFlowBlock) -> Iterator[ControlFlowBlock]:
    """The blocks reachable from ``source`` over one or more edges, level by level.

    ``source`` itself is yielded only when a cycle leads back to it. Within a level, blocks come in
    the order the previous level's successor lists name them.
    """
    seen: Dict[ControlFlowBlock, None] = {}
    frontier = [source]
    while frontier:
        level = dict.fromkeys(itertools.chain.from_iterable(successors[block] for block in frontier))
        frontier = [block for block in level if block not in seen]
        seen.update(dict.fromkeys(frontier))
        yield from frontier


def condensed_reach(nodes: List[ControlFlowBlock], successors: Dict[ControlFlowBlock, List[ControlFlowBlock]],
                    numbering: BlockNumbering) -> Dict[ControlFlowBlock, int]:
    """For each node, the bitset of the blocks it reaches over one or more edges, regions expanded.

    Iterative Tarjan: a component is complete only after every component it reaches, so its bitset is
    the OR over its outgoing edges of the target's expansion and the target's own bitset. A component
    on a cycle also reaches each of its members.
    """
    order: Dict[ControlFlowBlock, int] = {}
    low: Dict[ControlFlowBlock, int] = {}
    stack: List[ControlFlowBlock] = []
    on_stack: Dict[ControlFlowBlock, None] = {}
    reach: Dict[ControlFlowBlock, int] = {}
    for root in nodes:
        if root in order:
            continue
        order[root] = low[root] = len(order)
        stack.append(root)
        on_stack[root] = None
        work = [(root, iter(successors[root]))]
        while work:
            node, children = work[-1]
            descended = False
            for child in children:
                if child not in order:
                    order[child] = low[child] = len(order)
                    stack.append(child)
                    on_stack[child] = None
                    work.append((child, iter(successors[child])))
                    descended = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], order[child])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] != order[node]:
                continue
            component: Dict[ControlFlowBlock, None] = {}
            while True:
                member = stack.pop()
                del on_stack[member]
                component[member] = None
                if member is node:
                    break
            bits = 0
            cyclic = len(component) > 1
            for member in component:
                for child in successors[member]:
                    if child in component:
                        cyclic = True
                    else:
                        bits |= numbering.expanded(child) | reach[child]
            if cyclic:
                for member in component:
                    bits |= numbering.expanded(member)
            for member in component:
                reach[member] = bits
    return reach


class BlockReachability:
    """Reach sets of every control flow block below one top-level SDFG, as :class:`ReachSet` s.

    Mirrors :class:`~dace.transformation.passes.analysis.analysis.ControlFlowBlockReachability`: the
    single-level set of a block holds what its own region's graph reaches from it (plus the region's
    blocks when the region is a loop), regions expanded; the full set adds the closure of every
    enclosing region up to the SDFG.
    """

    def __init__(self, top_sdfg: SDFG) -> None:
        self.nodes: Dict[AbstractControlFlowRegion, List[ControlFlowBlock]] = {}
        self.successors: Dict[AbstractControlFlowRegion, Dict[ControlFlowBlock, List[ControlFlowBlock]]] = {}
        self.parents: Dict[AbstractControlFlowRegion, Optional[AbstractControlFlowRegion]] = {}
        self.numberings: Dict[AbstractControlFlowRegion, BlockNumbering] = {}
        self.single_level: Dict[AbstractControlFlowRegion, Dict[ControlFlowBlock, ReachSet]] = {}
        self.closure_bits: Dict[AbstractControlFlowRegion, int] = {}
        self.closure_items: Dict[AbstractControlFlowRegion, List[ControlFlowBlock]] = {}
        self.below_items: Dict[AbstractControlFlowRegion, List[ControlFlowBlock]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            numbering = BlockNumbering(sdfg)
            for region in sdfg.all_control_flow_regions():
                graph = region.nx
                adjacency = graph.adj
                self.nodes[region] = region.nodes()
                self.successors[region] = {node: list(adjacency[node]) for node in graph}
                self.parents[region] = region.parent_graph
                self.numberings[region] = numbering
                self.single_level[region] = self.single_level_sets(region, numbering)

    def single_level_sets(self, region: AbstractControlFlowRegion,
                          numbering: BlockNumbering) -> Dict[ControlFlowBlock, ReachSet]:
        successors = self.successors[region]
        reach = condensed_reach(list(successors), successors, numbering)
        loop_blocks = bits_at(numbering.positions[block]
                              for block in self.nodes[region]) if isinstance(region, LoopRegion) else 0
        return {
            node:
            ReachSet.deferred(reach[node] | loop_blocks, numbering,
                              functools.partial(self.single_level_order, region, node))
            for node in successors
        }

    def within(self, region: AbstractControlFlowRegion) -> List[ControlFlowBlock]:
        """``region.all_control_flow_blocks()`` in its traversal order, built from the snapshot the same way.

        A list, not a ``set``: blocks hash by identity, so a set's order follows memory addresses and every
        reach set laid out from it would change order from run to run.
        """
        blocks = self.below_items.get(region)
        if blocks is None:
            blocks = list(itertools.chain.from_iterable(self.nodes[inner] for inner in self.regions_from(region)))
            self.below_items[region] = blocks
        return blocks

    def regions_from(self, region: AbstractControlFlowRegion) -> Iterator[AbstractControlFlowRegion]:
        """``region.all_control_flow_regions()`` over the snapshot."""
        yield region
        for block in self.nodes[region]:
            if isinstance(block, AbstractControlFlowRegion):
                yield from self.regions_from(block)

    def single_level_order(self, region: AbstractControlFlowRegion,
                           node: ControlFlowBlock) -> Iterator[ControlFlowBlock]:
        for reached in breadth_first_order(self.successors[region], node):
            yield reached
            if isinstance(reached, AbstractControlFlowRegion):
                yield from self.within(reached)
        if isinstance(region, LoopRegion):
            yield from self.nodes[region]

    def full_set(self, block: ControlFlowBlock, single: Iterable[ControlFlowBlock], sdfg: SDFG) -> ReachSet:
        """The reach set of ``block`` across its enclosing regions, given its single-level set ``single``.

        ``single`` is the block's :class:`ReachSet`, or an empty ``set`` for a block its region's graph
        does not hold (a conditional branch).
        """
        region = block.parent_graph
        numbering = self.numberings[region]
        bits = 0
        if isinstance(single, ReachSet):
            bits = single.bits
            if isinstance(region, LoopRegion):
                bits |= numbering.below(region)
        if region is not sdfg:
            bits |= self.closure(region)
        return ReachSet.deferred(bits, numbering, functools.partial(self.full_order, region, single, sdfg))

    def full_order(self, region: AbstractControlFlowRegion, single: Iterable[ControlFlowBlock],
                   sdfg: SDFG) -> Iterator[ControlFlowBlock]:
        for reached in single:
            if isinstance(reached, AbstractControlFlowRegion):
                yield from self.within(reached)
            yield reached
        if region is not sdfg:
            yield from self.closure_order(region)

    def closure(self, region: AbstractControlFlowRegion) -> int:
        """Bitset of everything that may run after any block inside ``region``, within its SDFG."""
        bits = self.closure_bits.get(region)
        if bits is not None:
            return bits
        numbering = self.numberings[region]
        bits = numbering.expanded(region) if isinstance(region, LoopRegion) else 0
        parent = self.parents[region]
        single = self.single_level[parent].get(region)
        if single is not None:
            bits |= single.bits
            if isinstance(parent, LoopRegion):
                for block in self.nodes[parent]:
                    if isinstance(block, ControlFlowRegion):
                        bits |= numbering.below(block)
        for pivot in self.enclosing(parent):
            bits |= self.closure(pivot)
        self.closure_bits[region] = bits
        return bits

    def enclosing(self, pivot: Optional[AbstractControlFlowRegion]) -> Iterator[AbstractControlFlowRegion]:
        """``pivot`` and its ancestors, stopping at the SDFG or at a region without blocks."""
        while pivot is not None and not isinstance(pivot, SDFG) and self.nodes[pivot]:
            yield pivot
            pivot = self.parents[pivot]

    def closure_order(self, region: AbstractControlFlowRegion) -> List[ControlFlowBlock]:
        items = self.closure_items.get(region)
        if items is None:
            items = list(dict.fromkeys(self.closure_sequence(region)))
            self.closure_items[region] = items
        return items

    def closure_sequence(self, region: AbstractControlFlowRegion) -> Iterator[ControlFlowBlock]:
        if isinstance(region, LoopRegion):
            yield from self.within(region)
            yield region
        parent = self.parents[region]
        single = self.single_level[parent].get(region)
        for reached in single if single is not None else ():
            if isinstance(reached, ControlFlowRegion):
                yield from self.within(reached)
            yield reached
        for pivot in self.enclosing(parent):
            yield from self.closure_order(pivot)


def states_only(reach: OrderedSet) -> Optional[OrderedSet]:
    """The states in ``reach``, in its order, or ``None`` when there are none."""
    if isinstance(reach, ReachSet) and reach.replay is not None:
        bits = reach.bits & reach.numbering.states
        if not bits:
            return None
        return ReachSet.deferred(bits, reach.numbering, functools.partial(filter_states, reach))
    states = [block for block in reach if isinstance(block, SDFGState)]
    return OrderedSet(states) if states else None


def filter_states(reach: Iterable[ControlFlowBlock]) -> Iterator[SDFGState]:
    return (block for block in reach if isinstance(block, SDFGState))
