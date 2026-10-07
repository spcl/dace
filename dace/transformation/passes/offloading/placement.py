# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Place every container on the side that runs it, and copy where a container changes sides.

One walk over the structured control flow carries ``where``, the side each container is on, from block to block.
A block wants some containers on one side; the ones that are elsewhere are copied right before it, and its
accesses are renamed to the twin on the side they run. Regions end where they began, so a copy is never
placed on a path that does not need it:

* a loop is entered where its body first uses each container, and every path back to the header (the end of the
  body, ``continue``) and out of it (``break``) restores that location, so the loop's copy-in runs once;
* the arms of a conditional meet where the first arm leaves each container, or where it was before the
  conditional if no ``else`` makes the arms exhaustive; every other arm copies at its end;
* a ``return`` and the end of the SDFG restore the containers of the signature to the side they came in on.
"""

from ordered_set import OrderedSet

from dace import dtypes
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
)

import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace.transformation.passes.offloading import twins
from dace.transformation.passes.offloading.locations import Wants

#: Container name -> True when it is on the device, False on the host.
Where = dict[str, bool]


class Placement:
    """Place the containers of one SDFG, given what each of its states wants; run it with :meth:`apply`."""

    __slots__ = ("fixed_storage", "initial", "placed_on_gpu", "plan", "sdfg", "touched_by", "wants")

    def __init__(self, sdfg: SDFG, wants: dict[SDFGState, Wants]) -> None:
        self.sdfg = sdfg
        self.wants = wants
        # A view is placed with the container it aliases; a structure or container array is not placed.
        self.initial: Where = {
            name: helpers.is_array_stored_on_GPU(sdfg, name)
            for name, desc in sdfg.arrays.items()
            if not desc.transient and helpers.is_array(name, sdfg)
        }
        # A shared-memory or register container keeps the scope-local storage it was given.
        self.fixed_storage = OrderedSet(
            name
            for name, desc in sdfg.arrays.items()
            if desc.storage in (dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register)
        )
        #: Transients this placement put in device memory.
        self.placed_on_gpu: OrderedSet[str] = OrderedSet()
        self.plan = twins.CopyPlan(sdfg)
        self.touched_by: dict[SDFGState, OrderedSet[str]] = {}

    def apply(self) -> OrderedSet[str]:
        """Rename the accesses and insert the copies; return the transients put in device memory."""
        where, last = self.walk(self.sdfg, dict(self.initial), {})
        if where is not None:
            self.restore(where, self.initial, self.sdfg, last, before=False)
        self.plan.insert()
        twins.place_views(self.sdfg)
        return self.placed_on_gpu

    def walk(self, region: ControlFlowRegion, where: Where, loop_entry: Where) -> tuple[Where | None, ControlFlowBlock]:
        """Walk ``region`` from ``where``; return where it ends (None if it leaves through ``return``, ``break`` or
        ``continue``) and its last block. ``loop_entry`` is where the innermost loop around it is entered."""
        blocks = list(region.bfs_nodes())
        current: Where | None = where
        for index, block in enumerate(blocks):
            assert current is not None
            reads = self.edge_reads(region, block)
            if reads:  # an interstate edge is host code: copy right after the block it leaves
                self.move(current, dict.fromkeys(reads, False), region, blocks[index - 1], before=False)
                twins.rename_interstate_reads(self.sdfg, region.in_edges(block))
            if isinstance(block, SDFGState):
                self.visit_state(region, block, blocks[:index], current)
            elif isinstance(block, ReturnBlock):
                self.restore(current, self.initial, region, block, before=True)
                return None, block
            elif isinstance(block, (BreakBlock, ContinueBlock)):
                self.restore(current, loop_entry, region, block, before=True)
                return None, block
            elif isinstance(block, ConditionalBlock):
                current = self.visit_conditional(region, block, current, loop_entry)
            elif isinstance(block, LoopRegion):
                self.visit_loop(region, block, blocks[:index], current)
            elif isinstance(block, ControlFlowRegion):
                current, _ = self.walk(block, current, loop_entry)
            if current is None:
                return None, block
        return current, blocks[-1]

    def visit_state(
        self, region: ControlFlowRegion, state: SDFGState, above: list[ControlFlowBlock], where: Where
    ) -> None:
        wants = self.wants[state]
        targets: Where = {**dict.fromkeys(wants.gpu, True), **dict.fromkeys(wants.cpu, False)}
        self.place_copy_destinations(state, where, targets)
        self.move(where, targets, region, state, before=True, hoist_over=above)
        twins.rename_in_state(self.sdfg, state, where)

    def place_copy_destinations(self, state: SDFGState, where: Where, targets: Where) -> None:
        """A top-level container-to-container copy writes a container with no location yet on its source's side."""
        scopes = state.scope_dict()
        for edge in state.edges():
            src, dst = edge.src, edge.dst
            if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)) or edge.data.is_empty():
                continue
            if scopes[src] is not None or scopes[dst] is not None or dst.data in where or dst.data in targets:
                continue
            side = targets.get(src.data, where.get(src.data))
            if side is not None and helpers.is_array(dst.data, self.sdfg):
                targets[dst.data] = side

    def visit_loop(
        self, region: ControlFlowRegion, loop: LoopRegion, above: list[ControlFlowBlock], where: Where
    ) -> None:
        entry: Where = {}
        self.first_uses(loop, entry)
        self.move(where, entry, region, loop, before=True, hoist_over=above)
        twins.rename_header_reads(self.sdfg, loop, self.header_reads(loop))
        body_end, last = self.walk(loop, dict(where), dict(where))
        if body_end is not None:
            self.restore(body_end, where, loop, last, before=False)

    def visit_conditional(
        self, region: ControlFlowRegion, block: ConditionalBlock, where: Where, loop_entry: Where
    ) -> Where | None:
        self.move(where, dict.fromkeys(self.header_reads(block), False), region, block, before=True)
        twins.rename_header_reads(self.sdfg, block, self.header_reads(block))
        exits = []
        for _, arm in block.branches:
            end, last = self.walk(arm, dict(where), loop_entry)
            if end is not None:
                exits.append((arm, end, last))
        exhaustive = block.branches[-1][0] is None
        if exhaustive and not exits:
            return None
        meeting: Where = dict(exits[0][1]) if exhaustive else dict(where)
        for _, end, _ in exits:
            for name, on_gpu in end.items():
                meeting.setdefault(name, on_gpu)
        for arm, end, last in exits:
            self.restore(end, meeting, arm, last, before=False)
        return meeting

    def first_uses(self, region: ControlFlowRegion, entry: Where) -> None:
        """Where ``region`` first uses each container, in program order; the first use wins."""
        for block in region.bfs_nodes():
            for name in self.edge_reads(region, block):
                entry.setdefault(name, False)
            if isinstance(block, SDFGState):
                wants = self.wants[block]
                for name in wants.gpu:
                    entry.setdefault(name, True)
                for name in wants.cpu:
                    entry.setdefault(name, False)
                continue
            for name in self.header_reads(block):
                entry.setdefault(name, False)
            if isinstance(block, ConditionalBlock):
                for _, arm in block.branches:
                    self.first_uses(arm, entry)
            elif isinstance(block, ControlFlowRegion):
                self.first_uses(block, entry)

    def move(
        self,
        where: Where,
        targets: Where,
        region: ControlFlowRegion,
        block: ControlFlowBlock,
        before: bool,
        hoist_over: list[ControlFlowBlock] | None = None,
    ) -> None:
        """Put each container of ``targets`` where it is wanted, copying the ones that are elsewhere next to ``block``.

        A container with no location yet is placed on that side without a copy: a transient lives on the side of
        its first use. A copy to the device moves above the ``hoist_over`` states that do not touch the container:
        until it is needed it does not care where it is, and a run of kernels stays free of host states.
        """
        grouped: dict[tuple[ControlFlowBlock, bool], OrderedSet[str]] = {}
        for name, on_gpu in targets.items():
            current = where.get(name)
            where[name] = on_gpu
            if current is None:
                self.place_transient(name, on_gpu)
            elif current != on_gpu:
                anchor = block
                if on_gpu and hoist_over:
                    anchor = self.hoisted_anchor(region, hoist_over, block, name)
                grouped.setdefault((anchor, on_gpu), OrderedSet()).add(name)
        for (anchor, to_gpu), names in grouped.items():
            self.plan.add(region, anchor, before, names, to_gpu)

    def hoisted_anchor(
        self, region: ControlFlowRegion, above: list[ControlFlowBlock], block: ControlFlowBlock, name: str
    ) -> ControlFlowBlock:
        """The earliest block of ``above`` + ``block`` that ``name`` can be copied in front of."""
        anchor = block
        for previous in reversed(above):
            if (
                not isinstance(previous, SDFGState)
                or name in self.touched(previous)
                or name in self.edge_reads(region, anchor)
            ):
                break
            anchor = previous
        return anchor

    def touched(self, state: SDFGState) -> OrderedSet[str]:
        """The containers ``state`` wants or accesses, as its states were before any renaming."""
        if state not in self.touched_by:
            wants = self.wants[state]
            self.touched_by[state] = wants.gpu | wants.cpu | OrderedSet(node.data for node in state.data_nodes())
        return self.touched_by[state]

    def restore(
        self, where: Where, reference: Where, region: ControlFlowRegion, block: ControlFlowBlock, before: bool
    ) -> None:
        """Copy the containers that ``where`` has somewhere else than ``reference`` back, next to ``block``."""
        targets = {name: on_gpu for name, on_gpu in reference.items() if name in where}
        self.move(where, targets, region, block, before)

    def place_transient(self, name: str, on_gpu: bool) -> None:
        desc = self.sdfg.arrays[name]
        if not desc.transient or name in self.fixed_storage:
            return
        desc.storage = dtypes.StorageType.GPU_Global if on_gpu else dtypes.StorageType.Default
        self.fixed_storage.add(name)
        if on_gpu:
            self.placed_on_gpu.add(name)

    def edge_reads(self, region: ControlFlowRegion, block: ControlFlowBlock) -> OrderedSet[str]:
        """The arrays the interstate edges into ``block`` read."""
        return OrderedSet(
            name
            for edge in region.in_edges(block)
            for name in edge.data.used_arrays(self.sdfg.arrays)
            if helpers.is_array(name, self.sdfg)
        )

    def header_reads(self, block: ControlFlowBlock) -> OrderedSet[str]:
        """The containers a loop's or a conditional's header reads on the host."""
        if not isinstance(block, (ConditionalBlock, LoopRegion)):
            return OrderedSet()
        return OrderedSet(memlet.data for memlet in block.get_meta_read_memlets() if memlet.data in self.sdfg.arrays)
