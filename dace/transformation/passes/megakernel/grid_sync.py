# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Grid barriers of a persistent megakernel body: every kernel becomes a stage of its own, and a ``Sync(GRID)``
separates two stages wherever the later one may read what the earlier one wrote or write what it read or wrote.

The body is a hierarchical linear CFG (see ``discovery``), so one walk in program order sees every pair of stages that
can follow each other, plus the wrap-around of every loop: from the end of an iteration to the start of the next.
Dependences are tracked per container; only containers every thread sees (global memory) count.
"""

from dace import SDFG, SDFGState, dtypes, properties
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.megakernel.discovery import TARGETS, Target, is_kernel, linear_chain, states_in

#: Containers a stage reads and writes
Accesses = tuple[frozenset[str], frozenset[str]]
NO_ACCESSES: Accesses = (frozenset(), frozenset())


def split_into_stages(state: SDFGState, target: Target) -> list[SDFGState]:
    """Moves every kernel of ``state`` but the last into a state of its own before ``state``, in dataflow order."""
    kernels = [n for n in sdutil.dfs_topological_sort(state) if is_kernel(n, state, target)]
    stages = [
        xfh.state_fission(SubgraphView(state, state.scope_subgraph(k).nodes()), label=f"stage_{k.label}")
        for k in kernels[:-1]
    ]
    return stages + [state]


def shared_accesses(block: ControlFlowBlock, target: Target) -> Accesses:
    """The containers that every thread sees which ``block`` reads and writes."""
    reads, writes = set(), set()
    for state in states_in(block):
        state_reads, state_writes = state.read_and_write_sets()
        reads |= state_reads
        writes |= state_writes
    sdfg = block.sdfg
    shared = {name for name in reads | writes if target.shares(sdfg.arrays[name])}
    return frozenset(reads & shared), frozenset(writes & shared)


def conflict(before: Accesses, after: Accesses) -> bool:
    return bool(before[1] & (after[0] | after[1]) or before[0] & after[1])


def union(*accesses: Accesses) -> Accesses:
    return frozenset().union(*(a[0] for a in accesses)), frozenset().union(*(a[1] for a in accesses))


def is_grid_sync(block: ControlFlowBlock) -> bool:
    return isinstance(block, SDFGState) and any(
        isinstance(n, Sync) and n.scope == SyncScope.GRID for n in block.nodes()
    )


def add_grid_sync(region: ControlFlowRegion, after: ControlFlowBlock | None, before: ControlFlowBlock | None) -> None:
    """Adds a grid barrier state to ``region`` between ``after`` and ``before`` (``None``: the region's start or
    end)."""
    if after is None:
        state = region.add_state_before(before, "grid_sync", is_start_block=True)
    else:
        state = region.add_state_after(after, "grid_sync")
    state.add_node(Sync("grid_sync", scope=SyncScope.GRID))


def head_accesses(region: ControlFlowRegion, target: Target) -> Accesses:
    """What ``region`` touches before its first grid barrier: what the previous iteration must be done with."""
    seen = NO_ACCESSES
    for block in linear_chain(region):
        if is_grid_sync(block):
            break
        seen = union(seen, shared_accesses(block, target))
    return seen


class GridSyncPlacer:
    """Walks a megakernel body in program order and adds the grid barriers; counts them in ``added``."""

    __slots__ = ("added", "target")

    def __init__(self, target: Target) -> None:
        self.added = 0
        self.target = target

    def walk(self, region: ControlFlowRegion, pending: Accesses) -> Accesses:
        """Places the barriers of ``region``, entered with ``pending`` accesses not yet behind a barrier.

        :return: The accesses not behind a barrier when ``region`` ends.
        """
        previous = None
        for block in linear_chain(region):
            if isinstance(block, LoopRegion):
                pending = self.loop(block, pending)
            elif isinstance(block, ConditionalBlock):
                ends = [self.walk(branch, pending) for cond, branch in block.branches]
                otherwise = [] if any(cond is None for cond, branch in block.branches) else [pending]
                pending = union(*ends, *otherwise)
            else:
                accesses = shared_accesses(block, self.target)
                if conflict(pending, accesses):
                    add_grid_sync(region, previous, block)
                    self.added += 1
                    pending = accesses
                else:
                    pending = union(pending, accesses)
            previous = block
        return pending

    def loop(self, loop: LoopRegion, pending: Accesses) -> Accesses:
        end = self.walk(loop, pending)
        if conflict(end, head_accesses(loop, self.target)):
            add_grid_sync(loop, linear_chain(loop)[-1], None)
            self.added += 1
            end = NO_ACCESSES
        return union(pending, end)  # The loop may also run no iteration


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGridSyncs(ppl.Pass):
    """Splits a megakernel body (a hierarchical linear CFG of kernels) into one stage per kernel and adds the grid
    barriers between dependent stages. Returns the number of barriers added."""

    CATEGORY: str = "Optimization Preparation"

    device = properties.EnumProperty(
        dtype=dtypes.DeviceType, default=dtypes.DeviceType.GPU, desc="The device whose kernels are the stages"
    )

    def __init__(self, device: dtypes.DeviceType = dtypes.DeviceType.GPU) -> None:
        super().__init__()
        self.device = device

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        target = TARGETS[self.device]
        for state in list(sdfg.states()):
            if sum(is_kernel(n, state, target) for n in state.nodes()) > 1:
                split_into_stages(state, target)
        placer = GridSyncPlacer(target)
        placer.walk(sdfg, NO_ACCESSES)
        return placer.added or None
