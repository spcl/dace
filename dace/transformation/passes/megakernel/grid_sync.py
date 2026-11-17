# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Grid barriers of a persistent megakernel body: every kernel becomes a stage of its own, and a ``Sync(GRID)``
separates two stages wherever the later one may read what the earlier one wrote or write what it read or wrote.

The body is a hierarchical linear CFG (see ``discovery``), so one walk in program order sees every pair of stages that
can follow each other, plus the wrap-around of every loop: from the end of an iteration to the start of the next.
Dependences are tracked per container; only containers every thread sees (global memory) count, and none that both
stages touch only at the keys their threads own (``ownership``). Interstate edges and
conditions inside the body assign symbols and read only data the megakernel never writes (``discovery.uniform``), and
the body holds no views (``discovery.device_state``), so stage accesses by container name are all there is to order.
"""

from dace import SDFG, SDFGState, dtypes, properties
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.megakernel.discovery import (
    TARGETS,
    Target,
    bound_symbols,
    is_kernel,
    linear_chain,
    states_in,
)
from dace.transformation.passes.megakernel.ownership import OwnershipPlan, Signature, plan_ownership

#: Containers a stage reads and writes, those of them it touches off key, and the offsets of those it touches keyed
Accesses = tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[tuple[str, Signature]]]
NO_ACCESSES: Accesses = (frozenset(), frozenset(), frozenset(), frozenset())
#: A grid barrier to add to a region between two of its blocks (``None``: the region's start or end)
Barrier = tuple[ControlFlowRegion, ControlFlowBlock | None, ControlFlowBlock | None]


def split_into_stages(state: SDFGState, target: Target) -> list[SDFGState]:
    """Moves every kernel of ``state`` but the last into a state of its own before ``state``, in dataflow order."""
    kernels = [n for n in sdutil.dfs_topological_sort(state) if is_kernel(n, state, target)]
    stages = [
        xfh.state_fission(SubgraphView(state, state.scope_subgraph(k).nodes()), label=f"stage_{k.label}")
        for k in kernels[:-1]
    ]
    return stages + [state]


def stage_kernels(sdfg: SDFG, target: Target) -> list[tuple[SDFGState, nodes.MapEntry]]:
    """The stages of ``sdfg`` (split by :func:`split_into_stages`) and their kernels."""
    return [(s, n) for s in sdfg.states() for n in s.nodes() if is_kernel(n, s, target)]


def shared_accesses(block: ControlFlowBlock, target: Target, plan: OwnershipPlan | None = None) -> Accesses:
    """The containers that every thread sees which ``block`` reads and writes, those it touches off key (all of them
    outside owned stages), and the signatures of those it touches keyed."""
    reads, writes, off, keyed = set(), set(), set(), set()
    sdfg = block.sdfg
    shared_names = target.shared_containers(sdfg)
    for state in states_in(block):
        state_reads, state_writes = state.read_and_write_sets()
        touched = (state_reads | state_writes) & shared_names
        reads |= state_reads & shared_names
        writes |= state_writes & shared_names
        owner = plan.owners.get(state) if plan is not None else None
        if owner is None:
            off |= touched
        else:
            off |= touched - owner.keyed.keys()
            keyed |= {(name, sig) for name, sig in owner.keyed.items() if name in touched}
    return frozenset(reads), frozenset(writes), frozenset(off), frozenset(keyed)


def signatures_of(accesses: Accesses, name: str) -> set[Signature]:
    return {sig for keyed_name, sig in accesses[3] if keyed_name == name}


def conflict(before: Accesses, after: Accesses) -> bool:
    """Whether a thread of ``after`` may touch an element another thread touched in ``before``, one of them writing:
    a container both touch, which one of them touches off key, or both keyed but with different signatures."""
    shared = before[1] & (after[0] | after[1]) | before[0] & after[1]
    if shared & (before[2] | after[2]):
        return True
    return any(len(signatures_of(before, n) | signatures_of(after, n)) > 1 for n in shared)


def union(*accesses: Accesses) -> Accesses:
    return tuple(frozenset().union(*(a[i] for a in accesses)) for i in range(4))


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


class GridSyncPlacer:
    """Walks a megakernel body in program order and decides the grid barriers into ``barriers``; :meth:`insert` adds
    them. With ``sync_all``, every stage waits for all stages before it, dependent or not; with a ``plan``, accesses at
    the keys a thread owns need no barrier."""

    __slots__ = ("barriers", "before", "plan", "sync_all", "target")

    def __init__(self, target: Target, sync_all: bool = False, plan: OwnershipPlan | None = None) -> None:
        self.barriers: list[Barrier] = []
        self.before: set[ControlFlowBlock] = set()
        self.plan = plan
        self.sync_all = sync_all
        self.target = target

    def needs_sync(self, before: Accesses, after: Accesses) -> bool:
        if self.sync_all:
            return bool(before[0] or before[1])
        return conflict(before, after)

    def accesses(self, block: ControlFlowBlock) -> Accesses:
        return shared_accesses(block, self.target, self.plan)

    def add(self, region: ControlFlowRegion, after: ControlFlowBlock | None, before: ControlFlowBlock | None) -> None:
        self.barriers.append((region, after, before))
        if before is not None:
            self.before.add(before)

    def head_accesses(self, region: ControlFlowRegion) -> Accesses:
        """What ``region`` touches before its first grid barrier: what the previous iteration must be done with."""
        seen = NO_ACCESSES
        for block in linear_chain(region):
            if block in self.before or is_grid_sync(block):
                break
            seen = union(seen, self.accesses(block))
        return seen

    def walk(self, region: ControlFlowRegion, pending: Accesses) -> Accesses:
        """Decides the barriers of ``region``, entered with ``pending`` accesses not yet behind a barrier.

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
                accesses = self.accesses(block)
                if self.needs_sync(pending, accesses):
                    self.add(region, previous, block)
                    pending = accesses
                else:
                    pending = union(pending, accesses)
            previous = block
        return pending

    def loop(self, loop: LoopRegion, pending: Accesses) -> Accesses:
        end = self.walk(loop, pending)
        if self.needs_sync(end, self.head_accesses(loop)):
            self.add(loop, linear_chain(loop)[-1], None)
            end = NO_ACCESSES
        return union(pending, end)  # The loop may also run no iteration

    def insert(self) -> int:
        for region, after, before in self.barriers:
            add_grid_sync(region, after, before)
        return len(self.barriers)


def best_plan(sdfg: SDFG, target: Target, small: set[str]) -> tuple[OwnershipPlan | None, GridSyncPlacer]:
    """The ownership plan with the fewest grid barriers, and its placer: no plan (every stage spread over the whole
    grid), then key dimensions from the most to one; a plan must save a barrier to win."""
    stages = stage_kernels(sdfg, target)
    shared = target.shared_containers(sdfg)
    frozen = bound_symbols(sdfg)
    rank = max((len(e.map.params) for s, e in stages), default=0)
    best = None
    for plan in [None] + [plan_ownership(stages, k, shared, small, frozen) for k in range(rank, 0, -1)]:
        placer = GridSyncPlacer(target, plan=plan)
        placer.walk(sdfg, NO_ACCESSES)
        if best is None or len(placer.barriers) < len(best[1].barriers):
            best = plan, placer
    return best


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGridSyncs(ppl.Pass):
    """Splits a megakernel body (a hierarchical linear CFG of kernels) into one stage per kernel and adds the grid
    barriers between dependent stages, or between all stages with ``sync_all``. With ``ownership``, it picks the
    thread ownership of the stages (``ownership.py``) that needs the fewest barriers; ``plan`` keeps it for the
    lowering. Returns the number of barriers added."""

    CATEGORY: str = "Optimization Preparation"

    device = properties.EnumProperty(
        dtype=dtypes.DeviceType, default=dtypes.DeviceType.GPU, desc="The device whose kernels are the stages"
    )
    sync_all = properties.Property(dtype=bool, default=True, desc="Separate every two stages, dependent or not")
    ownership = properties.Property(
        dtype=bool, default=True, desc="Schedule stages by thread ownership where that saves barriers"
    )
    small_symbols = properties.SetProperty(
        element_type=str, default=set(), desc="Symbols whose extents stay a loop in every thread"
    )

    def __init__(
        self,
        device: dtypes.DeviceType = dtypes.DeviceType.GPU,
        sync_all: bool = True,
        ownership: bool = True,
        small_symbols: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.sync_all = sync_all
        self.ownership = ownership
        self.small_symbols = set(small_symbols or ())
        self._plan: OwnershipPlan | None = None

    @property
    def plan(self) -> OwnershipPlan | None:
        """The ownership plan the last run chose, if any."""
        return self._plan

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        target = TARGETS[self.device]
        for state in list(sdfg.states()):
            if sum(is_kernel(n, state, target) for n in state.nodes()) > 1:
                split_into_stages(state, target)
        if self.ownership and not self.sync_all:
            self._plan, placer = best_plan(sdfg, target, self.small_symbols)
        else:
            placer = GridSyncPlacer(target, self.sync_all)
            placer.walk(sdfg, NO_ACCESSES)
        return placer.insert() or None
