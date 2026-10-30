# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Megakernel preparation: turns host loops over kernels into control flow, moves the host code the kernels only take
values from ahead of them, and gathers the kernels of consecutive states into one state.

A host map whose body calls kernels (a loop over blocks of columns) becomes a loop region of the CFG
(``MapToForLoop``, then inlining), so one megakernel can run the loop in every thread. Between and inside such loops,
host code often only computes grid-uniform values: tasklets that copy a host scalar or array element into a scalar,
and interstate assignments that read loop bounds from host arrays. The scalars become symbols
(``ScalarToSymbolPromotion``), ``SymbolSSA`` gives each assignment a symbol of its own, ``SiftAssignmentsUp`` moves the
assignments ahead of the loops and states, and ``StateFusionExtended`` fuses the kernel states.
"""

from dace import SDFG, SDFGState, dtypes, properties
from dace import data as dt
from dace.sdfg import InterstateEdge, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate import InlineMultistateSDFG, StateFusionExtended
from dace.transformation.passes.megakernel.discovery import TARGETS, Target, is_kernel, linear_chain, outside_kernels
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.sift_assignments_up import SiftAssignmentsUp, effects, rhs_names
from dace.transformation.passes.symbol_ssa import SymbolSSA


def host_map_over_kernels(sdfg: SDFG, target: Target) -> tuple[SDFGState, nodes.MapEntry] | None:
    """A one-dimensional top-level host map of ``sdfg`` whose body is a nested SDFG calling ``target`` kernels."""
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry) or state.entry_node(node) is not None:
                continue
            if node.map.schedule == target.kernel_schedule or len(node.map.params) != 1:
                continue
            bodies = [n for n in state.scope_children()[node] if isinstance(n, nodes.NestedSDFG)]
            if any(is_kernel(k, s, target) for body in bodies for s in body.sdfg.states() for k in s.nodes()):
                return state, node
    return None


def inline(state: SDFGState, node: nodes.NestedSDFG) -> bool:
    if not InlineMultistateSDFG.can_be_applied_to(state.sdfg, nested_sdfg=node):
        return False
    InlineMultistateSDFG.apply_to(state.sdfg, verify=False, save=False, nested_sdfg=node)
    return True


def lower_host_maps(sdfg: SDFG, target: Target) -> int:
    """Turns every host map of ``sdfg`` over a body of kernels into a loop region of the CFG, inlining the nested SDFGs
    ``MapToForLoop`` leaves (the loop's own and the body's). Returns the number of maps lowered."""
    lowered = 0
    while (found := host_map_over_kernels(sdfg, target)) is not None:
        state, entry = found
        body = next(n for n in state.scope_children()[entry] if isinstance(n, nodes.NestedSDFG))
        loop_node, loop_state = MapToForLoop.apply_to(sdfg, verify=False, save=False, map_entry=entry)
        if inline(loop_state, body):
            inline(state, loop_node)
        lowered += 1
    return lowered


def host_scalars(sdfg: SDFG, target: Target) -> set[str]:
    """The transient host scalars of ``sdfg`` that tasklets outside kernels write and that no kernel or nested SDFG
    reads as data: interstate edges, conditions and host tasklets read them."""
    written, passed_in = set(), set()
    for state in sdfg.states():
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if not (isinstance(desc, dt.Scalar) and desc.transient and desc.storage not in target.top_level_storage):
                continue
            if outside_kernels(node, state, target) and any(
                isinstance(e.src, nodes.Tasklet) for e in state.in_edges(node)
            ):
                written.add(node.data)
            if not outside_kernels(node, state, target) or any(
                not isinstance(e.dst, nodes.Tasklet) or not outside_kernels(e.dst, state, target)
                for e in state.out_edges(node)
            ):
                passed_in.add(node.data)
    return written - passed_in


def promote_host_scalars(sdfg: SDFG, target: Target) -> None:
    """Promotes the host scalars that tasklets outside kernels write to symbols, so their assignments can sift."""
    promote = host_scalars(sdfg, target)
    if not promote:
        return
    others = {name for nested in sdfg.all_sdfgs_recursive() for name, desc in nested.arrays.items()}
    promotion = ScalarToSymbolPromotion()
    promotion.ignore = others - promote
    promotion.integers_only = False
    promotion.apply_pass(sdfg, {})


def prologue(state: SDFGState) -> list[nodes.Tasklet]:
    """The top-level tasklets of ``state`` that compute scalars only from data no kernel of ``state`` writes, in
    dataflow order."""
    found: list[nodes.Tasklet] = []
    for node in sdutil.dfs_topological_sort(state):
        if not isinstance(node, nodes.Tasklet) or state.entry_node(node) is not None:
            continue
        sources = [e.src for e in state.in_edges(node)]
        inputs_ready = all(
            isinstance(src, nodes.AccessNode) and all(e.src in found for e in state.in_edges(src)) for src in sources
        )
        outputs = [e.dst for e in state.out_edges(node)]
        writes_scalars = all(
            isinstance(dst, nodes.AccessNode) and isinstance(state.sdfg.arrays[dst.data], dt.Scalar) for dst in outputs
        )
        if inputs_ready and writes_scalars:
            found.append(node)
    return found


def gatherable(state: SDFGState, target: Target) -> bool:
    """Whether ``state`` holds only kernels, access nodes and its prologue at the top level."""
    tasklets = set(prologue(state))
    return all(
        state.entry_node(n) is not None
        or isinstance(n, (nodes.AccessNode, nodes.ExitNode))
        or is_kernel(n, state, target)
        or n in tasklets
        for n in state.nodes()
    )


def host_scalar_state(state: SDFGState, target: Target) -> bool:
    """Whether ``state`` only computes host scalars, from data it does not write."""
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    return (
        bool(tasklets)
        and not has_kernel(state, target)
        and len(prologue(state)) == len(tasklets)
        and gatherable(state, target)
    )


def swappable(region: ControlFlowRegion, block: ControlFlowBlock, state: SDFGState) -> bool:
    """Whether ``state``, which follows ``block`` in ``region``, can run before it instead."""
    if region.out_degree(block) != 1 or region.out_degree(state) != 1 or region.in_degree(block) > 1:
        return False
    between, after = region.out_edges(block)[0].data, region.out_edges(state)[0].data
    if not between.is_unconditional() or not after.is_unconditional():
        return False
    state_reads, state_writes = effects(state)
    block_reads, block_changes = effects(block)
    between_names = set(between.assignments) | {n for rhs in between.assignments.values() for n in rhs_names(rhs)}
    after_reads = {n for rhs in after.assignments.values() for n in rhs_names(rhs)}
    return not (
        block_changes & state_reads
        or (block_reads | block_changes) & state_writes
        or between_names & (state_reads | state_writes)
        or set(after.assignments) & between_names
        or after_reads & set(between.assignments)
    )


def swap(region: ControlFlowRegion, block: ControlFlowBlock, state: SDFGState) -> None:
    """Runs ``state`` before ``block``; the edges after both carry their assignments to what followed ``state``."""
    between, after = region.out_edges(block)[0], region.out_edges(state)[0]
    assignments = {**between.data.assignments, **after.data.assignments}
    for edge in region.in_edges(block):
        region.remove_edge(edge)
        region.add_edge(edge.src, state, edge.data)
    region.remove_edge(between)
    region.remove_edge(after)
    region.add_edge(state, block, InterstateEdge())
    region.add_edge(block, after.dst, InterstateEdge(assignments=assignments))
    if region.start_block is block:
        region.start_block = region.node_id(state)


def hoist_host_scalar_states(sdfg: SDFG, target: Target) -> int:
    """Swaps every state that only computes host scalars up past the blocks that neither change its inputs nor touch
    its outputs, so kernels before and after it meet. Returns the number of swaps."""
    swaps = 0
    for region in [r for r in sdfg.all_control_flow_regions() if not isinstance(r, ConditionalBlock)]:
        moved = True
        while moved:
            moved = False
            for state in [b for b in region.nodes() if isinstance(b, SDFGState) and host_scalar_state(b, target)]:
                while region.in_degree(state) == 1:
                    block = region.in_edges(state)[0].src
                    # Past another host scalar state nothing is gained, and two independent ones would swap forever
                    is_scalars = isinstance(block, SDFGState) and host_scalar_state(block, target)
                    if is_scalars or not swappable(region, block, state):
                        break
                    swap(region, block, state)
                    swaps += 1
                    moved = True
    return swaps


def has_kernel(state: SDFGState, target: Target) -> bool:
    return any(is_kernel(n, state, target) for n in state.nodes())


def runs_of_kernel_states(region: ControlFlowRegion, target: Target) -> list[list[SDFGState]]:
    """The maximal runs of consecutive gatherable kernel states of ``region``, longer than one state."""
    blocks = linear_chain(region) or []
    runs: list[list[SDFGState]] = [[]]
    for block in blocks:
        if isinstance(block, SDFGState) and has_kernel(block, target) and gatherable(block, target):
            runs[-1].append(block)
        elif runs[-1]:
            runs.append([])
    return [run for run in runs if len(run) > 1]


def fuse(region: ControlFlowRegion, run: list[SDFGState]) -> list[SDFGState]:
    """Fuses each state of ``run`` into the one before it where ``StateFusionExtended`` allows; returns the states
    left."""
    left = [run[0]]
    for state in run[1:]:
        pair = {"first_state": left[-1], "second_state": state}
        if StateFusionExtended.can_be_applied_to(region.sdfg, **pair):
            StateFusionExtended.apply_to(region.sdfg, verify=False, save=False, **pair)
            left[-1] = left[-1] if left[-1] in region.nodes() else state
        else:
            left.append(state)
    return left


def move_prologue_before(state: SDFGState) -> None:
    tasklets = prologue(state)
    if tasklets:
        moved = {n for t in tasklets for n in [t, *state.predecessors(t), *state.successors(t)]}
        xfh.state_fission(SubgraphView(state, moved), label=f"{state.label}_prologue")


def move_epilogue_after(state: SDFGState, target: Target) -> None:
    """Moves the host containers that ``state``'s kernels feed (copies back to the host), and what reads them, into a
    state after ``state``."""
    sdfg = state.sdfg
    host_writes = [
        n
        for n in state.data_nodes()
        if state.entry_node(n) is None
        and state.in_degree(n) > 0
        and sdfg.arrays[n.data].storage not in target.top_level_storage
    ]
    epilogue = {n for written in host_writes for n in state.bfs_nodes(written)}
    if epilogue:
        kernels = [n for n in state.nodes() if n not in epilogue]
        xfh.state_fission(SubgraphView(state, kernels), label=f"{state.label}_kernels")


@properties.make_properties
@transformation.explicit_cf_compatible
class GatherKernelStates(ppl.Pass):
    """Lowers host maps over kernels to loop regions, promotes the host scalars they compute to symbols, renames
    symbols into single-assignment form and sifts assignments up, then fuses runs of consecutive kernel states into as
    few states as ``StateFusionExtended`` allows. Host tasklets left that only compute scalars for the kernels move
    into a state before each fused one, and copies back to the host into a state after it. Returns the number of
    states removed."""

    CATEGORY: str = "Optimization Preparation"

    device = properties.EnumProperty(
        dtype=dtypes.DeviceType, default=dtypes.DeviceType.GPU, desc="The device whose kernels are gathered"
    )

    def __init__(self, device: dtypes.DeviceType = dtypes.DeviceType.GPU) -> None:
        super().__init__()
        self.device = device

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def runs(sdfg: SDFG, target: Target) -> list[tuple[ControlFlowRegion, list[SDFGState]]]:
        regions = [r for r in sdfg.all_control_flow_regions() if not isinstance(r, ConditionalBlock)]
        return [(region, run) for region in regions for run in runs_of_kernel_states(region, target)]

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        target = TARGETS[self.device]
        before = sum(1 for state in sdfg.states())
        lower_host_maps(sdfg, target)
        promote_host_scalars(sdfg, target)
        ppl.Pipeline([SymbolSSA(), SiftAssignmentsUp()]).apply_pass(sdfg, {})
        hoist_host_scalar_states(sdfg, target)
        for region, run in self.runs(sdfg, target):
            fuse(region, run)
        for state in [s for s in sdfg.states() if has_kernel(s, target) and gatherable(s, target)]:
            move_prologue_before(state)
            move_epilogue_after(state, target)
        removed = before - sum(1 for state in sdfg.states())
        return removed or None
