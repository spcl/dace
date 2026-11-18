# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegakernelLoops``: a persistent kernel only where kernels are launched inside a sequential loop.

A loop that launches its kernels once per iteration (a time-step loop, a wavefront over diagonals) pays one launch per
kernel and iteration. When that work is small, the launches dominate, and one persistent kernel that runs the whole
loop with grid barriers between its stages is faster. A straight-line sequence of kernels outside a loop launches each
kernel once, and fusing it only adds barriers (ICON velocity tendencies ran 2x slower as one megakernel), so it is left
alone.

The candidate is the outermost sequential ``LoopRegion`` that :class:`MegaKernelOptimization` can run whole: every
block in it is device work, and its control flow reads only grid-uniform values. On the GPU the loop is kept twice
behind a runtime test: the persistent kernel when every kernel's iteration space fits the threads the device holds at
once, the per-launch loop otherwise. On the CPU one OpenMP team runs the loop, its maps work-shared inside, at any size.
"""

import copy

import sympy

from dace import SDFG, SDFGState, dtypes, properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.megakernel.discovery import (
    TARGETS,
    MegakernelChain,
    find_chains,
    kernels_in,
    linear_chain,
    states_in,
)
from dace.transformation.passes.megakernel.gather_kernels import GatherKernelStates
from dace.transformation.passes.megakernel.megakernel import MegaKernelOptimization


def loop_chains(chains: list[MegakernelChain]) -> list[MegakernelChain]:
    """The sequential loops among the blocks of ``chains`` that hold kernels, each as a chain of its own."""
    return [
        MegakernelChain(chain.region, [block], list(chain.kernels))
        for chain in chains
        for block in chain.blocks
        if isinstance(block, LoopRegion) and block.loop_variable
    ]


def iteration_space(entry: nodes.MapEntry) -> sympy.Expr:
    return sympy.Mul(*[symbolic.pystr_to_symbolic(n) for n in entry.map.range.size()])


def loop_free_bound(work: sympy.Expr, loop: LoopRegion) -> sympy.Expr | None:
    """An upper bound of ``work`` over every iteration of ``loop`` that does not name the loop variable, or ``None``.
    Work affine in the variable takes its largest value at an end of the range."""
    var = symbolic.pystr_to_symbolic(loop.loop_variable)
    if var not in work.free_symbols:
        return work
    if not work.is_polynomial(var) or sympy.degree(work, var) > 1:
        return None
    begin, end = loop_range(loop)
    if begin is None:
        return None
    return sympy.Max(work.subs(var, begin), work.subs(var, end))


def loop_range(loop: LoopRegion) -> tuple:
    begin, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    if begin is None or end is None:
        return None, None
    return begin, end


def launch_bound(blocks: list[ControlFlowBlock], target) -> sympy.Expr | None:
    """The largest iteration space any kernel of ``blocks`` runs at once, free of the variables of the loops among
    them, or ``None`` when no such bound is known."""
    bounds = []
    for block in blocks:
        inner = [] if isinstance(block, SDFGState) else block.all_control_flow_regions(recursive=True)
        loops = [r for r in [block, *inner] if is_loop(r)]
        for kernel in kernels_in(block, target):
            work: sympy.Expr | None = iteration_space(kernel)
            for loop in loops:
                work = None if work is None else loop_free_bound(work, loop)
            bounds.append(work)
    if not bounds or any(b is None for b in bounds):
        return None
    return sympy.Max(*bounds)


def is_loop(block: ControlFlowBlock) -> bool:
    return isinstance(block, LoopRegion) and bool(block.loop_variable)


def per_launch_copy(chain: MegakernelChain) -> tuple[ControlFlowBlock, dict]:
    """The blocks of ``chain`` as they launch their kernels today (one block, or a region running them in order), with
    the descriptors of the containers they access: fusing moves their transients into the megakernel, and the copy
    still needs its own."""
    copies = [copy.deepcopy(block) for block in chain.blocks]
    if len(copies) == 1:
        fallback: ControlFlowBlock = copies[0]
    else:
        fallback = ControlFlowRegion(f"{chain.blocks[0].label}_run", sdfg=chain.region.sdfg)
        fallback.add_node(copies[0], is_start_block=True)
        for i, (before, after) in enumerate(zip(chain.blocks, chain.blocks[1:])):
            (edge,) = chain.region.edges_between(before, after)
            fallback.add_edge(copies[i], copies[i + 1], copy.deepcopy(edge.data))
    accessed = {node.data for block in copies for state in states_in(block) for node in state.data_nodes()}
    arrays = chain.region.sdfg.arrays
    return fallback, {name: copy.deepcopy(desc) for name, desc in arrays.items() if name in accessed}


def guard_with_fallback(host: SDFGState, fallback: ControlFlowBlock, fits: str) -> None:
    """Replaces ``host`` (the state launching the megakernel) by a conditional: ``host`` when ``fits`` holds, the
    original per-launch ``fallback`` otherwise."""
    region: ControlFlowRegion = host.parent_graph
    guard = ConditionalBlock(f"{host.label}_fits")
    region.add_node(guard, is_start_block=region.start_block is host)
    for edge in region.in_edges(host):
        region.add_edge(edge.src, guard, edge.data)
    for edge in region.out_edges(host):
        region.add_edge(guard, edge.dst, edge.data)
    region.remove_node(host)
    persistent = ControlFlowRegion(f"{host.label}_persistent", sdfg=region.sdfg)
    persistent.add_node(host, is_start_block=True)
    per_launch = ControlFlowRegion(f"{host.label}_per_launch", sdfg=region.sdfg)
    per_launch.add_node(fallback, is_start_block=True)
    guard.add_branch(CodeBlock(fits), persistent)
    guard.add_branch(None, per_launch)


@properties.make_properties
@transformation.explicit_cf_compatible
class MegakernelLoops(ppl.Pass):
    """Runs every sequential loop of kernels as one persistent kernel (module docstring); or, given ``blocks``, the one
    run of consecutive blocks they name."""

    CATEGORY: str = "Optimization Preparation"

    block_size = properties.Property(dtype=int, default=256, desc="Threads per block of a GPU megakernel")
    device = properties.EnumProperty(
        dtype=dtypes.DeviceType, default=dtypes.DeviceType.GPU, desc="The device whose kernel loops are fused"
    )
    blocks = properties.SetProperty(
        element_type=str,
        default=set(),
        desc="Labels of consecutive blocks (states of kernels, loops) to run as one persistent kernel; empty runs every "
        "loop of kernels as one each",
    )

    def __init__(self, device: dtypes.DeviceType = dtypes.DeviceType.GPU, blocks: set[str] | None = None) -> None:
        super().__init__()
        self.device = device
        self.blocks = set(blocks or ())

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        target = TARGETS[self.device]
        if self.blocks:
            return self.fuse_named(sdfg, target)
        # Cheapest structural check first: no sequential loop holds a kernel, nothing to do
        if not any(is_loop(r) and kernels_in(r, target) for r in sdfg.all_control_flow_regions(recursive=True)):
            return None
        GatherKernelStates(self.device).apply_pass(sdfg, {})
        fused = 0
        for chain in loop_chains(find_chains(sdfg, sdfg, target)):
            fused += self.fuse(sdfg, chain, target, f"{chain.region.sdfg.name}_loop_megakernel_{fused}")
        return fused or None

    def fuse_named(self, sdfg: SDFG, target) -> int | None:
        """Fuses the run of blocks :attr:`blocks` names, or nothing. Blocks of one region must follow one another
        there; that is decided before gathering, which changes the SDFG, and their run moves into a region of its own,
        so gathering fuses its states only with each other. Blocks of different regions (a schedule may nest each
        loop on its own) are decided after gathering: they must lie in one chain whose run from the first to the last holds
        every named kernel and no other. Gathering inlines, wraps and renames blocks, so they are found by guid."""
        named = [b for b in sdfg.all_control_flow_blocks(recursive=True) if b.label in self.blocks]
        if {b.label for b in named} != self.blocks:
            return None
        kernels = {n.guid for b in named for n, holder in b.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
        marks = {b.guid for b in named} | kernels
        region = named[0].parent_graph
        if all(b.parent_graph is region for b in named):
            if not consecutive(named):
                return None
            marks |= guids_in(wrapped(named))
        GatherKernelStates(self.device).apply_pass(sdfg, {})
        for chain in find_chains(sdfg, sdfg, target):
            held = [i for i, block in enumerate(chain.blocks) if marks & guids_in(block)]
            if not held:
                continue
            run = chain.blocks[held[0] : held[-1] + 1]
            found = {k.guid for block in run for k in kernels_in(block, target)}
            # blocks between the named ones may only be host glue (inlining leaves empty states): no kernel
            if not kernels <= marks_of(run) or found - kernels:
                return None
            fused = self.fuse(
                sdfg,
                MegakernelChain(chain.region, run, [k for b in run for k in kernels_in(b, target)]),
                target,
                f"{chain.region.sdfg.name}_run_megakernel",
            )
            return fused or None
        return None

    def fuse(self, sdfg: SDFG, chain: MegakernelChain, target, handle: str) -> int:
        """Runs ``chain`` as one persistent kernel, on the GPU behind a test that its largest kernel fits the resident
        threads, with the per-launch blocks as the fallback. 1 when fused, else 0 (nothing changed)."""
        gpu = self.device == dtypes.DeviceType.GPU
        bound = launch_bound(chain.blocks, target) if gpu else None
        if gpu and bound is None:
            return 0
        fallback, descriptors = per_launch_copy(chain) if gpu else (None, {})
        region_sdfg = chain.region.sdfg
        mk = MegaKernelOptimization(device=self.device)
        mk.block_size = self.block_size
        fused_chain = mk.fuse_chain(sdfg, chain, target, handle)
        if fused_chain is None:
            return 0
        host, nblk = fused_chain
        if gpu:
            for name, desc in descriptors.items():
                if name not in region_sdfg.arrays:
                    region_sdfg.add_datadesc(name, desc)
            guard_with_fallback(host, fallback, f"({symbolic.symstr(bound)}) <= {nblk} * {self.block_size}")
        return 1


def wrapped(blocks: list[ControlFlowBlock]) -> ControlFlowRegion:
    """The consecutive ``blocks`` moved, in their order, into a new region that takes their place."""
    region = blocks[0].parent_graph
    order = linear_chain(region) or blocks
    run = sorted(blocks, key=order.index)
    inner = ControlFlowRegion(f"{run[0].label}_run", sdfg=region.sdfg)
    region.add_node(inner, is_start_block=region.start_block is run[0])
    for edge in region.in_edges(run[0]):
        region.add_edge(edge.src, inner, edge.data)
    for edge in region.out_edges(run[-1]):
        region.add_edge(inner, edge.dst, edge.data)
    between = [region.edges_between(a, b)[0].data for a, b in zip(run, run[1:])]
    for block in run:
        region.remove_node(block)
    inner.add_node(run[0], is_start_block=True)
    for (before, after), data in zip(zip(run, run[1:]), between):
        inner.add_node(after)
        inner.add_edge(before, after, data)
    return inner


def consecutive(blocks: list[ControlFlowBlock]) -> bool:
    """Whether ``blocks`` follow one another, unconditionally, in one region."""
    region = blocks[0].parent_graph
    if any(b.parent_graph is not region for b in blocks):
        return False
    order = linear_chain(region)
    if order is None or any(b not in order for b in blocks):
        return len(blocks) == 1
    held = sorted(order.index(b) for b in blocks)
    return held == list(range(held[0], held[0] + len(blocks)))


def marks_of(run: list[ControlFlowBlock]) -> set[str]:
    """The guids of every block and kernel map entry in ``run``."""
    return set().union(*(guids_in(block) for block in run))


def guids_in(block: ControlFlowBlock) -> set[str]:
    """The guids of ``block``, of the blocks inside it and of its kernels' map entries."""
    regions = [] if isinstance(block, SDFGState) else block.all_control_flow_blocks(recursive=True)
    return (
        {block.guid}
        | {b.guid for b in regions}
        | {n.guid for n, holder in block.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
    )
