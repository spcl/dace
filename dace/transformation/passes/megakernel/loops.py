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
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.megakernel.discovery import TARGETS, MegakernelChain, find_chains, kernels_in, states_in
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


def launch_bound(loop: LoopRegion, target) -> sympy.Expr | None:
    """The largest iteration space any kernel of ``loop`` runs in one iteration, free of the loop variable, or
    ``None`` when no such bound is known."""
    bounds = [loop_free_bound(iteration_space(k), loop) for k in kernels_in(loop, target)]
    if not bounds or any(b is None for b in bounds):
        return None
    return sympy.Max(*bounds)


def per_launch_copy(loop: LoopRegion) -> tuple[LoopRegion, dict]:
    """``loop`` as it launches its kernels today, with the descriptors of the containers it accesses: fusing ``loop``
    moves its transients into the megakernel, and the copy still needs its own."""
    fallback = copy.deepcopy(loop)
    accessed = {node.data for state in states_in(fallback) for node in state.data_nodes()}
    return fallback, {name: copy.deepcopy(desc) for name, desc in loop.sdfg.arrays.items() if name in accessed}


def guard_with_fallback(host: SDFGState, fallback: LoopRegion, fits: str) -> None:
    """Replaces ``host`` (the state launching the megakernel) by a conditional: ``host`` when ``fits`` holds, the
    original per-launch ``fallback`` loop otherwise."""
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
    """Runs every sequential loop of kernels as one persistent kernel (module docstring)."""

    CATEGORY: str = "Optimization Preparation"

    block_size = properties.Property(dtype=int, default=256, desc="Threads per block of a GPU megakernel")
    device = properties.EnumProperty(
        dtype=dtypes.DeviceType, default=dtypes.DeviceType.GPU, desc="The device whose kernel loops are fused"
    )
    loops = properties.SetProperty(
        element_type=str, default=set(), desc="Labels of the loops to fuse; empty fuses every loop of kernels"
    )

    def __init__(self, device: dtypes.DeviceType = dtypes.DeviceType.GPU, loops: set[str] | None = None) -> None:
        super().__init__()
        self.device = device
        self.loops = set(loops or ())

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        target = TARGETS[self.device]
        # Cheapest structural check first: no sequential loop holds a kernel, nothing to do
        if not any(
            isinstance(r, LoopRegion) and kernels_in(r, target) for r in sdfg.all_control_flow_regions(recursive=True)
        ):
            return None
        GatherKernelStates(self.device).apply_pass(sdfg, {})
        fused = 0
        for chain in loop_chains(find_chains(sdfg, sdfg, target)):
            loop = chain.blocks[0]
            if self.loops and loop.label not in self.loops:
                continue
            gpu = self.device == dtypes.DeviceType.GPU
            bound = launch_bound(loop, target) if gpu else None
            if gpu and bound is None:
                continue
            fallback, descriptors = per_launch_copy(loop) if gpu else (None, {})
            loop_sdfg = loop.sdfg
            mk = MegaKernelOptimization(device=self.device)
            mk.block_size = self.block_size
            fused_chain = mk.fuse_chain(sdfg, chain, target, f"{chain.region.sdfg.name}_loop_megakernel_{fused}")
            if fused_chain is None:
                continue
            host, nblk = fused_chain
            if gpu:
                for name, desc in descriptors.items():
                    if name not in loop_sdfg.arrays:
                        loop_sdfg.add_datadesc(name, desc)
                guard_with_fallback(host, fallback, f"({symbolic.symstr(bound)}) <= {nblk} * {self.block_size}")
            fused += 1
        return fused or None
