# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which parts of an SDFG one persistent megakernel can run: maximal hierarchical linear chains of blocks that hold
GPU kernels.

A region is hierarchical linear when its blocks form one unconditional chain and every loop body and conditional
branch inside is again such a chain. Every block of a fusable chain runs on the device or only moves grid-uniform
symbols, so all of it can live in one kernel: kernel states, empty states, and loops and conditionals whose
conditions read only symbols.
"""

from dataclasses import dataclass, field

from dace import SDFG, dtypes, symbolic
from dace import data as dt
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)


@dataclass(slots=True)
class MegakernelChain:
    """Consecutive blocks of ``region`` that one megakernel runs, and the kernels inside them."""

    region: ControlFlowRegion
    blocks: list[ControlFlowBlock]
    kernels: list[nodes.MapEntry] = field(default_factory=list)


def linear_chain(region: ControlFlowRegion) -> list[ControlFlowBlock] | None:
    """``region``'s blocks in order when they form one unconditional chain, else ``None``."""
    blocks = [region.start_block]
    while region.out_degree(blocks[-1]) == 1:
        edge = region.out_edges(blocks[-1])[0]
        if not edge.data.is_unconditional() or edge.dst in blocks:
            return None
        blocks.append(edge.dst)
    if region.out_degree(blocks[-1]) > 1:
        return None
    return blocks if len(blocks) == region.number_of_nodes() else None


def holds_jump(block: ControlFlowBlock) -> bool:
    """Whether ``block`` is or holds a break, continue or return, which leave a megakernel's straight-line control
    flow."""
    if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
        return True
    return isinstance(block, (ControlFlowRegion, ConditionalBlock)) and any(
        isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in block.all_control_flow_blocks()
    )


def is_kernel(node: nodes.Node, state: SDFGState) -> bool:
    return (
        isinstance(node, nodes.MapEntry)
        and state.entry_node(node) is None
        and node.map.schedule == dtypes.ScheduleType.GPU_Device
    )


def device_state(state: SDFGState) -> bool:
    """Whether everything ``state`` does at the top level can run in one kernel: its kernels, and copies between
    containers on the device (a copy becomes one more grid-stride stage). Host code or host data ends the chain."""
    sdfg = state.sdfg
    for node in state.nodes():
        if state.entry_node(node) is not None or isinstance(node, nodes.ExitNode) or is_kernel(node, state):
            continue
        if not isinstance(node, nodes.AccessNode):
            return False
        if sdfg.arrays[node.data].storage != dtypes.StorageType.GPU_Global:
            return False
    return True


def states_in(block: ControlFlowBlock) -> list[SDFGState]:
    if isinstance(block, SDFGState):
        return [block]
    return [b for b in block.all_control_flow_blocks() if isinstance(b, SDFGState)]


def written_in(block: ControlFlowBlock) -> set[str]:
    return {name for state in states_in(block) for name in state.read_and_write_sets()[1]}


def uniform(names: set[str], sdfg: SDFG, written: set[str]) -> bool:
    """Whether code reading ``names`` computes the same value in every thread: it reads symbols, and scalars that
    nothing in the megakernel writes (they reach the kernel by value)."""
    data = names & sdfg.arrays.keys()
    return all(isinstance(sdfg.arrays[name], dt.Scalar) and name not in written for name in data)


def code_names(code: CodeBlock) -> set[str]:
    return {str(name) for stmt in code.code for name in symbolic.symbols_in_ast(stmt)}


def uniform_edges(region: ControlFlowRegion, sdfg: SDFG, written: set[str]) -> bool:
    """Whether every interstate edge of ``region`` assigns values that are the same in every thread."""
    return all(
        uniform(symbolic.free_symbols_and_functions(rhs) | symbolic.arrays(rhs), sdfg, written)
        for e in region.edges()
        for rhs in e.data.assignments.values()
    )


def fusable(block: ControlFlowBlock, sdfg: SDFG) -> bool:
    """Whether one megakernel can run ``block`` whole: it is device work, and all control flow inside it is
    hierarchical linear and grid-uniform."""
    if holds_jump(block):
        return False
    if isinstance(block, SDFGState):
        return device_state(block)
    written = written_in(block)
    if isinstance(block, LoopRegion):
        loop_code = [block.loop_condition, block.init_statement, block.update_statement]
        if not all(uniform(code_names(code), sdfg, written) for code in loop_code if code is not None):
            return False
        return fusable_region(block, sdfg, written)
    if isinstance(block, ConditionalBlock):
        if not all(cond is None or uniform(code_names(cond), sdfg, written) for cond, branch in block.branches):
            return False
        return all(fusable_region(branch, sdfg, written) for cond, branch in block.branches)
    return False


def fusable_region(region: ControlFlowRegion, sdfg: SDFG, written: set[str]) -> bool:
    blocks = linear_chain(region)
    return blocks is not None and uniform_edges(region, sdfg, written) and all(fusable(b, sdfg) for b in blocks)


def kernels_in(block: ControlFlowBlock) -> list[nodes.MapEntry]:
    return [n for state in states_in(block) for n in state.nodes() if is_kernel(n, state)]


def find_chains(region: ControlFlowRegion, sdfg: SDFG) -> list[MegakernelChain]:
    """The maximal fusable chains with kernels in ``region`` and, where a block cannot be fused whole, inside it."""
    blocks = linear_chain(region)
    if blocks is None:  # Not linear here: every block stands alone
        blocks_and_breaks = [(b, True) for b in region.nodes()]
    else:
        blocks_and_breaks = [(b, False) for b in blocks]
    chains: list[MegakernelChain] = []
    current: list[ControlFlowBlock] = []

    def close() -> None:
        kernels = [k for b in current for k in kernels_in(b)]
        if kernels:
            chains.append(MegakernelChain(region, list(current), kernels))
        current.clear()

    for block, alone in blocks_and_breaks:
        written = set().union(*(written_in(b) for b in current), written_in(block))
        uniform_in = all(
            uniform(symbolic.free_symbols_and_functions(rhs) | symbolic.arrays(rhs), sdfg, written)
            for e in region.in_edges(block)
            for rhs in e.data.assignments.values()
        )
        if current and (alone or not uniform_in):
            close()
        if fusable(block, sdfg):
            current.append(block)
            if alone:
                close()
            continue
        close()
        for child, child_sdfg in child_regions(block, sdfg):
            chains.extend(find_chains(child, child_sdfg))
    close()
    return chains


def runs_on_host(node: nodes.Node, state: SDFGState) -> bool:
    """Whether no GPU scope encloses ``node`` in ``state``."""
    scope = state.entry_node(node)
    while scope is not None:
        if scope.map.schedule in dtypes.GPU_SCHEDULES:
            return False
        scope = state.entry_node(scope)
    return True


def child_regions(block: ControlFlowBlock, sdfg: SDFG) -> list[tuple[ControlFlowRegion, SDFG]]:
    """The regions inside ``block`` that may hold chains, with their SDFGs: branches, a region's body, and the nested
    SDFGs a state calls from the host (a host loop over a nested SDFG of kernels, for example)."""
    if isinstance(block, ConditionalBlock):
        return [(branch, sdfg) for cond, branch in block.branches]
    if isinstance(block, ControlFlowRegion):
        return [(block, sdfg)]
    if isinstance(block, SDFGState):
        return [(n.sdfg, n.sdfg) for n in block.nodes() if isinstance(n, nodes.NestedSDFG) and runs_on_host(n, block)]
    return []
