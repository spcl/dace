# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Find the buffers of a loop that can be used for staging.

These are the transients each iteration fills with a copy before it reads them.

A staged buffer carries no value from one iteration to the next, and the copy that fills it depends on nothing the
iteration computes. Its copies can therefore run ahead of the rest of the body, which is what multiple buffering does.
"""

import dataclasses

import networkx as nx

from dace import SDFG, SDFGState, data
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowBlock, LoopRegion


@dataclasses.dataclass
class StagedCopy:
    """One copy filling a staged buffer used for multi-buffered loops

    The copy is ``src -> dst`` directly or going through a :class:`CopyLibraryNode`.
    """

    state: SDFGState
    src: nodes.AccessNode
    dst: nodes.AccessNode
    edges: list[MultiConnectorEdge[Memlet]]
    copy_node: CopyLibraryNode | None = None


@dataclasses.dataclass
class StagedBuffer:
    """A transient the loop body fills with ``copies`` at the start of every iteration."""

    name: str
    copies: list[StagedCopy]


def find_staged_buffers(
    loop: LoopRegion, sdfg: SDFG, names: set[str] | None = None
) -> tuple[list[StagedBuffer], dict[str, str]]:
    """Find the staged buffers of ``loop``.

    A transient array ``T`` is staged if:
    - Every write to ``T`` in the loop is a copy from another array, made at the top-level scope of a state that is a
      direct child of the loop, and all of them in the same state.
    - The copy sources are not written in the loop, and the copy subsets use no symbol the loop body assigns.
    - Every access to ``T`` in the loop comes after the copies in the same iteration, and ``T`` is not accessed outside
      the loop.

    :param loop: The loop to analyze.
    :param sdfg: The SDFG that holds ``loop``.
    :param names: If given, only these arrays are considered.
    :return: The staged buffers, and the reason each rejected candidate is not staged.
    """
    loop_states = set(loop.states())
    written = {node.data for state in loop_states for node in state.data_nodes() if state.in_degree(node) > 0}
    assigned = {sym for edge in loop.all_interstate_edges() for sym in edge.data.assignments}
    assigned |= {region.loop_variable for region in loop.all_control_flow_regions() if isinstance(region, LoopRegion)}
    assigned.discard(loop.loop_variable)
    interstate_reads = set()
    for edge in sdfg.all_interstate_edges():
        interstate_reads |= {str(sym) for sym in edge.data.free_symbols}

    accessed_in: dict[str, list[tuple[SDFGState, nodes.AccessNode]]] = {}
    accessed_outside: set[str] = set()
    for state in sdfg.states():
        for node in state.data_nodes():
            if state in loop_states:
                accessed_in.setdefault(node.data, []).append((state, node))
            else:
                accessed_outside.add(node.data)

    candidates = {name for name in written if names is None or name in names}
    staged: list[StagedBuffer] = []
    rejected: dict[str, str] = {}
    for name in sorted(candidates):
        desc = sdfg.arrays[name]
        if not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, (data.View, data.Reference)):
            rejected[name] = "not a transient array"
            continue
        if name in accessed_outside:
            rejected[name] = "accessed outside the loop"
            continue
        if name in interstate_reads:
            rejected[name] = "read on an interstate edge"
            continue
        copies, reason = _find_copies(name, accessed_in[name], loop, sdfg, written, assigned)
        if reason is None:
            reason = _check_order(copies, accessed_in[name], loop)
        if reason is None:
            staged.append(StagedBuffer(name, copies))
        else:
            rejected[name] = reason
    return staged, rejected


def _find_copies(
    name: str,
    accesses: list[tuple[SDFGState, nodes.AccessNode]],
    loop: LoopRegion,
    sdfg: SDFG,
    written: set[str],
    assigned: set[str],
) -> tuple[list[StagedCopy], str | None]:
    """The copies that write ``name``, or the reason one of its writes is not a stageable copy."""
    copies = []
    for state, node in accesses:
        for edge in state.in_edges(node):
            copy_node = None
            edges = [edge]
            src = edge.src
            if isinstance(src, CopyLibraryNode):
                if state.in_degree(src) != 1 or state.out_degree(src) != 1:
                    # TODO: add an exception for TMA descriptor once rebase onto synchronous_tma is complete
                    return [], f"copy {src} has more than one input or output"
                copy_node = src
                edges.insert(0, state.in_edges(src)[0])
                src = edges[0].src
            if not isinstance(src, nodes.AccessNode) or src.data == name:
                return [], f"written by {src}, which is not a copy"
            if state.parent_graph is not loop:
                return [], f"copied in {state}, which is not a direct child of the loop"
            if state.entry_node(node) is not None:
                return [], "copied inside a scope"
            if src.data in written:
                return [], f"copy source {src.data} is written in the loop"
            if isinstance(sdfg.arrays[src.data], (data.View, data.Reference)):
                return [], f"copy source {src.data} is a view or reference"
            used_symbols = {str(sym) for e in edges for sym in e.data.free_symbols}
            if used_symbols & assigned:
                return [], f"copy subsets use symbols the loop body assigns: {sorted(used_symbols & assigned)}"
            copies.append(StagedCopy(state, src, node, edges, copy_node))
    if len({copy.state for copy in copies}) > 1:
        return [], "copied in more than one state"
    return copies, None


def _check_order(
    copies: list[StagedCopy], accesses: list[tuple[SDFGState, nodes.AccessNode]], loop: LoopRegion
) -> str | None:
    """Whether every access in the loop comes after its copies in the same iteration."""
    copy_state = copies[0].state
    graph = nx.DiGraph(loop.nx)
    dominators = nx.immediate_dominators(graph, loop.start_block)
    copy_dsts = [copy.dst for copy in copies]
    for state, node in accesses:
        if state is copy_state:
            if node not in copy_dsts and not any(nx.has_path(state.nx, dst, node) for dst in copy_dsts):
                return f"accessed in {state} before it is copied"
            continue
        block = _child_of(loop, state)
        if not _dominates(dominators, copy_state, block):
            return f"accessed in {state}, which does not follow the copy"
    return None


def _child_of(loop: LoopRegion, block: ControlFlowBlock) -> ControlFlowBlock:
    """The direct child of ``loop`` that holds ``block``."""
    while block.parent_graph is not loop:
        block = block.parent_graph
    return block


def _dominates(dominators: dict, dominator: ControlFlowBlock, block: ControlFlowBlock) -> bool:
    while True:
        if block is dominator:
            return True
        parent = dominators.get(block)
        if parent is None or parent is block:
            return False
        block = parent
