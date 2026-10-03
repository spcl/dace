# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Program structure for global layout assignment: one kernel per state, and the flat line graph of kernels (:func:`kernel_per_state`, :func:`check_kernel_per_state`, :func:`line_graph`)."""
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation.helpers import state_fission
from dace.transformation.layout.externalize import nest_entries


def kernel_per_state(sdfg: SDFG) -> int:
    """Split every state with several top-level map scopes into one state per kernel (A2); runs to fixpoint since ``state_fission`` can drag a second nest along. Returns the split count."""

    def excess_nests() -> int:
        return sum(len(nest_entries(s)) - 1 for s in sdfg.states() if len(nest_entries(s)) > 1)

    splits = 0
    while True:
        multi = [s for s in list(sdfg.states()) if len(nest_entries(s)) > 1]
        if not multi:
            return splits
        excess = excess_nests()
        for state in multi:
            entries = nest_entries(state)
            entry_set = set(entries)
            first = next(n for n in dfs_topological_sort(state) if n in entry_set)
            state_fission(state.scope_subgraph(first), label=f"{state.label}_k{splits}", allow_isolated_nodes=False)
            splits += 1
        if excess_nests() >= excess:
            offenders = [s.label for s in sdfg.states() if len(nest_entries(s)) > 1]
            raise RuntimeError(f"kernel_per_state: state_fission made no progress separating the nests of "
                               f"{offenders} (a shared sink access node or a view between the nests drags the "
                               f"second nest into the fissioned state); refusing rather than reporting a split "
                               f"that did not happen.")


def check_kernel_per_state(sdfg: SDFG) -> None:
    """A6 invariant guard: raises if a later pass (``simplify``/``StateFusion``) fused split states back together."""
    for state in sdfg.states():
        entries = nest_entries(state)
        if len(entries) > 1:
            raise RuntimeError(f"kernel-per-state invariant broken: state '{state.label}' holds {len(entries)} "
                               f"top-level map scopes ({[e.map.label for e in entries]}). A pass fused split "
                               f"states back (simplify/StateFusion must not run after kernel_per_state); re-split "
                               f"or drop the offending pass.")


@dataclass
class KernelState:
    """One kernel of the line graph. ``loop`` is the (single) enclosing LoopRegion when the kernel lives
    inside a loop body, else None -- consecutive kernels sharing a ``loop`` form one loop span, which the
    body-uniform layout model pins to a single layout (see ``loop_spans``)."""
    state: SDFGState
    map_entry: nodes.MapEntry
    index: int
    loop: Optional[LoopRegion] = None


def state_does_work(state: SDFGState) -> bool:
    """True iff ``state`` moves data: beyond bare access nodes and connector-less tasklets, access nodes joined by copy memlets count too (the edge scan catches those)."""
    for n in state.nodes():
        if isinstance(n, nodes.AccessNode):
            continue
        if isinstance(n, nodes.Tasklet) and not n.in_connectors and not n.out_connectors:
            continue
        return True
    return any(e.data is not None and e.data.data is not None for e in state.edges())


def is_relayout_state(state: SDFGState) -> bool:
    """True iff every working node of ``state`` is a ``LayoutChange`` node (a relayout boundary, task A4)."""
    from dace.libraries.layout import LayoutChange  # lazy: avoid eager import of dace.libraries.layout

    work = [n for n in state.nodes() if not isinstance(n, nodes.AccessNode)]
    return len(work) > 0 and all(isinstance(n, LayoutChange) for n in work)


def add_state_kernel(sdfg: SDFG, block: SDFGState, kernels: List[KernelState], loop: Optional[LoopRegion]) -> None:
    """Append the one map nest of ``block`` as a kernel (tagged with its enclosing ``loop``), or refuse non-map work."""
    entries = nest_entries(block)
    if len(entries) > 1:
        check_kernel_per_state(sdfg)  # raises with the A6 message
    if len(entries) == 1:
        kernels.append(KernelState(state=block, map_entry=entries[0], index=len(kernels), loop=loop))
    elif is_relayout_state(block):
        # a relayout takes no kernel position, which is right at top level but WRONG inside a loop body: the
        # kernels on either side would fuse into one span (pinning one layout) while the LayoutChange it sits
        # between silently re-runs every iteration, unmodelled
        if loop is not None:
            raise NotImplementedError(f"line_graph: relayout state '{block.label}' inside loop '{loop.label}' "
                                      f"-- a LayoutChange in a loop body re-runs every iteration and is not "
                                      f"modelled; hoist it out of the loop.")
    elif state_does_work(block):
        raise NotImplementedError(f"line_graph: state '{block.label}' does non-map work at top level -- v1 scores "
                                  f"map nests only (expand/canonicalize the state first).")


def collect_line(sdfg: SDFG, region, kernels: List[KernelState], loop: Optional[LoopRegion]) -> None:
    """Walk ``region``'s blocks as a flat line, appending kernels; recurse once into a LoopRegion (body must
    itself be a flat line). Refuses branches, DAGs, cycles, nested loops, conditionals, and unreachable blocks."""
    if region.number_of_nodes() == 0:  # an empty loop body contributes no kernels
        return
    block: Optional[ControlFlowBlock] = region.start_block
    seen = set()
    while block is not None:
        if block in seen:
            raise NotImplementedError(f"line_graph: cycle through block '{block.label}' -- not a line.")
        seen.add(block)
        out_edges = region.out_edges(block)
        if len(out_edges) > 1 or region.in_degree(block) > 1:
            raise NotImplementedError(f"line_graph: block '{block.label}' has {len(out_edges)} successors / "
                                      f"{region.in_degree(block)} predecessors -- v1 refuses branches and DAGs.")
        if isinstance(block, SDFGState):
            add_state_kernel(sdfg, block, kernels, loop)
        elif isinstance(block, LoopRegion):
            if loop is not None:
                raise NotImplementedError(f"line_graph: nested LoopRegion '{block.label}' inside '{loop.label}' -- "
                                          f"v1 handles a single loop level (nested loops are deferred).")
            collect_line(sdfg, block, kernels, loop=block)
        else:
            raise NotImplementedError(f"line_graph: control-flow block '{block.label}' of type "
                                      f"{type(block).__name__} -- v1 handles states and single-level LoopRegions "
                                      f"only (conditionals are deferred).")
        block = out_edges[0].dst if out_edges else None
    if len(seen) != region.number_of_nodes():
        unreached = {b.label for b in region.nodes()} - {b.label for b in seen}
        raise NotImplementedError(f"line_graph: unreachable blocks {sorted(unreached)} in '{region.label}' "
                                  f"-- not a line.")


def line_graph(sdfg: SDFG) -> List[KernelState]:
    """Extract the ordered kernel sequence of ``sdfg`` (A3). A LoopRegion whose body is itself a flat line is
    admitted: its body kernels are appended in program order, each tagged with the loop (``KernelState.loop``);
    the body-uniform layout model pins each such span to one layout. Still refuses branches, DAGs, nested
    loops, conditionals, and states with non-map work at top level."""
    kernels: List[KernelState] = []
    collect_line(sdfg, sdfg, kernels, loop=None)
    return kernels


def loop_spans(kernels: List[KernelState]) -> List[Tuple[int, int]]:
    """The ``[start, end)`` index ranges of the maximal runs of kernels sharing one enclosing LoopRegion.
    Each span is pinned to a single layout by the body-uniform model, so entry/exit relayouts hoist outside
    the loop. Kernels outside any loop (``loop is None``) are not spans."""
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < len(kernels):
        loop = kernels[start].loop
        end = start
        while end < len(kernels) and kernels[end].loop is loop:
            end += 1
        if loop is not None:
            spans.append((start, end))
        start = end
    return spans


def locked_transitions(kernels: List[KernelState]) -> Set[int]:
    """Kernel indices ``k`` whose transition from ``k-1`` must not change layout: the internal transitions of
    each loop span (body-uniform model). The transition INTO a span (prologue -> first body kernel) and OUT of
    it (last body kernel -> epilogue) stay free, so each span's one-time entry/exit relayout is still priced."""
    locked: Set[int] = set()
    for start, end in loop_spans(kernels):
        locked.update(range(start + 1, end))
    return locked
