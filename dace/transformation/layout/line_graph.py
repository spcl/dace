# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Program structure for the global layout assignment: one kernel per state, and the flat line
graph of kernels (GLOBAL_LAYOUT_DESIGN.md, tasks A2 + A3 + the A6 invariant).

The assignment machinery reasons about a program as a LINE of kernel states: relayouts are only
ever inserted on state boundaries, per-nest costs attach to one state each, and the per-array DP
walks the states in order. This module establishes and guards that shape:

  * :func:`kernel_per_state` (A2) -- split every multi-kernel state until each top-level map scope
    stands alone, using ``dace.transformation.helpers.state_fission`` (which already owns the hard
    part: moving a scope with its dependencies into a fresh state and duplicating the boundary
    access node so the producer writes it in one state and the consumer reads it in the next).
  * :func:`check_kernel_per_state` (A6) -- the invariant guard. DaCe ``simplify``/``StateFusion``
    FUSES split states back together, so after the split the pipeline must never run them again on
    this SDFG (schedule passes run standalone on externalized copies); every later stage re-checks
    the invariant through this guard and refuses loudly if it broke.
  * :func:`line_graph` (A3) -- the ordered kernel sequence. v1 is deliberately narrow: branches,
    DAG-shaped state machines and LoopRegions are REFUSED loudly (the nested/loop-carried design is
    documented as deferred); states with no work are passed through, states with non-map work are
    refused.
"""
from dataclasses import dataclass
from typing import List, Optional

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation.helpers import state_fission
from dace.transformation.layout.externalize import nest_entries


def kernel_per_state(sdfg: SDFG) -> int:
    """Split every state holding several top-level map scopes so each kernel stands alone (A2).

    The topologically-first nest is fissioned into a fresh state BEFORE the current one, and the
    scan runs to FIXPOINT over all states -- ``state_fission`` can drag a second nest along into
    the state it creates (a shared sink access node pulls the other producer in via the upstream
    closure), so the states it creates are re-examined, and a round that separates nothing raises
    instead of mis-reporting success. Producer/consumer chains through in-state access nodes are
    handled by ``state_fission`` (the boundary access node is duplicated: written in the first
    state, read in the next). Returns the number of splits performed.
    """

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
    """The A6 invariant guard: every state holds at most one top-level map scope. Raises if a later
    pass (``simplify``/``StateFusion`` are the known offenders) fused split states back together."""
    for state in sdfg.states():
        entries = nest_entries(state)
        if len(entries) > 1:
            raise RuntimeError(f"kernel-per-state invariant broken: state '{state.label}' holds {len(entries)} "
                               f"top-level map scopes ({[e.map.label for e in entries]}). A pass fused split "
                               f"states back (simplify/StateFusion must not run after kernel_per_state); re-split "
                               f"or drop the offending pass.")


@dataclass
class KernelState:
    """One kernel of the line graph: its state, its single top-level map entry, and its position."""
    state: SDFGState
    map_entry: nodes.MapEntry
    index: int


def state_does_work(state: SDFGState) -> bool:
    """True iff ``state`` moves data: anything beyond bare access nodes and connector-less tasklets.
    A tasklet without connectors touches no array (canonicalize's symbol-assumption checks are the
    known case) -- structural for layout purposes, passed through. A state of bare access nodes
    joined by copy memlets DOES move data (the frontend keeps direct copies): it must not slip
    through as structural, or apply_assignment's liveness never sees its reads/writes -- the edge
    scan catches it and :func:`line_graph` then refuses it as non-map work."""
    for n in state.nodes():
        if isinstance(n, nodes.AccessNode):
            continue
        if isinstance(n, nodes.Tasklet) and not n.in_connectors and not n.out_connectors:
            continue
        return True
    return any(e.data is not None and e.data.data is not None for e in state.edges())


def is_relayout_state(state: SDFGState) -> bool:
    """True iff every working node of ``state`` is a ``LayoutChange`` library node -- i.e. the state
    is a relayout boundary this pipeline inserted (task A4). Such states sit BETWEEN kernels in the
    line graph and take no kernel position."""
    from dace.libraries.layout import LayoutChange  # local: dace.libraries.layout imports lazily

    work = [n for n in state.nodes() if not isinstance(n, nodes.AccessNode)]
    return len(work) > 0 and all(isinstance(n, LayoutChange) for n in work)


def line_graph(sdfg: SDFG) -> List[KernelState]:
    """Extract the flat ordered kernel sequence of ``sdfg`` (A3).

    v1 contract, enforced loudly:

      * the top-level control flow is a plain LINE of ``SDFGState``s -- any non-state block
        (LoopRegion, conditional region) and any branch/merge (out- or in-degree > 1) is refused;
      * every working state holds EXACTLY one top-level map scope (A2 ran; A6 holds) -- a state
        with work but no map is refused (v1 scores map nests only);
      * empty/structural states are passed through and take no position in the kernel order.
    """
    for block in sdfg.nodes():
        if not isinstance(block, SDFGState):
            raise NotImplementedError(
                f"line_graph: control-flow block '{block.label}' of type {type(block).__name__} -- "
                f"v1 handles a flat line of states only (LoopRegions/conditionals are deferred).")

    kernels: List[KernelState] = []
    block: Optional[ControlFlowBlock] = sdfg.start_block
    seen = set()
    while block is not None:
        if block in seen:
            raise NotImplementedError(f"line_graph: cycle through state '{block.label}' -- not a line.")
        seen.add(block)
        out_edges = sdfg.out_edges(block)
        if len(out_edges) > 1 or sdfg.in_degree(block) > 1:
            raise NotImplementedError(f"line_graph: state '{block.label}' has {len(out_edges)} successors / "
                                      f"{sdfg.in_degree(block)} predecessors -- v1 refuses branches and DAGs.")
        entries = nest_entries(block)
        if len(entries) > 1:
            check_kernel_per_state(sdfg)  # raises with the A6 message
        if len(entries) == 1:
            kernels.append(KernelState(state=block, map_entry=entries[0], index=len(kernels)))
        elif state_does_work(block) and not is_relayout_state(block):
            raise NotImplementedError(f"line_graph: state '{block.label}' does non-map work at top level -- v1 scores "
                                      f"map nests only (expand/canonicalize the state first).")
        block = out_edges[0].dst if out_edges else None
    if len(seen) != sdfg.number_of_nodes():
        unreached = {b.label for b in sdfg.nodes()} - {b.label for b in seen}
        raise NotImplementedError(f"line_graph: unreachable states {sorted(unreached)} -- not a line.")
    return kernels
