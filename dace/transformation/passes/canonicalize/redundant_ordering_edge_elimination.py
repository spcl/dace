# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop ordering edges whose happens-before is already implied by another path."""

import collections
from typing import Any

from dace import SDFG
from dace.memlet import Memlet
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.route_ordering_through_data import (
    StateFacts,
    blocking_ordering_edges,
    guards_nothing,
)

#: One state edge, as the tuple needed to re-add it unchanged.
EdgeTuple = tuple[nodes.Node, str | None, nodes.Node, str | None, Memlet]


def is_ordering_edge(edge: MultiConnectorEdge[Memlet]) -> bool:
    """Is ``edge`` a pure ordering edge, i.e. an empty memlet on bare connectors?"""
    return edge.data is not None and edge.data.is_empty() and edge.src_conn is None and edge.dst_conn is None


def candidate_edges(state: SDFGState) -> list[MultiConnectorEdge[Memlet]]:
    """Ordering edges of ``state``, in the pass's canonical processing order."""
    if not any(is_ordering_edge(edge) for edge in state.edges()):
        return []  # the common case: nothing to order, so no sort and no reachability
    topo = {id(node): i for i, node in enumerate(sdutil.dfs_topological_sort(state))}
    if len(topo) != state.number_of_nodes():
        return []  # cyclic or unreachable node: no canonical order to speak of
    ranked = [(topo[id(e.src)], topo[id(e.dst)], i, e) for i, e in enumerate(state.edges()) if is_ordering_edge(e)]
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return [entry[3] for entry in ranked]


def reaches(state: SDFGState, src: nodes.Node, dst: nodes.Node, ignored: OrderedSet[int]) -> bool:
    """Is ``dst`` reachable from ``src`` over the state's edges, ignoring ``ignored``?"""
    visited = OrderedSet([src])
    queue = collections.deque([src])
    while queue:
        for edge in state.out_edges(queue.popleft()):
            if id(edge) in ignored:
                continue
            if edge.dst is dst:
                return True
            if edge.dst not in visited:
                visited.add(edge.dst)
                queue.append(edge.dst)
    return False


def scope_identity(state: SDFGState) -> dict[int, int | None] | None:
    """Object-identity view of ``state.scope_dict()``, for before/after comparison."""
    try:
        scopes = state.scope_dict()
    except (RuntimeError, ValueError):
        return None
    return {id(node): (None if scope is None else id(scope)) for node, scope in scopes.items()}


def restore_edges(state: SDFGState, snapshot: list[EdgeTuple]) -> None:
    """Put ``state``'s edge list back exactly as ``snapshot`` recorded it."""
    for edge in list(state.edges()):
        state.remove_edge(edge)
    for src, src_conn, dst, dst_conn, memlet in snapshot:
        state.add_edge(src, src_conn, dst, dst_conn, memlet)


def is_order_hub(node: nodes.Node) -> bool:
    """Whether ``node`` is an ``order_hub`` tasklet :meth:`SDFGState.keep_order_through` inserted."""
    return (
        isinstance(node, nodes.Tasklet)
        and node.label == "order_hub"
        and not node.in_connectors
        and not node.out_connectors
        and not node.code.as_string.strip()
    )


def dissolve_hubs(state: SDFGState) -> int:
    """Remove every order hub of ``state`` that orders nothing another path does not; one with a single predecessor or
    successor becomes direct edges."""
    removed = 0
    for hub in [n for n in state.nodes() if is_order_hub(n)]:
        ins, outs = state.in_edges(hub), state.out_edges(hub)
        ignored = OrderedSet(id(e) for e in [*ins, *outs])
        if all(reaches(state, i.src, o.dst, ignored) for i in ins for o in outs):
            state.detach_node(hub)
        elif len(ins) == 1 or len(outs) == 1:
            state.remove_node(hub)
        else:
            continue
        removed += 1
    return removed


def drop_unguarded_nest_edges(state: SDFGState) -> int:
    """Remove the ordering edges of each nested SDFG in ``state`` that ``InlineSDFG`` refuses, when none of them guards
    anything: the nest becomes inlinable."""
    blocked = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and any(blocking_ordering_edges(state, n))]
    if not blocked:
        return 0
    facts = StateFacts(state)
    removed = 0
    for node in blocked:
        edges = [e for side in blocking_ordering_edges(state, node) for e in side]
        snapshot: list[EdgeTuple] | None = None
        # One at a time against the current graph: parallel copies of one edge each guard nothing while the others stay
        for edge in edges:
            if not guards_nothing(state, edge, facts):
                if snapshot is not None:
                    restore_edges(state, snapshot)
                break
            if snapshot is None:
                snapshot = [(e.src, e.src_conn, e.dst, e.dst_conn, e.data) for e in state.edges()]
            state.remove_edge(edge)
        else:
            removed += len(edges)
    return removed


def reduce_state(state: SDFGState) -> int:
    """Remove every redundant ordering edge of one state."""
    candidates = candidate_edges(state)
    if not candidates:
        return 0
    rank = {id(edge): r for r, edge in enumerate(candidates)}
    # Both are taken lazily, on the first removal: until then the state is untouched and the
    # rollback material would be paid for by every state that turns out to have nothing to remove.
    snapshot: list[EdgeTuple] | None = None
    before: dict[int, int | None] | None = None

    removed = 0
    for r, edge in enumerate(candidates):
        src, dst = edge.src, edge.dst
        # Scope membership follows the in-edges: a node that loses its last one becomes a source of the state and is
        # re-attributed to the global scope, which is how a node ends up floating outside the map it belongs to.
        if state.in_degree(dst) < 2 or state.degree(src) < 2:
            continue
        ignored = OrderedSet([id(edge)])
        for parallel in state.edges_between(src, dst):
            if rank.get(id(parallel), r) > r:
                ignored.add(id(parallel))
        if not reaches(state, src, dst, ignored):
            continue
        if snapshot is None:
            snapshot = [(e.src, e.src_conn, e.dst, e.dst_conn, e.data) for e in state.edges()]
            before = scope_identity(state)
            if before is None:
                return 0
        state.remove_edge(edge)
        removed += 1

    if snapshot is not None and scope_identity(state) != before:
        restore_edges(state, snapshot)
        return 0
    return removed


@transformation.explicit_cf_compatible
class RedundantOrderingEdgeElimination(ppl.Pass):
    """Remove ordering (empty-memlet) edges implied by another path in the same state, and nested SDFGs' ordering
    edges that guard nothing."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # State fusion merges two states into one, which is exactly when an ordering edge
        # that was necessary in isolation becomes implied by the merged dataflow.
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes | ppl.Modifies.States))

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Reduce every state of ``sdfg`` and of its nested SDFGs."""
        count = 0
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                count += dissolve_hubs(state) + drop_unguarded_nest_edges(state) + reduce_state(state)
        return count or None
