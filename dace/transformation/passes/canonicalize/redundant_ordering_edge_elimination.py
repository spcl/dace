# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop ordering edges whose happens-before is already implied by another path.

An ordering edge is an EMPTY memlet (:meth:`dace.memlet.Memlet.is_empty` -- ``data``,
``subset`` and ``other_subset`` all ``None``). It carries no data; it exists only to say
that its source must run before its destination. A data edge says the same thing and
moves data as well, so ordering is implied by *every* edge in a state, not just the empty
ones.

That makes a redundant ordering edge easy to name: ``u -> v`` is redundant when ``v`` is
still reachable from ``u`` over the remaining edges. Deleting it leaves the set of legal
topological orders of the state exactly as it was, which is the whole content of the edge.

Only empty memlets are ever candidates. A zero-``volume`` NAMED memlet (``a[0:0]``) is a
real reader, and ``volume`` is a separate symbolic property rather than the subset size, so
the test is ``is_empty()`` and never ``volume``. Deleting a load-bearing ordering edge does
not fail validation and does not fail to compile -- it silently makes the result depend on
iteration order -- so every rule here errs towards keeping the edge.
"""
import collections
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from dace import SDFG
from dace.memlet import Memlet
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: One state edge, as the tuple needed to re-add it unchanged.
EdgeTuple = Tuple[nodes.Node, Optional[str], nodes.Node, Optional[str], Memlet]


def is_ordering_edge(edge: MultiConnectorEdge[Memlet]) -> bool:
    """Is ``edge`` a pure ordering edge, i.e. an empty memlet on bare connectors?

    Both connectors must be ``None``: an empty memlet occupying a named connector (as on a
    scope node) would leave that connector dangling behind the removal, which validation
    rejects. Ordering edges are added with ``add_nedge``, so this excludes nothing real.

    :param edge: The state edge to classify.
    :returns: ``True`` if the edge carries ordering only.
    """
    return edge.data is not None and edge.data.is_empty() and edge.src_conn is None and edge.dst_conn is None


def candidate_edges(state: SDFGState) -> List[MultiConnectorEdge[Memlet]]:
    """Ordering edges of ``state``, in the pass's canonical processing order.

    The order is a topological sort of the state, ties broken by the edge's position in the
    state's edge list -- its insertion order. Both halves are needed: a topological sort of
    a state is not unique, and insertion order alone is not a dependence order. Neither
    involves a hash, so the order does not move with ``PYTHONHASHSEED``.

    :param state: The state to scan.
    :returns: The candidate edges, earliest first; empty if the state has none or is not a DAG.
    """
    if not any(is_ordering_edge(edge) for edge in state.edges()):
        return []  # the common case: nothing to order, so no sort and no reachability
    topo = {id(node): i for i, node in enumerate(sdutil.dfs_topological_sort(state))}
    if len(topo) != state.number_of_nodes():
        return []  # cyclic or unreachable node: no canonical order to speak of
    ranked = [(topo[id(e.src)], topo[id(e.dst)], i, e) for i, e in enumerate(state.edges()) if is_ordering_edge(e)]
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return [entry[3] for entry in ranked]


def reaches(state: SDFGState, src: nodes.Node, dst: nodes.Node, ignored: OrderedSet) -> bool:
    """Is ``dst`` reachable from ``src`` over the state's edges, ignoring ``ignored``?

    Reachability runs over ALL edges, data and ordering alike, because a data edge implies
    the same happens-before an ordering edge does.

    :param state: The state to search.
    :param src: The node to start from.
    :param dst: The node to look for.
    :param ignored: ``id()`` of the edges to leave out of the search.
    :returns: ``True`` if a path exists.
    """
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


def scope_identity(state: SDFGState) -> Optional[Dict[int, Optional[int]]]:
    """Object-identity view of ``state.scope_dict()``, for before/after comparison.

    ``scope_dict`` is derived from connectivity, so an edge removal can silently move a node
    out of the map it belongs to -- and it maps node objects to node objects, which compare
    by value. Keying on ``id()`` compares the mapping itself.

    :param state: The state to summarize.
    :returns: ``{id(node): id(enclosing scope node) or None}``, or ``None`` if the scopes no
              longer resolve at all.
    """
    try:
        scopes = state.scope_dict()
    except (RuntimeError, ValueError):
        return None
    return {id(node): (None if scope is None else id(scope)) for node, scope in scopes.items()}


def restore_edges(state: SDFGState, snapshot: List[EdgeTuple]) -> None:
    """Put ``state``'s edge list back exactly as ``snapshot`` recorded it.

    Clearing and re-adding in the recorded order restores insertion order as well as the
    edges themselves, so an aborted run leaves the state bit-identical. The original
    ``Memlet`` objects are re-used, and each still belongs to exactly one edge.

    :param state: The state to roll back.
    :param snapshot: The edge tuples taken before the removals.
    """
    for edge in list(state.edges()):
        state.remove_edge(edge)
    for src, src_conn, dst, dst_conn, memlet in snapshot:
        state.add_edge(src, src_conn, dst, dst_conn, memlet)


def reduce_state(state: SDFGState) -> int:
    """Remove every redundant ordering edge of one state.

    Candidates are tested one at a time against the CURRENT graph and each removal is
    committed before the next test, so two ordering edges that imply each other do not both
    disappear. The survivor is the first-inserted one: when testing a candidate, the parallel
    ordering edges ranked AFTER it are ignored alongside it, so an earlier edge can justify a
    later one and never the other way round. A path of length two or more never runs through
    a direct ``u -> v`` edge, so ignoring those costs no other removal.

    :param state: The state to reduce in place.
    :returns: Number of edges removed.
    """
    candidates = candidate_edges(state)
    if not candidates:
        return 0
    rank = {id(edge): r for r, edge in enumerate(candidates)}
    # Both are taken lazily, on the first removal: until then the state is untouched and the
    # rollback material would be paid for by every state that turns out to have nothing to remove.
    snapshot: Optional[List[EdgeTuple]] = None
    before: Optional[Dict[int, Optional[int]]] = None

    removed = 0
    for r, edge in enumerate(candidates):
        src, dst = edge.src, edge.dst
        # Scope membership follows the in-edges: a node that loses its last one becomes a
        # source of the state and is re-attributed to the global scope, which is how a node
        # ends up floating outside the map it belongs to. An isolated node is rejected by
        # validation outright.
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
    """Remove ordering (empty-memlet) edges implied by another path in the same state."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # State fusion merges two states into one, which is exactly when an ordering edge
        # that was necessary in isolation becomes implied by the merged dataflow.
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes | ppl.Modifies.States))

    def depends_on(self) -> List[Union[Type[ppl.Pass], ppl.Pass]]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Reduce every state of ``sdfg`` and of its nested SDFGs.

        Nested SDFGs are visited as SDFGs of their own: ``scope_dict`` restarts at each
        nesting boundary, so a state is only ever reasoned about on its own terms.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Unused; the pass has no dependencies.
        :returns: Number of ordering edges removed, or ``None`` if none.
        """
        count = 0
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.all_states():
                count += reduce_state(state)
        return count or None
