# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop, or move onto the access nodes it reads and writes, the ordering edges of a nested SDFG.

``InlineSDFG`` refuses a nested SDFG with an empty (ordering) memlet to or from anything but a scope node,
since nothing inside the nest would carry it once the node is gone. The frontend leaves such edges for
write-after-read order (CloudSC: ``where(zqxfg[2] > 0, ...)`` read by a nest before a later store to
``zqxfg``), and the nest then survives canonicalization as a whole-array black box at the top of a loop
body, where it blocks every dependence test and the team hoist.

An ordering edge whose removal unorders no two conflicting accesses -- no pair of nodes it alone
orders where one writes what the other touches -- is dropped (:func:`guards_nothing`). For the rest,
the nest's data inputs and outputs are access nodes, so the same order is expressed by edges from the
access nodes the nest writes to what had to follow it, and from what had to precede it to the access
nodes it reads: the nest still runs between them, and inlining keeps those access nodes. That reroute
is refused where it would close a cycle.
"""

import collections
from typing import Any

from dace import SDFG, Memlet, data
from dace import graphlib as nx
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def blocking_ordering_edges(
    state: SDFGState, node: nodes.NestedSDFG
) -> tuple[list[MultiConnectorEdge[Memlet]], list[MultiConnectorEdge[Memlet]]]:
    """The empty in- and out-edges of ``node`` that ``InlineSDFG`` refuses: not from or to a scope node."""
    ins = [e for e in state.in_edges(node) if e.data.is_empty() and not isinstance(e.src, nodes.EntryNode)]
    outs = [e for e in state.out_edges(node) if e.data.is_empty() and not isinstance(e.dst, nodes.ExitNode)]
    return ins, outs


def reachable(
    state: SDFGState, start: nodes.Node, forward: bool, skipped: MultiConnectorEdge[Memlet] | None
) -> set[nodes.Node]:
    """The nodes reachable from ``start`` (``forward``) or reaching it (backward), not using ``skipped``."""
    seen = {start}
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        for edge in state.out_edges(current) if forward else state.in_edges(current):
            if edge is skipped:
                continue
            nxt = edge.dst if forward else edge.src
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def viewed_data(state: SDFGState) -> dict[str, str | None]:
    """The data each view of ``state`` aliases, by name; ``None`` where the chain does not resolve."""
    roots: dict[str, str | None] = {}
    for node in state.data_nodes():
        if isinstance(node.desc(state.sdfg), data.View):
            root = sdutil.get_last_view_node(state, node)
            roots[node.data] = root.data if root is not None else None
    return roots


def accesses(state: SDFGState, node: nodes.Node, roots: dict[str, str | None]) -> tuple[set[str], set[str]] | None:
    """``(touched, written)`` data of ``node``, views resolved to what they alias; ``None`` where a view
    does not resolve. An access node touches its own data and writes it where a data edge other than a
    view's binding comes in; a map entry only reads (its out-edges carry its inputs into the scope); any
    other node writes along its out-edges."""
    if isinstance(node, nodes.AccessNode):
        binding = sdutil.get_view_edge(state, node) if node.data in roots else None
        carrying = [e for e in state.all_edges(node) if not e.data.is_empty()]
        names = [node.data] if carrying else []
        written = [node.data] if any(e.dst is node and e is not binding for e in carrying) else []
    else:
        names = [e.data.data for e in state.all_edges(node) if not e.data.is_empty()]
        writes = [] if isinstance(node, nodes.EntryNode) else state.out_edges(node)
        written = [e.data.data for e in writes if not e.data.is_empty()]
    resolved = {name: roots.get(name, name) for name in names}
    if None in resolved.values():
        return None
    return set(resolved.values()), {resolved[name] for name in written}


def conflict(left: tuple[set[str], set[str]] | None, right: tuple[set[str], set[str]] | None) -> bool:
    """Whether two ``(touched, written)`` access sets must stay ordered: one writes what the other touches.
    An unresolved side conflicts with everything."""
    if left is None or right is None:
        return True
    return bool(left[1] & right[0] or right[1] & left[0])


def guards_nothing(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> bool:
    """Whether removing the ordering ``edge`` unorders no two nodes whose accesses conflict.

    Removing ``u -> v`` unorders a node before (or at) ``u`` from a node after (or at) ``v`` exactly when
    every path between them used the edge. Such a pair conflicts when one writes data the other reads or
    writes, a view standing for the data it aliases; an access node counts as touching its data, since
    the code it feeds may be on either side.
    """
    if state.degree(edge.src) < 2 or state.degree(edge.dst) < 2:
        return False  # the edge is all that keeps an endpoint in the state
    roots = viewed_data(state)
    after = reachable(state, edge.dst, True, None)
    for before in reachable(state, edge.src, False, None):
        lost = after - reachable(state, before, True, edge)
        if lost and any(conflict(accesses(state, before, roots), accesses(state, node, roots)) for node in lost):
            return False
    return True


def route_through_data(state: SDFGState, node: nodes.NestedSDFG) -> bool:
    """Drop ``node``'s blocking ordering edges that guard nothing, and replace the others by edges through
    its data access nodes if that keeps the state acyclic.

    :returns: Whether anything was dropped or rerouted.
    """
    ins, outs = blocking_ordering_edges(state, node)
    dropped = [edge for edge in [*ins, *outs] if guards_nothing(state, edge)]
    for edge in dropped:
        state.remove_edge(edge)
    ins, outs = blocking_ordering_edges(state, node)
    if not ins and not outs:
        return bool(dropped)
    reads = [e.src for e in state.in_edges(node) if not e.data.is_empty()]
    writes = [e.dst for e in state.out_edges(node) if not e.data.is_empty()]
    if not all(isinstance(n, nodes.AccessNode) for n in reads + writes):
        return bool(dropped)
    if (outs and not writes) or (ins and not reads):
        return bool(dropped)
    graph = state._nx
    if any(nx.has_path(graph, e.dst, w) for e in outs for w in writes):
        return bool(dropped)
    if any(nx.has_path(graph, r, e.src) for e in ins for r in reads):
        return bool(dropped)
    for edge in outs:
        state.remove_edge(edge)
        for written in writes:
            state.add_nedge(written, edge.dst, Memlet())
    for edge in ins:
        state.remove_edge(edge)
        for read in reads:
            state.add_nedge(edge.src, read, Memlet())
    return True


@transformation.explicit_cf_compatible
class RouteOrderingThroughData(ppl.Pass):
    """Drop or reroute the ordering edges that keep nested SDFGs from being inlined; see the module docstring."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges))

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Drop or reroute every nested SDFG's blocking ordering edges in ``sdfg`` and its nests.

        :returns: How many nested SDFGs changed, or ``None`` if none.
        """
        rerouted = 0
        for owner in sdfg.all_sdfgs_recursive():
            for state in owner.states():
                for node in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    rerouted += route_through_data(state, node)
        return rerouted or None
