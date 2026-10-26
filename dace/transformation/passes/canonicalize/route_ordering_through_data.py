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

from typing import Any

from dace import SDFG, Memlet, data
from dace import graphlib as nx
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import helpers, transformation
from dace.transformation import pass_pipeline as ppl


def blocking_ordering_edges(
    state: SDFGState, node: nodes.NestedSDFG
) -> tuple[list[MultiConnectorEdge[Memlet]], list[MultiConnectorEdge[Memlet]]]:
    """The empty in- and out-edges of ``node`` that ``InlineSDFG`` refuses: not from or to a scope node."""
    ins = [e for e in state.in_edges(node) if e.data.is_empty() and not isinstance(e.src, nodes.EntryNode)]
    outs = [e for e in state.out_edges(node) if e.data.is_empty() and not isinstance(e.dst, nodes.ExitNode)]
    return ins, outs


def reach_bits(state: SDFGState, order: list[nodes.Node], index: dict[nodes.Node, int], skipped) -> dict:
    """Per node of ``state``, the bitset (bit ``index[n]``) of the nodes it reaches, not using edge ``skipped``.

    One pass in reverse topological ``order``: a state is acyclic.
    """
    bits: dict[nodes.Node, int] = {}
    for current in reversed(order):
        reach = 0
        for edge in state.out_edges(current):
            if edge is not skipped:
                reach |= (1 << index[edge.dst]) | bits[edge.dst]
        bits[current] = reach
    return bits


class StateFacts:
    """What the decisions in one state read and no drop, reroute or split changes: the data each view aliases and
    each node's accesses. A split leaves every existing node's touched and written data as it was."""

    def __init__(self, state: SDFGState) -> None:
        self.state = state
        self.roots = viewed_data(state)
        self._accesses: dict[nodes.Node, tuple[set[str], set[str]] | None] = {}

    def accesses(self, node: nodes.Node) -> tuple[set[str], set[str]] | None:
        if node not in self._accesses:
            self._accesses[node] = accesses(self.state, node, self.roots)
        return self._accesses[node]


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


def guards_nothing(state: SDFGState, edge: MultiConnectorEdge[Memlet], facts: StateFacts) -> bool:
    """Whether removing the ordering ``edge`` unorders no two nodes whose accesses conflict.

    Removing ``u -> v`` unorders a node before (or at) ``u`` from a node after (or at) ``v`` exactly when
    every path between them used the edge. Such a pair conflicts when one writes data the other reads or
    writes, a view standing for the data it aliases; an access node counts as touching its data, since
    the code it feeds may be on either side.
    """
    if state.degree(edge.src) < 2 or state.degree(edge.dst) < 2:
        return False  # the edge is all that keeps an endpoint in the state
    scope = state.scope_dict()
    if (state.in_degree(edge.dst) == 1 and scope[edge.dst] is not None) or (
        state.out_degree(edge.src) == 1 and scope[edge.src] is not None
    ):
        return False  # the edge is all that keeps an endpoint in its scope
    order = list(nx.topological_sort(state.nx))
    index = {n: i for i, n in enumerate(order)}
    full, without = reach_bits(state, order, index, None), reach_bits(state, order, index, edge)
    after = full[edge.dst] | (1 << index[edge.dst])
    source_bit = 1 << index[edge.src]
    for before in order:
        if before is not edge.src and not full[before] & source_bit:
            continue
        lost = after & ~(without[before] | (1 << index[before]))
        while lost:
            low = lost & -lost
            node = order[low.bit_length() - 1]
            if conflict(facts.accesses(before), facts.accesses(node)):
                return False
            lost ^= low
    return True


def own_access(state: SDFGState, node: nodes.NestedSDFG, access: nodes.AccessNode, outgoing: bool) -> list:
    """``node``'s data edges into (``outgoing``) or out of ``access``, when ``access`` can hand them to an access node
    of its own: a non-view sink (source) with other writers (readers), whose in-edges (out-edges) are unordered among
    themselves. ``[]`` otherwise."""
    if isinstance(access.desc(state.sdfg), data.View):
        return []
    if (state.out_degree(access) if outgoing else state.in_degree(access)) > 0:
        return []
    edges = state.out_edges(node) if outgoing else state.in_edges(node)
    mine = [e for e in edges if not e.data.is_empty() and (e.dst if outgoing else e.src) is access]
    others = state.in_degree(access) if outgoing else state.out_degree(access)
    return mine if others > len(mine) else []


def route_through_data(state: SDFGState, node: nodes.NestedSDFG, facts: StateFacts) -> bool:
    """Drop ``node``'s blocking ordering edges that guard nothing, and replace the others by edges through
    its data access nodes if that keeps the state acyclic.

    An access node that closes the cycle only through its other writers (a shared sink) or readers (a shared source)
    is first split, so that ``node`` gets an access node of its own.

    :returns: Whether anything was dropped or rerouted.
    """
    ins, outs = blocking_ordering_edges(state, node)
    dropped = [edge for edge in [*ins, *outs] if guards_nothing(state, edge, facts)]
    for edge in dropped:
        state.remove_edge(edge)
    ins, outs = blocking_ordering_edges(state, node)
    if not ins and not outs:
        return bool(dropped)
    reads = list(dict.fromkeys(e.src for e in state.in_edges(node) if not e.data.is_empty()))
    writes = list(dict.fromkeys(e.dst for e in state.out_edges(node) if not e.data.is_empty()))
    if not all(isinstance(n, nodes.AccessNode) for n in reads + writes):
        return bool(dropped)
    if (outs and not writes) or (ins and not reads):
        return bool(dropped)
    graph = state.nx
    # An access node the nest owns alone cannot close a cycle; one it shares can, and is split if it may be.
    to_split: list[tuple[nodes.AccessNode, list, bool]] = []
    for written in writes:
        if any(nx.has_path(graph, e.dst, written) for e in outs):
            to_split.append((written, own_access(state, node, written, outgoing=True), True))
    for read in reads:
        if any(nx.has_path(graph, read, e.src) for e in ins):
            to_split.append((read, own_access(state, node, read, outgoing=False), False))
    if any(not edges for _, edges, _ in to_split):
        return bool(dropped)
    for access, edges, outgoing in to_split:
        own = state.add_access(access.data)
        for edge in edges:
            helpers.redirect_edge(state, edge, **({"new_dst": own} if outgoing else {"new_src": own}))
    reads = list(dict.fromkeys(e.src for e in state.in_edges(node) if not e.data.is_empty()))
    writes = list(dict.fromkeys(e.dst for e in state.out_edges(node) if not e.data.is_empty()))
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
        return ppl.Modifies.Edges | ppl.Modifies.AccessNodes

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
                blocked = [
                    n
                    for n in state.nodes()
                    if isinstance(n, nodes.NestedSDFG) and any(blocking_ordering_edges(state, n))
                ]
                facts = StateFacts(state) if blocked else None
                for node in blocked:
                    rerouted += route_through_data(state, node, facts)
        return rerouted or None
