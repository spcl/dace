# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Move a map's read off the access node of a write that misses it."""

from collections.abc import Iterable
from typing import Any

from dace import data, properties, subsets, symbolic
from dace.sdfg import SDFG, nodes
from dace.sdfg.sdfg import memoize_facts
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class DetachDisjointReads(ppl.Pass):
    """Slices of one array computed in sequence chain their maps through its access nodes, though none needs another::

        t -> M1 -> t[:, 6] -> M2 -> t[:, 7]     t -> M1, M2 -> t[:, 6], t[:, 7]
             (M2 reads t[:, 1])

    The chain is a false dependence: ``M2`` reads nothing ``M1`` wrote, and fusion refuses a map that reads what both
    write (fv3's tracers in ``q``, cloudsc's ``zsolqa``, lulesh's corners). So ``M2`` reads from the node before ``M1``,
    and its write joins that of ``M1`` on one node. Only a map of the writer's range is detached, for fusion to take,
    and only when no order the chain gave is needed: :func:`lost_order_is_free`.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:return: the number of maps detached, or None if there is none."""
        detached = 0
        with memoize_facts(sdfg):
            for nsdfg in sdfg.all_sdfgs_recursive():
                for state in nsdfg.states():
                    detached += detach_in_state(state)
        return detached or None


def detach_in_state(state: SDFGState) -> int:
    """Detach every map of ``state`` that qualifies."""
    candidates = [n for n in state.data_nodes() if state.in_degree(n) and state.out_degree(n)]
    candidates = [n for n in candidates if all(isinstance(e.src, nodes.MapExit) for e in state.in_edges(n))]
    if not candidates:
        return 0
    detached = 0
    scope = state.scope_dict()
    for node in candidates:
        if node not in state.nodes() or scope.get(node) is not None or type(node.desc(state.sdfg)) is not data.Array:
            continue
        while any(detach(state, node, reader) for reader in state.successors(node)):
            detached += 1
    return detached


def stored(state: SDFGState, writes: Iterable[Any]) -> list[subsets.Subset]:
    return [e.data.get_dst_subset(e, state) for e in writes]


def miss(first: list[subsets.Subset], second: list[subsets.Subset], facts: symbolic.Facts) -> bool:
    """Whether nothing of ``first`` overlaps anything of ``second``, provably."""
    return all(subsets.intersects(a, b, facts) is symbolic.Truth.FALSE for a in first for b in second)


def detach(state: SDFGState, node: nodes.AccessNode, reader: nodes.Node) -> bool:
    """Make ``reader``, a map reading ``node``, a sibling of the maps writing ``node``, if it reads none of what they
    wrote. :return: whether it did."""
    if not isinstance(reader, nodes.MapEntry):
        return False
    name = node.data
    writes = state.in_edges(node)
    writers = list(dict.fromkeys(state.entry_node(e.src) for e in writes))
    if any(w.map.range != reader.map.range for w in writers) or any(e.data.wcr is not None for e in writes):
        return False
    reads = [e for e in state.in_edges(reader) if e.data.data == name]
    if any(e.src is not node for e in reads):
        return False
    facts = state.sdfg.facts()
    sources = [e for e in reads if not e.data.is_empty()]
    if not sources or not miss([e.data.get_src_subset(e, state) for e in sources], stored(state, writes), facts):
        return False
    reader_exit = state.exit_node(reader)
    stores = [e for e in state.out_edges(reader_exit) if e.data.data == name]
    targets = list(dict.fromkeys(e.dst for e in stores))
    if len(targets) > 1 or any(e.data.wcr is not None for e in stores):
        return False
    target = targets[0] if targets else None
    before = list(
        dict.fromkeys(
            e.src
            for w in writers
            for e in state.in_edges(w)
            if isinstance(e.src, nodes.AccessNode) and e.data.data == name
        )
    )
    if len(before) > 1:
        return False
    if target is not None:
        reader_writes = stored(state, stores)
        if not miss(stored(state, writes), reader_writes, facts):
            return False
        loads = [e for w in writers for e in state.in_edges(w) if e.data.data == name]
        loads += [e for e in state.out_edges(node) if e.dst is not reader]
        if not miss([e.data.get_src_subset(e, state) for e in loads if not e.data.is_empty()], reader_writes, facts):
            return False
    if not lost_order_is_free(state, node, reader, target, before[0] if before else None, reads, stores):
        return False
    source = before[0] if before else state.add_access(name)
    for edge in reads:
        state.remove_edge(edge)
        state.add_edge(source, None, reader, edge.dst_conn, edge.data)
    if target is not None:
        for edge in state.in_edges(target):
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, node, edge.dst_conn, edge.data)
        for edge in state.out_edges(target):
            state.remove_edge(edge)
            state.add_edge(node, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
        state.remove_node(target)
    return True


def closure(state: SDFGState, start: Iterable[nodes.Node], backward: bool, skip: set[int]) -> set[nodes.Node]:
    """``start`` and all that reaches it (``backward``) or that it reaches, past no edge in ``skip``."""
    seen = set(start)
    stack = list(seen)
    while stack:
        current = stack.pop()
        for edge in state.in_edges(current) if backward else state.out_edges(current):
            other = edge.src if backward else edge.dst
            if id(edge) not in skip and other not in seen:
                seen.add(other)
                stack.append(other)
    return seen


def lost_order_is_free(
    state: SDFGState,
    node: nodes.AccessNode,
    reader: nodes.MapEntry,
    target: nodes.AccessNode | None,
    before: nodes.AccessNode | None,
    reads: list[Any],
    stores: list[Any],
) -> bool:
    """Whether what ran before ``reader`` only through ``node`` shares nothing with what then ran after it, other than
    the accesses of ``node``'s container :func:`detach` proved apart."""
    removed = {id(e) for e in reads}
    earlier = closure(state, [node], True, removed)
    if before is not None:
        earlier -= closure(state, [before], True, removed)
    later = closure(state, [reader], False, removed)
    later -= closure(state, [node] + ([target] if target is not None else []), False, removed)
    exempt = removed | {id(e) for e in stores} | {id(e) for e in state.in_edges(node)}
    exempt |= {id(e) for w in earlier if isinstance(w, nodes.MapEntry) for e in state.in_edges(w) if e.src is before}
    first_reads, first_writes = footprint(state, earlier, exempt)
    second_reads, second_writes = footprint(state, later, exempt)
    if first_reads is None or second_reads is None:
        return False
    if node.data in first_reads | first_writes | second_reads | second_writes:
        return False
    return not (first_writes & (second_reads | second_writes) or first_reads & second_writes)


def footprint(state: SDFGState, among: set[nodes.Node], exempt: set[int]) -> tuple[set[str] | None, set[str]]:
    """The containers read and written across ``among``'s edges to access nodes outside ``exempt``; None for a view,
    which names another container."""
    reads: set[str] = set()
    writes: set[str] = set()
    for edge in state.edges():
        if id(edge) in exempt or edge.data.is_empty():
            continue
        if isinstance(edge.src, nodes.AccessNode) and edge.dst in among:
            reads.add(edge.src.data)
            view = edge.src
        elif isinstance(edge.dst, nodes.AccessNode) and edge.src in among:
            writes.add(edge.dst.data)
            view = edge.dst
        else:
            continue
        if isinstance(view.desc(state.sdfg), data.View):
            return None, writes
    return reads, writes
