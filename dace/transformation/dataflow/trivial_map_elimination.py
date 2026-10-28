# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the trivial-map-elimination transformation."""

import dace
from dace.memlet import Memlet
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@make_properties
class TrivialMapElimination(transformation.SingleStateTransformation):
    """Implements the Trivial-Map Elimination pattern.

    Trivial-Map Elimination removes all dimensions containing only one
    element from a map. If this applies to all ranges the map is removed.
    Example: Map[i=0:I,j=7] -> Map[i=0:I]
    Example: Map[i=0  ,j=7] -> nothing

    There are some special cases:
    - GPU maps are ignored as they are syntactically needed.
    - If all map ranges are trivial and the map has dynamic map ranges,
        the map is not removed, and one map parameter is retained.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        if map_entry.map.schedule in dace.dtypes.GPU_SCHEDULES:
            return False
        if not any(r[0] == r[1] for r in map_entry.map.range):
            return False
        if (map_entry.map.get_param_num()) == 1 and (
            any(not e.dst_conn.startswith("IN_") for e in graph.in_edges(map_entry) if not e.data.is_empty())
        ):
            # There is only one map parameter and there are dynamic map ranges, this can not be resolved.
            return False
        return True

    def apply(self, graph, sdfg):
        map_entry = self.map_entry

        remaining_ranges = []
        remaining_params = []
        scope = graph.scope_subgraph(map_entry)
        for map_param, ranges in zip(map_entry.map.params, map_entry.map.range.ranges):
            map_from, map_to, _ = ranges
            if map_from == map_to:
                # Replace the map index variable with the value it obtained
                scope.replace(map_param, map_from)
            else:
                remaining_ranges.append(ranges)
                remaining_params.append(map_param)

        map_entry.map.range = remaining_ranges
        map_entry.map.params = remaining_params

        if len(remaining_params) != 0:
            # There are still some dimensions left, so no need to remove the map
            pass

        elif any(not e.dst_conn.startswith("IN_") for e in graph.in_edges(map_entry) if not e.data.is_empty()):
            # The map has dynamic map ranges, thus we can not remove the map.
            #  Instead we add one dimension back to keep the SDFG valid.
            map_entry.map.params = [map_param]
            map_entry.map.range = [ranges]

        else:
            # The map is empty and there are no dynamic map ranges.
            self.remove_empty_map(graph, sdfg)

    def remove_empty_map(self, graph, sdfg):
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        ordering = self.ordering_pairs(graph, map_entry, map_exit)
        conflicts = scope_conflicts(graph, map_entry, map_exit)

        # Redirect map entry's out edges
        write_only_map = True
        for edge in graph.out_edges(map_entry):
            if edge.data.is_empty():
                continue
            # Add an edge directly from the previous source connector to the destination
            path = graph.memlet_path(edge)
            index = path.index(edge)
            graph.add_edge(path[index - 1].src, path[index - 1].src_conn, edge.dst, edge.dst_conn, edge.data)
            write_only_map = False

        # Redirect map exit's in edges.
        for edge in graph.in_edges(map_exit):
            path = graph.memlet_path(edge)
            index = path.index(edge)

            # Add an edge directly from the source to the next destination connector
            if len(path) > index + 1:
                graph.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)
                if write_only_map:
                    outer_exit = path[index + 1].dst
                    outer_entry = graph.entry_node(outer_exit)
                    if outer_entry is not None:
                        graph.add_edge(outer_entry, None, edge.src, None, Memlet())

        for src, dst in ordering:
            if not graph.edges_between(src, dst):
                graph.add_edge(src, None, dst, None, Memlet())

        # Remove map
        graph.remove_nodes_from([map_entry, map_exit])
        # The scope ordered its whole body after every input and before every output; where a body node conflicts
        # with what precedes an input (or follows an output), that order now needs an edge of its own
        for src, dst in conflicts:
            if not graph.edges_between(src, dst):
                graph.add_edge(src, None, dst, None, Memlet())

    @staticmethod
    def ordering_pairs(graph, map_entry, map_exit):
        """Return the (src, dst) pairs that keep the order the scope nodes impose after they are removed. An empty
        edge that ends at the entry or starts at the exit exists only to order the whole body, and a body node that
        the entry reaches only by an empty edge runs after everything the entry waits for. Both orderings vanish
        with the scope nodes unless they are re-attached to the body."""
        preds = [e.src for e in graph.in_edges(map_entry)]
        succs = [e.dst for e in graph.out_edges(map_exit)]
        heads = [e.dst for e in graph.out_edges(map_entry) if e.dst is not map_exit]
        tails = [e.src for e in graph.in_edges(map_exit) if e.src is not map_entry]
        pairs = []
        for e in graph.in_edges(map_entry):
            if e.data.is_empty():
                pairs += [(e.src, head) for head in heads]
        for e in graph.out_edges(map_entry):
            if e.data.is_empty() and e.dst is not map_exit:
                pairs += [(pred, e.dst) for pred in preds]
        for e in graph.out_edges(map_exit):
            if e.data.is_empty():
                pairs += [(tail, e.dst) for tail in tails]
        for e in graph.in_edges(map_exit):
            if e.data.is_empty() and e.src is not map_entry:
                pairs += [(e.src, succ) for succ in succs]
        return [
            (src, dst) for src, dst in pairs if src not in (map_entry, map_exit) and dst not in (map_entry, map_exit)
        ]


def accessed(graph, node) -> tuple[set[str], set[str]]:
    """The containers ``node`` reads and writes through its edges."""
    if isinstance(node, nodes.AccessNode):
        return ({node.data} if graph.out_degree(node) else set()), ({node.data} if graph.in_degree(node) else set())
    reads = {e.data.data for e in graph.in_edges(node) if e.data.data is not None}
    writes = {e.data.data for e in graph.out_edges(node) if e.data.data is not None}
    return reads, writes


def conflicting(reads, writes, other_reads, other_writes) -> bool:
    """Whether two accesses race: either writes what the other reads or writes."""
    return bool(writes & (other_reads | other_writes) or reads & other_writes)


def closure(graph, sources, forward: bool) -> dict:
    """``sources`` and every node they reach (``forward``) or that reaches one of them, in visit order."""
    seen = dict.fromkeys(sources)
    stack = list(seen)
    while stack:
        current = stack.pop()
        for edge in graph.out_edges(current) if forward else graph.in_edges(current):
            other = edge.dst if forward else edge.src
            if other not in seen:
                seen[other] = None
                stack.append(other)
    return seen


def propagate(graph, region: dict, seeds: dict, forward: bool) -> dict:
    """Per node of ``region``, the OR of the ``seeds`` bitmasks of every region node on a path to it within
    ``region`` (along edges if ``forward``, against them otherwise)."""
    mask = {n: seeds.get(n, 0) for n in region}
    pending = dict.fromkeys(region, 0)
    for n in region:
        for edge in graph.out_edges(n) if forward else graph.in_edges(n):
            other = edge.dst if forward else edge.src
            if other in pending:
                pending[other] += 1
    ready = [n for n, count in pending.items() if not count]
    while ready:
        current = ready.pop()
        for edge in graph.out_edges(current) if forward else graph.in_edges(current):
            other = edge.dst if forward else edge.src
            if other in pending:
                mask[other] |= mask[current]
                pending[other] -= 1
                if not pending[other]:
                    ready.append(other)
    return mask


def scope_level_groups(graph, map_entry, map_exit) -> tuple[list, dict]:
    """``(head, tail, reads, writes)`` per node at ``map_entry``'s own scope level, plus every body node: a nested
    scope is one group headed by its entry and tailed by its exit, reading and writing what any node inside it does."""
    children = graph.scope_children()
    groups = []
    body = {}
    for child in children[map_entry]:
        if child is map_exit:
            continue
        body[child] = None
        reads, writes = accessed(graph, child)
        tail = child
        if isinstance(child, nodes.EntryNode):
            tail = graph.exit_node(child)
            stack = [child]
            while stack:
                for inner in children[stack.pop()]:
                    body[inner] = None
                    r, w = accessed(graph, inner)
                    reads |= r
                    writes |= w
                    if isinstance(inner, nodes.EntryNode):
                        stack.append(inner)
        groups.append((child, tail, reads, writes))
    return groups, body


def boundary_seeds(graph, scope_node, body: dict, entry: bool) -> tuple[list, dict]:
    """The scope's inputs (``entry``) or outputs, and per body node at the boundary the bitmask of those it is joined
    to once the scope node is removed: by its reconnected data edge, and by the empty-edge orderings that
    ``ordering_pairs`` re-attaches to every body node next to the boundary."""
    outer = graph.in_edges(scope_node) if entry else graph.out_edges(scope_node)
    ends = list(dict.fromkeys(e.src if entry else e.dst for e in outer))
    bit = {n: 1 << i for i, n in enumerate(ends)}
    every = (1 << len(ends)) - 1
    ordered = 0
    for e in outer:
        if e.data.is_empty():
            ordered |= bit[e.src if entry else e.dst]
    seeds = {}
    for e in graph.out_edges(scope_node) if entry else graph.in_edges(scope_node):
        inner = e.dst if entry else e.src
        if inner not in body:
            continue
        mask = ordered
        if e.data.is_empty():
            mask = every
        else:
            path = graph.memlet_path(e)
            index = path.index(e)
            if entry and index > 0:
                mask |= bit[path[index - 1].src]
            elif not entry and index + 1 < len(path):
                mask |= bit[path[index + 1].dst]
        seeds[inner] = seeds.get(inner, 0) | mask
    return ends, seeds


def accesses_of(graph, names: set[str], skip: dict) -> dict:
    """``node -> (reads, writes)`` restricted to ``names``, for every node of ``graph`` outside ``skip`` touching one,
    as :func:`accessed` gives it, in one pass over the edges."""
    found = {}
    for edge in graph.edges():
        src, dst, name = edge.src, edge.dst, edge.data.data
        # An access node reads its own container where an edge leaves it and writes it where one enters; any other
        # node reads what its in-edges carry and writes what its out-edges carry
        if src not in skip:
            if isinstance(src, nodes.AccessNode):
                if src.data in names:
                    found.setdefault(src, (set(), set()))[0].add(src.data)
            elif name in names:
                found.setdefault(src, (set(), set()))[1].add(name)
        if dst not in skip:
            if isinstance(dst, nodes.AccessNode):
                if dst.data in names:
                    found.setdefault(dst, (set(), set()))[1].add(dst.data)
            elif name in names:
                found.setdefault(dst, (set(), set()))[0].add(name)
    return found


def scope_conflicts(graph, map_entry, map_exit) -> list:
    """``(src, dst)`` orders the scope imposes through its data edges that touch a conflicting container: an input
    before a body group that conflicts with a node preceding the input, and a body group before an output that
    conflicts with a node following it. Body nodes are taken at the scope's own level, so no order crosses into a
    nested scope, and a pair is listed only when no path through the reconnected edges orders it already."""
    groups, body = scope_level_groups(graph, map_entry, map_exit)
    body_reads, body_writes = set(), set()
    for _, _, r, w in groups:
        body_reads |= r
        body_writes |= w
    scope = dict(body)
    scope[map_entry] = scope[map_exit] = None
    candidates = {
        node: (reads, writes)
        for node, (reads, writes) in accesses_of(graph, body_reads | body_writes, scope).items()
        if conflicting(reads, writes, body_reads, body_writes)
    }
    if not candidates:
        return []

    sides = [boundary_seeds(graph, map_entry, body, True), boundary_seeds(graph, map_exit, body, False)]
    pairs = []
    for entry, (ends, seeds) in zip((True, False), sides):
        inside = propagate(graph, body, seeds, forward=entry)
        bit = {n: 1 << i for i, n in enumerate(ends)}
        # An input never follows an output (nor an output precedes an input), and an end its own reconnected edge
        # already orders before (after) every group it conflicts with needs no walk
        others = dict.fromkeys(sides[1][0] if entry else sides[0][0])
        pending = {
            node: access
            for node, access in candidates.items()
            if node not in others
            and not (
                node in bit
                and all(
                    bit[node] & inside[head if entry else tail]
                    for head, tail, r, w in groups
                    if conflicting(*access, r, w)
                )
            )
        }
        if not pending:
            continue
        outside = closure(graph, ends, forward=not entry)
        joined = propagate(graph, outside, bit, forward=not entry)
        for node, (reads, writes) in pending.items():
            if node not in outside:
                continue
            for head, tail, r, w in groups:
                rep = head if entry else tail
                if conflicting(reads, writes, r, w) and not joined[node] & inside[rep]:
                    lowest = joined[node] & -joined[node]
                    end = ends[lowest.bit_length() - 1]
                    pairs.append((end, rep) if entry else (rep, end))
                    inside[rep] |= lowest
    return pairs
