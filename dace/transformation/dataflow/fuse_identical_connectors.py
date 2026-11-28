# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse the connectors of a nested SDFG that bind the same outer container into one connector."""

from dace import SDFG, Memlet, data, properties, subsets, symbolic
from dace.sdfg import graph, nodes, utils
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState, SymbolResolver
from dace.transformation import helpers as xfh
from dace.transformation import transformation as pm


def aliased_container(state: SDFGState, node: nodes.NestedSDFG, conn: str) -> tuple[str, str | None] | None:
    """The outer container ``conn`` stands for as a whole and the accumulation its write carries, or ``None`` if it is
    not one.

    Under the nested SDFG contract a connector whose descriptor is the outer container's own addresses that container
    in its own coordinates, whatever subset its edge covers (``InlineSDFG.inlinable_shared_connectors``); codegen
    passes it the container's base pointer. Two such connectors of one container are the same memory.
    """
    in_edges = list(state.in_edges_by_connector(node, conn))
    out_edges = list(state.out_edges_by_connector(node, conn))
    edges = in_edges + out_edges
    names = dict.fromkeys(e.data.data for e in edges)
    if len(in_edges) > 1 or len(out_edges) > 1 or len(names) != 1:
        return None
    if any(e.data.is_empty() or e.data.other_subset is not None for e in edges):
        return None
    # Only a write-only connector may accumulate: it merges with connectors accumulating the same way
    wcr = out_edges[0].data.wcr if out_edges else None
    if wcr is not None and in_edges:
        return None
    outer_name = next(iter(names))
    inner = node.sdfg.arrays.get(conn)
    outer = state.sdfg.arrays[outer_name]
    if any(not isinstance(d, data.Array) or isinstance(d, (data.View, data.Reference)) for d in (inner, outer)):
        return None
    if inner.transient or list(inner.offset) != list(outer.offset):
        return None
    return (outer_name, wcr) if inner.is_equivalent(outer, symbol_mapping=node.symbol_mapping) else None


def aliased_groups(state: SDFGState, node: nodes.NestedSDFG) -> list[list[str]]:
    """The connectors of ``node`` grouped by the container they alias, kept connector first; only groups of two or
    more."""
    groups: dict[tuple[str, str | None], list[str]] = {}
    for conn in dict.fromkeys([*node.in_connectors, *node.out_connectors]):
        key = aliased_container(state, node, conn)
        if key is not None:
            groups.setdefault(key, []).append(conn)
    # The connector named after its container is kept, so a renamed form reads like its caller
    return [sorted(conns, key=lambda c: c != key[0]) for key, conns in groups.items() if len(conns) > 1]


def union_is_exact(
    merged: subsets.Subset, first: subsets.Subset, second: subsets.Subset, facts: symbolic.Facts
) -> bool:
    """Whether ``merged`` holds no element outside ``first`` and ``second``."""
    if first == second:
        return True
    if subsets.intersects(first, second, facts) is not symbolic.Truth.FALSE:
        return False
    return symbolic.equal(merged.num_elements(), first.num_elements() + second.num_elements()) is True


def fold_edge(
    state: SDFGState, node: nodes.NestedSDFG, kept: MultiConnectorEdge[Memlet], edge: MultiConnectorEdge[Memlet]
) -> None:
    """Fold ``edge`` into ``kept``, which binds the same container on the same side of ``node``."""
    is_input = edge.dst is node
    facts = SymbolResolver().facts_at(state, node)
    memlet = kept.data
    merged = subsets.union(memlet.subset, edge.data.subset, facts)
    if merged is None:
        merged = subsets.Range.from_array(state.sdfg.arrays[memlet.data])
    # A bounding box over a gap would claim writes to elements neither edge writes
    may_write_less = not is_input and not union_is_exact(merged, memlet.subset, edge.data.subset, facts)
    memlet.subset = merged
    memlet.volume = memlet.volume + edge.data.volume
    memlet.dynamic = memlet.dynamic or edge.data.dynamic
    xfh.drop_folded_boundary_edge(state.sdfg, state, edge, kept, node, is_input)
    if may_write_less:
        for path_edge in state.memlet_path(kept):
            path_edge.data.dynamic = True


def merge_connector(state: SDFGState, node: nodes.NestedSDFG, kept: str, dropped: str) -> None:
    """Bind ``dropped`` through ``kept`` on both sides of ``node`` and name its inner array ``kept``."""
    for is_input in (True, False):
        edges_of = state.in_edges_by_connector if is_input else state.out_edges_by_connector
        connectors = node.in_connectors if is_input else node.out_connectors
        edge = next(edges_of(node, dropped), None)
        if edge is None:
            continue
        kept_edge = next(edges_of(node, kept), None)
        dtype = connectors[dropped]
        if kept_edge is not None:
            fold_edge(state, node, kept_edge, edge)
        elif is_input:
            edge.dst_conn = kept
        else:
            edge.src_conn = kept
        if is_input:
            node.remove_in_connector(dropped)
            node.add_in_connector(kept, dtype, force=True)
        else:
            node.remove_out_connector(dropped)
            node.add_out_connector(kept, dtype, force=True)
    node.sdfg.remove_data(dropped, validate=False)
    node.sdfg.replace_dict({dropped: kept})


@properties.make_properties
class FuseIdenticalConnectors(pm.SingleStateTransformation):
    """Fuse the connectors of a nested SDFG that bind the same outer container, whole, into one.

    A nested SDFG reading ``A`` through ``A`` and ``A_0`` renders as a helper taking ``A`` twice, as two pointers to
    the same memory. Each such pair becomes one connector: a second edge on the same side folds into the first (its
    subset joins the kept one), an edge on the other side moves to the kept connector (an in and an out connector
    become one in/out pair), and the inner array of the dropped connector is renamed to the kept one. A connector
    whose descriptor is a window of its container is left alone: its indices are relative to the window, so its inner
    array is not the container. Runs after ``PruneConnectors``, which drops the connectors nothing reads.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls) -> list[graph.OrderedDiGraph[nodes.Node, None]]:
        return [utils.node_path_graph(cls.nsdfg)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        return len(aliased_groups(graph, self.nsdfg)) > 0

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        node = self.nsdfg
        for kept, *dropped in aliased_groups(graph, node):
            for name in dropped:
                merge_connector(graph, node, kept, name)
