# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse two Maps over the same range into one, keeping every intermediate in memory.

`MapFusionVertical` privatizes the data the first Map hands to the second into a per-iteration buffer. That needs
the first Map to read and write the intermediate point-wise, which a recurrence along a row never does (the first
Map reads ``p[i, j - 1]`` and writes ``p[i, j]``), so it refuses such pairs (spcl/dace#2725). This transformation
does not privatize: the fused Map runs the body of the first Map, then the body of the second, per iteration, both
on the original containers. That is valid when every container both bodies touch, and at least one writes, is
row-private: one dimension of every access to it is exactly the same Map parameter, so no two iterations touch the
same element.

Scope of this version: both Maps are top-level and have the same range. What a body touches is read off the edges
at its scope nodes (the propagated per-iteration subsets), so a body may be anything, a fused body included. Map
parameters named in the second body's code (not only in its memlets and nested-SDFG symbol mappings) are not
renamed, so such a pair is refused.
"""

from __future__ import annotations

import copy

from dace import data, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.propagation import propagate_memlets_state
from dace.transformation import transformation


@make_properties
@transformation.explicit_cf_compatible
class MapFusionInPlace(transformation.SingleStateTransformation):
    """Fuse ``first_map_exit -> array -> second_map_entry`` into one Map without privatizing ``array``."""

    #: Why the last can_be_applied() refused, or None when it accepted.
    refusal: str | None = None

    first_map_exit = transformation.PatternNode(nodes.MapExit)
    array = transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    # -- matching ---------------------------------------------------------------------------------------------

    @staticmethod
    def body(state: SDFGState, entry: nodes.MapEntry) -> list[nodes.Node]:
        """Every node of ``entry``'s scope but its entry and exit."""
        return list(state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes())

    @staticmethod
    def names_params_in_code(body: list[nodes.Node], params: list[str]) -> bool:
        """Whether a tasklet of ``body`` names one of ``params`` in its code (which renaming would not reach)."""
        for node in body:
            if isinstance(node, nodes.Tasklet) and set(params) & set(map(str, node.free_symbols)):
                return True
        return False

    @staticmethod
    def param_replacement(first: nodes.Map, second: nodes.Map) -> dict[str, str] | None:
        """``second``'s parameters renamed to ``first``'s, or ``None`` when the ranges differ."""
        if len(first.params) != len(second.params) or first.range != second.range:
            return None
        return {b: a for a, b in zip(first.params, second.params) if a != b}

    @staticmethod
    def renamed(memlet: Memlet, repl: dict[str, str]) -> Memlet:
        out = copy.deepcopy(memlet)
        if repl:
            out.replace({k: symbolic.symbol(v) for k, v in repl.items()})  # Memlet.replace takes names
        return out

    def body_accesses(
        self, state: SDFGState, entry: nodes.MapEntry, repl: dict[str, str]
    ) -> tuple[dict[str, list[subsets.Subset]], set[str]]:
        """Per container, the subsets ``entry``'s body reads or writes in one iteration (parameters renamed by
        ``repl``), read off the inner edges of its scope nodes, and the containers it writes."""
        accesses: dict[str, list[subsets.Subset]] = {}
        written: set[str] = set()
        for edge in state.out_edges(entry):
            if not edge.data.is_empty():
                accesses.setdefault(edge.data.data, []).append(self.renamed(edge.data, repl).subset)
        for edge in state.in_edges(state.exit_node(entry)):
            if not edge.data.is_empty():
                accesses.setdefault(edge.data.data, []).append(self.renamed(edge.data, repl).subset)
                written.add(edge.data.data)
        return accesses, written

    @staticmethod
    def row_private(subsets_of: list[subsets.Subset], params: list[str]) -> bool:
        """Some dimension of every subset is exactly the same Map parameter, a single index."""
        if not subsets_of or any(not isinstance(s, subsets.Range) for s in subsets_of):
            return False
        dims = {s.dims() for s in subsets_of}
        if len(dims) != 1:
            return False
        for dim in range(dims.pop()):
            for param in params:
                sym = symbolic.symbol(param)
                if all((s[dim][0] - sym) == 0 and (s[dim][1] - sym) == 0 for s in subsets_of):
                    return True
        return False

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        self.refusal = self.refusal_reason(graph, sdfg)
        return self.refusal is None

    def refusal_reason(self, graph: SDFGState, sdfg: SDFG) -> str | None:
        """Why the match cannot be fused in place, or ``None`` when it can."""
        first_exit, second_entry = self.first_map_exit, self.second_map_entry
        scope = graph.scope_dict()
        first_entry = graph.entry_node(first_exit)
        if scope.get(first_entry) is not None or scope.get(second_entry) is not None:
            return "a Map is not at the top level of the state"
        repl = self.param_replacement(first_entry.map, second_entry.map)
        if repl is None:
            return "the Map ranges differ"
        second_exit = graph.exit_node(second_entry)
        if repl and self.names_params_in_code(self.body(graph, second_entry), list(repl)):
            return "the second body names a Map parameter in tasklet code"
        # The second Map reads only through AccessNodes and writes only to AccessNodes.
        if any(not isinstance(e.src, nodes.AccessNode) for e in graph.in_edges(second_entry) if not e.data.is_empty()):
            return "the second Map reads from something other than an AccessNode"
        if any(not isinstance(e.dst, nodes.AccessNode) for e in graph.out_edges(second_exit)):
            return "the second Map writes to something other than an AccessNode"
        # Whatever the first Map writes and the second reads must reach it from the first Map's exit.
        first_writes = {e.data.data for e in graph.out_edges(first_exit) if not e.data.is_empty()}
        for edge in graph.in_edges(second_entry):
            if edge.data.is_empty() or edge.data.data not in first_writes:
                continue
            if not any(e.src is first_exit for e in graph.in_edges(edge.src)):
                return "the second Map reads data the first writes without it reaching it from the first Map's exit"
        # The second Map may depend on the first only through the nodes the first Map's exit writes directly, and
        #  the first Map may not depend on the second at all (the fused Map would close a cycle).
        after_first = set(graph.bfs_nodes(first_exit))
        for edge in graph.in_edges(second_entry):
            if edge.src in after_first and not any(e.src is first_exit for e in graph.in_edges(edge.src)):
                return "the second Map depends on a node downstream of the first other than its direct outputs"
        after_second = set(graph.bfs_nodes(second_exit))
        if any(edge.src in after_second for edge in graph.in_edges(first_entry)):
            return "the first Map depends on the second"
        a_acc, a_w = self.body_accesses(graph, first_entry, {})
        b_acc, b_w = self.body_accesses(graph, second_entry, repl)
        params = list(first_entry.map.params)
        for name in sorted((set(a_acc) & set(b_acc)) & (a_w | b_w)):
            if isinstance(sdfg.arrays[name], data.View):
                return "a shared container is a View"
            if not self.row_private(a_acc[name] + b_acc[name], params):
                return f"'{name}' is not row-private: some access of it does not fix one dimension to the same Map parameter"
        # A container only one of them writes, but both touch, is covered above; one that only one touches is
        #  private to that body already.
        return None

    # -- rewrite ----------------------------------------------------------------------------------------------

    @staticmethod
    def fresh_connector(node: nodes.EntryNode | nodes.ExitNode, stem: str) -> str:
        name, k = stem, 0
        while f"IN_{name}" in node.in_connectors or f"OUT_{name}" in node.out_connectors:
            k += 1
            name = f"{stem}_{k}"
        node.add_in_connector(f"IN_{name}")
        node.add_out_connector(f"OUT_{name}")
        return name

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        first_exit, second_entry = self.first_map_exit, self.second_map_entry
        first_entry = graph.entry_node(first_exit)
        second_exit = graph.exit_node(second_entry)
        repl = self.param_replacement(first_entry.map, second_entry.map)
        sym_repl = {symbolic.symbol(k): symbolic.symbol(v) for k, v in repl.items()}
        first_sinks = {e.src for e in graph.in_edges(first_exit)}
        second_sources = {e.dst for e in graph.out_edges(second_entry)}

        # The second body's parameters become the first Map's: its inner memlets and its nested SDFGs' symbols.
        if repl:
            inner = graph.scope_subgraph(second_entry, include_entry=False, include_exit=False)
            for edge in inner.edges():
                edge.data.replace({k: symbolic.symbol(v) for k, v in repl.items()})
            for node in inner.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    for name, value in list(node.symbol_mapping.items()):
                        node.symbol_mapping[name] = symbolic.pystr_to_symbolic(str(value)).subs(sym_repl)

        produced = {e.dst: e for e in graph.out_edges(first_exit) if isinstance(e.dst, nodes.AccessNode)}
        inner_nodes: dict[str, nodes.AccessNode] = {}

        def inner_access(outer: nodes.AccessNode) -> nodes.AccessNode:
            """An access node of ``outer``'s container inside the fused scope, fed by the first body's write of it."""
            if outer.data in inner_nodes:
                return inner_nodes[outer.data]
            out_edge = produced[outer]
            inner_edge = next(e for e in graph.in_edges(first_exit) if e.dst_conn == "IN_" + out_edge.src_conn[4:])
            node = graph.add_access(outer.data)
            graph.add_edge(inner_edge.src, inner_edge.src_conn, node, None, copy.deepcopy(inner_edge.data))
            graph.add_edge(node, None, first_exit, inner_edge.dst_conn, copy.deepcopy(inner_edge.data))
            graph.remove_edge(inner_edge)
            inner_nodes[outer.data] = node
            return node

        # Inputs of the second body: from the first body's write, or routed through the first Map's entry.
        for edge in list(graph.out_edges(second_entry)):
            outer_edge = (
                next((e for e in graph.in_edges(second_entry) if e.dst_conn == "IN_" + edge.src_conn[4:]), None)
                if edge.src_conn
                else None
            )
            memlet = self.renamed(edge.data, repl)
            if outer_edge is not None and outer_edge.src in produced:
                graph.add_edge(inner_access(outer_edge.src), None, edge.dst, edge.dst_conn, memlet)
            elif outer_edge is not None:
                conn = self.fresh_connector(first_entry, outer_edge.data.data or "in")
                graph.add_edge(
                    outer_edge.src, outer_edge.src_conn, first_entry, f"IN_{conn}", copy.deepcopy(outer_edge.data)
                )
                graph.add_edge(first_entry, f"OUT_{conn}", edge.dst, edge.dst_conn, memlet)
            else:
                graph.add_edge(first_entry, None, edge.dst, edge.dst_conn, Memlet())
            graph.remove_edge(edge)
        # Outputs of the second body: routed through the first Map's exit.
        for edge in list(graph.in_edges(second_exit)):
            outer_edge = (
                next(
                    (e for e in graph.out_edges(second_exit) if e.src_conn == "OUT_" + (edge.dst_conn or "")[3:]), None
                )
                if edge.dst_conn
                else None
            )
            if outer_edge is None:
                graph.add_edge(edge.src, edge.src_conn, first_exit, None, Memlet())
            else:
                conn = self.fresh_connector(first_exit, outer_edge.data.data or "out")
                graph.add_edge(edge.src, edge.src_conn, first_exit, f"IN_{conn}", self.renamed(edge.data, repl))
                graph.add_edge(
                    first_exit, f"OUT_{conn}", outer_edge.dst, outer_edge.dst_conn, copy.deepcopy(outer_edge.data)
                )
            graph.remove_edge(edge)
        # The second body runs after the first in each iteration: order every node the first body ends in before
        #  every node the second body starts from (a write-after-read through memory has no data edge).
        for sink in first_sinks:
            for source in second_sources:
                if sink is not source and sink is not first_entry and source is not second_exit:
                    graph.add_edge(sink, None, source, None, Memlet())

        graph.remove_node(second_entry)
        graph.remove_node(second_exit)
        for node in list(produced):
            if graph.degree(node) == 0:
                graph.remove_node(node)
        for conn in list(first_entry.out_connectors):
            if conn.startswith("OUT_") and not any(e.src_conn == conn for e in graph.out_edges(first_entry)):
                first_entry.remove_out_connector(conn)
                first_entry.remove_in_connector("IN_" + conn[4:])
                for e in [e for e in graph.in_edges(first_entry) if e.dst_conn == "IN_" + conn[4:]]:
                    graph.remove_edge(e)
        propagate_memlets_state(sdfg, graph)
