"""``PerfLoopNesting`` — duplicate a parent map around each inner child.

Single-state transformation applied at a top-level ``MapEntry``. The
match requires:

- The ``MapEntry``'s body contains exactly one ``NestedSDFG``.
- That ``NestedSDFG`` has exactly one state.
- That state has K >= 2 top-level children -- either ``MapEntry`` nodes
  or ``Tasklet`` nodes outside any map scope.

Apply fissions the parent map into K copies, each wrapping a pruned
``NestedSDFG`` that retains exactly one child's subgraph. Top-level
``Tasklet`` children are first wrapped in a single-iteration trivial map
so every sibling is a ``MapEntry``; this preserves perfect-nesting shape
in the output.

Outer-state edges are re-routed: each duplicate receives only the inputs
its child reads and only the outputs its child writes. Arrays no longer
referenced inside a duplicate's pruned state are dropped from that
``NestedSDFG`` so validation's non-transient-as-connector requirement
holds.
"""
import copy as _copy
from typing import Dict, Set

from dace import memlet as mm, nodes, properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf


@properties.make_properties
class PerfLoopNesting(xf.SingleStateTransformation):
    """Duplicate a parent map so each inner map gets its own wrapper."""

    parent_entry = xf.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.parent_entry)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg: SDFG, permissive=False) -> bool:
        pe = self.parent_entry
        if graph.entry_node(pe) is not None:
            return False
        body = [n for n in graph.nodes() if graph.entry_node(n) is pe]
        nsdfgs = [n for n in body if isinstance(n, nodes.NestedSDFG)]
        if len(nsdfgs) != 1:
            return False
        inner = nsdfgs[0].sdfg
        if len(list(inner.states())) != 1:
            return False
        inner_state = list(inner.states())[0]
        top_children = [
            n for n in inner_state.nodes()
            if inner_state.entry_node(n) is None and isinstance(n, (nodes.MapEntry, nodes.Tasklet))
        ]
        return len(top_children) >= 2

    def apply(self, graph: SDFGState, sdfg: SDFG):
        pe = self.parent_entry
        px = graph.exit_node(pe)
        body = [n for n in graph.nodes() if graph.entry_node(n) is pe]
        orig_nsdfg = next(n for n in body if isinstance(n, nodes.NestedSDFG))
        inner_state = list(orig_nsdfg.sdfg.states())[0]

        for n in list(inner_state.nodes()):
            if isinstance(n, nodes.Tasklet) and inner_state.entry_node(n) is None:
                _wrap_tasklet_in_trivial_map(inner_state, n)

        child_entries = [
            n for n in inner_state.nodes() if isinstance(n, nodes.MapEntry) and inner_state.entry_node(n) is None
        ]
        outer_in_by, outer_out_by = _outer_plumbing(graph, pe, px, orig_nsdfg)

        for ch_entry in child_entries:
            keep = _child_subgraph(inner_state, ch_entry)
            _build_duplicate(graph, pe, px, orig_nsdfg, inner_state, keep, outer_in_by, outer_out_by)

        for ie in list(graph.in_edges(pe)):
            graph.remove_edge(ie)
        for oe in list(graph.out_edges(px)):
            graph.remove_edge(oe)
        for n in body + [pe, px]:
            if n in graph.nodes():
                graph.remove_node(n)


def _copy_state_contents(src: SDFGState, dst: SDFGState) -> Dict[nodes.Node, nodes.Node]:
    """Deep-copy all nodes and edges from ``src`` into ``dst``."""
    nmap: Dict[nodes.Node, nodes.Node] = {}
    for n in src.nodes():
        new_n = _copy.deepcopy(n)
        dst.add_node(new_n)
        nmap[n] = new_n
    for e in src.edges():
        dst.add_edge(nmap[e.src], e.src_conn, nmap[e.dst], e.dst_conn, _copy.deepcopy(e.data))
    return nmap


def _child_subgraph(inner_state: SDFGState, ch_entry: nodes.MapEntry) -> Set[nodes.Node]:
    """Subgraph belonging to one child map: its MapEntry, MapExit, every
    node scoped under the MapEntry, and the top-level AccessNodes directly
    connected to the entry's in-edges or exit's out-edges."""
    ch_exit = inner_state.exit_node(ch_entry)
    keep: Set[nodes.Node] = {ch_entry, ch_exit}
    keep.update(n for n in inner_state.nodes() if inner_state.entry_node(n) is ch_entry)
    for e in inner_state.in_edges(ch_entry):
        if isinstance(e.src, nodes.AccessNode):
            keep.add(e.src)
    for e in inner_state.out_edges(ch_exit):
        if isinstance(e.dst, nodes.AccessNode):
            keep.add(e.dst)
    return keep


def _outer_plumbing(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit, orig_nsdfg: nodes.NestedSDFG):
    in_by: Dict[str, list] = {}
    for e in graph.in_edges(orig_nsdfg):
        n_conn = e.dst_conn
        me_in_conn = e.src_conn.replace("OUT_", "IN_") if e.src_conn else None
        outer_edges = [ie for ie in graph.in_edges(pe) if ie.dst_conn == me_in_conn]
        in_by.setdefault(n_conn, []).extend((ie.src, ie.data, e.data) for ie in outer_edges)
    out_by: Dict[str, list] = {}
    for e in graph.out_edges(orig_nsdfg):
        n_conn = e.src_conn
        mx_out_conn = e.dst_conn.replace("IN_", "OUT_") if e.dst_conn else None
        outer_edges = [oe for oe in graph.out_edges(px) if oe.src_conn == mx_out_conn]
        out_by.setdefault(n_conn, []).extend((oe.dst, oe.data, e.data) for oe in outer_edges)
    return in_by, out_by


def _wrap_tasklet_in_trivial_map(state: SDFGState, t: nodes.Tasklet):
    """Enclose a bare top-level ``Tasklet`` in a single-iteration map."""
    in_edges = list(state.in_edges(t))
    out_edges = list(state.out_edges(t))

    me, mx = state.add_map(f"{t.label}_wrap", {"_p": "0:1"})

    for idx, e in enumerate(in_edges):
        key = e.dst_conn if e.dst_conn else f"in_{idx}"
        in_c = f"IN_{key}"
        out_c = f"OUT_{key}"
        me.add_in_connector(in_c)
        me.add_out_connector(out_c)
        state.remove_edge(e)
        state.add_edge(e.src, e.src_conn, me, in_c, _copy.deepcopy(e.data))
        state.add_edge(me, out_c, t, e.dst_conn, _copy.deepcopy(e.data))

    for idx, e in enumerate(out_edges):
        key = e.src_conn if e.src_conn else f"out_{idx}"
        in_c = f"IN_{key}"
        out_c = f"OUT_{key}"
        mx.add_in_connector(in_c)
        mx.add_out_connector(out_c)
        state.remove_edge(e)
        state.add_edge(t, e.src_conn, mx, in_c, _copy.deepcopy(e.data))
        state.add_edge(mx, out_c, e.dst, e.dst_conn, _copy.deepcopy(e.data))

    if not in_edges:
        state.add_nedge(me, t, mm.Memlet())
    if not out_edges:
        state.add_nedge(t, mx, mm.Memlet())


def _build_duplicate(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit, orig_nsdfg: nodes.NestedSDFG,
                     orig_inner_state: SDFGState, keep: Set[nodes.Node], outer_in_by: Dict[str, list],
                     outer_out_by: Dict[str, list]):
    new_inner_sdfg = SDFG(orig_nsdfg.sdfg.name + "_pn")
    for name, desc in orig_nsdfg.sdfg.arrays.items():
        new_inner_sdfg.add_datadesc(name, _copy.deepcopy(desc))
    for name, stype in orig_nsdfg.sdfg.symbols.items():
        if name not in new_inner_sdfg.symbols:
            new_inner_sdfg.add_symbol(name, stype)

    new_inner_state = new_inner_sdfg.add_state("body", is_start_block=True)
    nmap = _copy_state_contents(orig_inner_state, new_inner_state)

    to_remove = [nmap[n] for n in orig_inner_state.nodes() if n not in keep]
    for n in to_remove:
        for e in list(new_inner_state.in_edges(n)) + list(new_inner_state.out_edges(n)):
            new_inner_state.remove_edge(e)
    for n in to_remove:
        if n in new_inner_state.nodes():
            new_inner_state.remove_node(n)

    used_in: Set[str] = set()
    used_out: Set[str] = set()
    for n in new_inner_state.nodes():
        if not isinstance(n, nodes.AccessNode):
            continue
        if n.data in orig_nsdfg.in_connectors and new_inner_state.out_degree(n) > 0:
            used_in.add(n.data)
        if n.data in orig_nsdfg.out_connectors and new_inner_state.in_degree(n) > 0:
            used_out.add(n.data)

    referenced: Set[str] = set()
    for n in new_inner_state.nodes():
        if isinstance(n, nodes.AccessNode):
            referenced.add(n.data)
    for e in new_inner_state.edges():
        if e.data is not None and e.data.data:
            referenced.add(e.data.data)
    for name in list(new_inner_sdfg.arrays):
        if name not in referenced:
            try:
                new_inner_sdfg.remove_data(name, validate=False)
            except Exception:
                new_inner_sdfg.arrays.pop(name, None)

    new_map = _copy.deepcopy(pe.map)
    new_pe = nodes.MapEntry(new_map)
    new_px = nodes.MapExit(new_map)
    graph.add_node(new_pe)
    graph.add_node(new_px)

    new_nsdfg = graph.add_nested_sdfg(new_inner_sdfg,
                                      used_in,
                                      used_out,
                                      symbol_mapping=_copy.copy(orig_nsdfg.symbol_mapping))

    for conn in used_in:
        new_pe.add_in_connector("IN_" + conn)
        new_pe.add_out_connector("OUT_" + conn)
        for (outer_access, outer_memlet, inner_memlet) in outer_in_by.get(conn, []):
            graph.add_edge(outer_access, None, new_pe, "IN_" + conn, _copy.deepcopy(outer_memlet))
            graph.add_edge(new_pe, "OUT_" + conn, new_nsdfg, conn, _copy.deepcopy(inner_memlet))
    for conn in used_out:
        new_px.add_in_connector("IN_" + conn)
        new_px.add_out_connector("OUT_" + conn)
        for (outer_access, outer_memlet, inner_memlet) in outer_out_by.get(conn, []):
            graph.add_edge(new_nsdfg, conn, new_px, "IN_" + conn, _copy.deepcopy(inner_memlet))
            graph.add_edge(new_px, "OUT_" + conn, outer_access, None, _copy.deepcopy(outer_memlet))

    if not used_in:
        graph.add_nedge(new_pe, new_nsdfg, mm.Memlet())
    if not used_out:
        graph.add_nedge(new_nsdfg, new_px, mm.Memlet())
