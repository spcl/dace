# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Make the contiguous axis of an imperfect device nest the thread dimension.

A kernel whose outer map walks a strided axis and whose inner map walks the unit-stride axis::

    GPU_Device map k:        # one thread per k
        per-k work           # pass-through reads, scalar chains
        map l (Sequential):  # l indexes the unit-stride dimension
            body(k, l)

runs one thread per row, each walking its row serially, so a warp touches as many rows as it has
lanes. This pass sinks the per-k work into the ``l`` map, recomputing it per lane, and collapses
the two maps into one ``GPU_Device`` map over ``[k, l]``. The last parameter becomes
``threadIdx.x``, so adjacent threads touch adjacent elements.

Legality of the sink. Per-k work is either an access node that only forwards data from the outer
entry to the inner entry (rewired, no data moves), or a side-effect-free tasklet / size-1
``Register`` transient chain that

* reads, through the outer entry, single elements of data nothing in the nest writes -- a
  per-lane recompute must see the value the single per-k evaluation saw, and must stay O(1)
  (a per-k reduction over a row is not re-run in every lane);
* writes only transients that no node outside the chain accesses (per-iteration private), and
  feeds its results only to the inner map.

Anything else between the two entries (a nested map such as a per-k reduction, a nested SDFG, a
larger buffer), any node between the two exits, a write-conflict-resolved output of the inner
map, or an inner range that depends on the outer parameters is refused, leaving the graph
untouched. The inner map's last parameter must be the one that indexes the unit-stride
dimension of the device-memory accesses in its body, and no outer parameter may index it.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, Memlet, SDFGState, data, dtypes, properties, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.helpers import redirect_edge

SERIAL_OR_DEVICE = (dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Default, dtypes.ScheduleType.GPU_Device)
SINKABLE_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)


def unit_stride_param(state: SDFGState, sdfg: SDFG, outer: nodes.MapEntry, inner: nodes.MapEntry) -> Optional[str]:
    outer_params = OrderedSet(outer.map.params)
    inner_params = OrderedSet(inner.map.params)
    found = OrderedSet()
    for edge in state.scope_subgraph(inner).edges():
        if edge.data.is_empty():
            continue
        desc = sdfg.arrays[edge.data.data]
        if not isinstance(desc, data.Array) or desc.storage == dtypes.StorageType.Register:
            continue
        for dim, stride in enumerate(desc.strides):
            if stride != 1:
                continue
            names = OrderedSet(str(s) for s in symbolic.symlist(edge.data.subset[dim]))
            if names & outer_params:
                return None
            found |= names & inner_params
    return found[0] if len(found) == 1 else None


def exits_adjacent(state: SDFGState, outer_exit: nodes.MapExit, inner_exit: nodes.MapExit,
                   prologue: List[nodes.Node]) -> bool:
    if any(e.dst is not outer_exit or e.data.wcr is not None for e in state.out_edges(inner_exit)):
        return False
    return all(e.src is inner_exit or (e.src in prologue and e.data.is_empty()) for e in state.in_edges(outer_exit))


def forwards(state: SDFGState, node: nodes.Node, outer: nodes.MapEntry, inner: nodes.MapEntry) -> bool:
    if not isinstance(node, nodes.AccessNode) or state.in_degree(node) != 1 or state.out_degree(node) == 0:
        return False
    edges = [*state.in_edges(node), *state.out_edges(node)]
    ends_ok = state.in_edges(node)[0].src is outer and all(e.dst is inner for e in state.out_edges(node))
    return ends_ok and all(
        not e.data.is_empty() and e.data.data == node.data and e.data.other_subset is None and e.data.wcr is None
        for e in edges)


def written_in_scope(state: SDFGState, outer: nodes.MapEntry) -> OrderedSet:
    written = OrderedSet()
    for edge in state.scope_subgraph(outer).edges():
        if edge.data.is_empty():
            continue
        if isinstance(edge.dst, nodes.AccessNode):
            written.add(edge.dst.data)
        elif isinstance(edge.dst, nodes.ExitNode):
            written.add(edge.data.data)
    return written


def private_to(sdfg: SDFG, name: str, members: List[nodes.Node]) -> bool:
    for state in sdfg.all_states():
        if any(n.data == name and n not in members for n in state.data_nodes()):
            return False
    return all(name not in e.data.read_symbols() for e in sdfg.all_interstate_edges(recursive=True))


def sinkable_node(sdfg: SDFG, node: nodes.Node, members: List[nodes.Node]) -> bool:
    if isinstance(node, nodes.Tasklet):
        return not node.has_side_effects(sdfg)
    if not isinstance(node, nodes.AccessNode):
        return False
    desc = node.desc(sdfg)
    return (desc.transient and desc.total_size == 1 and desc.storage in SINKABLE_STORAGE
            and private_to(sdfg, node.data, members))


def sink_edges_legal(state: SDFGState, node: nodes.Node, members: List[nodes.Node], ends: Tuple[nodes.Node, ...],
                     written: OrderedSet) -> bool:
    outer, inner, outer_exit = ends
    for edge in state.in_edges(node):
        if edge.data.wcr is not None or (edge.src is not outer and edge.src not in members):
            return False
        if edge.src is not outer or edge.data.is_empty():
            continue
        if edge.data.data in written or not isinstance(node, nodes.Tasklet) or edge.data.subset.num_elements() != 1:
            return False
    for edge in state.out_edges(node):
        if edge.data.wcr is not None:
            return False
        if edge.dst in members or (edge.dst is outer_exit and edge.data.is_empty()):
            continue
        if edge.dst is not inner:
            return False
        feeds = isinstance(node, nodes.AccessNode) and edge.data.data == node.data and edge.data.other_subset is None
        if not (edge.data.is_empty() or feeds):
            return False
    return True


def sink_plan(state: SDFGState, sdfg: SDFG, ends: Tuple[nodes.Node, ...],
              prologue: List[nodes.Node]) -> Optional[Tuple[List[nodes.Node], List[nodes.Node]]]:
    outer, inner, _ = ends
    forwarded = [n for n in prologue if forwards(state, n, outer, inner)]
    sunk = [n for n in prologue if n not in forwarded]
    if not sunk:
        return forwarded, sunk
    written = written_in_scope(state, outer)
    for node in sunk:
        if not sinkable_node(sdfg, node, sunk) or not sink_edges_legal(state, node, sunk, ends, written):
            return None
    return forwarded, sunk


def rewire_forwarded(state: SDFGState, node: nodes.AccessNode, outer: nodes.MapEntry, inner: nodes.MapEntry):
    source = state.in_edges(node)[0]
    for edge in state.out_edges(node):
        state.add_edge(outer, source.src_conn, inner, edge.dst_conn, copy.deepcopy(edge.data))
    state.remove_node(node)


def sink_node(state: SDFGState, node: nodes.Node, ends: Tuple[nodes.Node, ...], inner_exit: nodes.MapExit):
    outer, inner, outer_exit = ends
    for edge in list(state.in_edges(node)):
        if edge.src is not outer:
            continue
        if edge.data.is_empty():
            redirect_edge(state, edge, new_src=inner)
            continue
        conn = inner.next_connector()
        inner.add_in_connector('IN_' + conn)
        inner.add_out_connector('OUT_' + conn)
        state.add_edge(outer, edge.src_conn, inner, 'IN_' + conn, copy.deepcopy(edge.data))
        redirect_edge(state, edge, new_src=inner, new_src_conn='OUT_' + conn)
    for edge in list(state.out_edges(node)):
        if edge.dst is outer_exit:
            redirect_edge(state, edge, new_dst=inner_exit)
        elif edge.dst is inner:
            if not edge.data.is_empty():
                conn = edge.dst_conn[len('IN_'):]
                for use in list(state.out_edges_by_connector(inner, 'OUT_' + conn)):
                    state.add_edge(node, edge.src_conn, use.dst, use.dst_conn, copy.deepcopy(use.data))
                    state.remove_edge(use)
                inner.remove_in_connector(edge.dst_conn)
                inner.remove_out_connector('OUT_' + conn)
            state.remove_edge(edge)


@properties.make_properties
class ContiguousAxisToThreads(ppl.Pass):
    """Collapse ``GPU_Device map k { per-k work; map l }`` into ``GPU_Device map [k, l]`` when ``l``
    indexes the unit-stride dimension, sinking the per-k work into the lanes."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def promote(self, state: SDFGState, sdfg: SDFG, outer: nodes.MapEntry, collapse: MapCollapse) -> bool:
        children = state.scope_children()[outer]
        inners = [c for c in children if isinstance(c, nodes.MapEntry)]
        if len(inners) != 1 or inners[0].map.schedule not in SERIAL_OR_DEVICE:
            return False
        inner = inners[0]
        outer_exit, inner_exit = state.exit_node(outer), state.exit_node(inner)
        prologue = [c for c in children if c not in (inner, inner_exit, outer_exit)]
        if any(not c.startswith('IN_') for c in inner.in_connectors):
            return False
        if any(OrderedSet(map(str, symbolic.symlist(rng))) & OrderedSet(outer.map.params) for rng in inner.map.range):
            return False
        if unit_stride_param(state, sdfg, outer, inner) != inner.map.params[-1]:
            return False
        if not exits_adjacent(state, outer_exit, inner_exit, prologue):
            return False
        ends = (outer, inner, outer_exit)
        plan = sink_plan(state, sdfg, ends, prologue)
        if plan is None:
            return False
        forwarded, sunk = plan
        for node in forwarded:
            rewire_forwarded(state, node, outer, inner)
        for node in sunk:
            sink_node(state, node, ends, inner_exit)
        for node in sunk:
            if state.in_degree(node) == 0:
                state.add_edge(inner, None, node, None, Memlet())
            if state.out_degree(node) == 0:
                state.add_edge(node, None, inner_exit, None, Memlet())
        collapse.setup_match(sdfg,
                             state.parent_graph.cfg_id,
                             state.block_id, {
                                 MapCollapse.outer_map_entry: outer,
                                 MapCollapse.inner_map_entry: inner
                             },
                             0,
                             override=True)
        if not collapse.can_be_applied(state, 0, sdfg, permissive=True):
            raise RuntimeError(f'{outer.map.label}: MapCollapse refused a nest this pass made perfect')
        collapse.apply(state, sdfg)
        return True

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Collapse every device nest whose inner map walks the unit-stride axis.

        :param sdfg: the offloaded SDFG, in place.
        :param _pipeline_results: unused.
        :returns: how many kernels were collapsed, or ``None`` if none were.
        """
        collapse = MapCollapse()
        collapsed = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                scope = state.scope_dict()
                kernels = [
                    n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scope[n] is None
                    and n.map.schedule == dtypes.ScheduleType.GPU_Device
                ]
                collapsed += sum(self.promote(state, sd, k, collapse) for k in kernels)
        return collapsed or None
