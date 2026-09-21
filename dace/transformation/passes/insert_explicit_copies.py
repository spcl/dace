# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing implicit copy patterns with explicit ``CopyLibraryNode`` instances."""
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dace import data, dtypes, nodes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutils
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.helper import CPU_RESIDENT_STORAGES, GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.copy import CopyLibraryNode


def _derive_matching_dst_subset(src_subset: subsets.Range, dst_desc: data.Data) -> subsets.Range:
    """Derive the absent side of a copy memlet.

    A copy edge that names only one side moves ``src_subset``'s volume; the legacy generator
    (``cpp.memlet_copy_to_absolute_strides``) drives the copy shape from that known side. So the
    derived side is a region of ``src_subset``'s shape at the array origin -- EXCEPT when the whole
    destination array provably holds exactly that volume, which is the reshape case (a ``[20]`` into a
    ``[4, 5]``): there the full array is the intended target. A merely *unprovable* equality
    (``i + 1`` vs ``20``, symbolic) is NOT a reshape -- taking the full array would copy the wrong
    element count (and read out of bounds on the smaller side), so it falls to ``src_subset``.

    :param src_subset: the known (source) side of the copy.
    :param dst_desc: descriptor whose subset is being derived.
    :returns: the destination :class:`~dace.subsets.Range`.
    """
    dst_range = subsets.Range.from_array(dst_desc)
    # Equalize first: two instances of the same symbol name make equal() answer None on identical counts.
    src_count, dst_count = symbolic.equalize_symbols(src_subset.num_elements(), dst_range.num_elements())
    if symbolic.equal(src_count, dst_count) is True:
        return dst_range
    return src_subset


#: ``(position, rank)``: where plain codegen emits a write within one state (see :class:`DispatchOrder`).
EmissionKey = Tuple[int, int]


class DispatchOrder:
    """The order plain codegen emits one state's writes in, taken before any copy is lifted.

    ``dispatch_subgraph`` walks a DFS topological order and emits a map scope at its entry. An
    implicit copy has no node of its own: a copy out of an access node is emitted during that node's
    visit, in out-edge order, and a stage-in copy during the visit of the access node it fills, ahead
    of that node's out-edges. So a key's rank is 0 for a node, positive for a copy out of an access
    node and negative for a stage-in copy.
    """

    __slots__ = ('position', 'rank')

    def __init__(self, state: SDFGState) -> None:
        sources = [node for node in state.nodes() if state.in_degree(node) == 0]
        walk = sdutils.dfs_topological_sort(state, sources)
        self.position: Dict[nodes.Node, int] = {node: index for index, node in enumerate(walk)}
        self.rank: Dict[MultiConnectorEdge, int] = {}
        for node in state.data_nodes():
            for index, edge in enumerate(state.out_edges(node)):
                self.rank[edge] = index + 1
            in_edges = state.in_edges(node)
            for index, edge in enumerate(in_edges):
                if isinstance(edge.src, nodes.MapEntry):
                    self.rank[edge] = index - len(in_edges)

    def node_key(self, node: nodes.Node) -> EmissionKey:
        """Key of a node's own emission."""
        return self.position[node], 0

    def copy_key(self, edge: MultiConnectorEdge) -> EmissionKey:
        """Key of an implicit copy edge, emitted while its access node is visited."""
        visited = edge.dst if isinstance(edge.src, nodes.MapEntry) else edge.src
        return self.position[visited], self.rank.get(edge, 0)


@dataclass(slots=True)
class LiftedCopy:
    """A copy the pass gave a node of its own, with the region it writes and its emission key.

    :ivar node: the inserted copy node.
    :ivar target: node the copy writes through (access node, or map exit when staging out).
    :ivar name: data name the copy writes.
    :ivar subset: region the copy writes.
    :ivar key: where plain codegen emitted the copy before it was lifted.
    """
    node: CopyLibraryNode
    target: nodes.Node
    name: str
    subset: subsets.Subset
    key: EmissionKey


def writer_anchor(state: SDFGState, edge: MultiConnectorEdge, order: DispatchOrder,
                  lifted: Dict[nodes.Node, EmissionKey]) -> Optional[Tuple[EmissionKey, nodes.Node, nodes.Node]]:
    """Emission key of a write reaching a node, with the nodes a happens-before edge attaches to.

    :param state: the state holding ``edge``.
    :param edge: the write.
    :param order: the state's emission order before lifting.
    :param lifted: emission key of every copy node lifted in this state.
    :returns: ``(key, tail, head)``: an edge leaving ``tail`` runs after the write, an edge entering
              ``head`` runs before it. ``None`` for a stage-in copy left implicit, which has no node.
    """
    src = edge.src
    if src in lifted:
        return lifted[src], src, src
    if isinstance(src, nodes.MapEntry):
        return None
    if isinstance(src, nodes.MapExit):
        entry = state.entry_node(src)
        return order.node_key(entry), src, entry
    if isinstance(src, nodes.AccessNode):
        return order.copy_key(edge), src, src
    return order.node_key(src), src, src


def add_ordering_edge(state: SDFGState, before: nodes.Node, after: nodes.Node) -> None:
    """Add the happens-before edge ``before -> after``, unless an edge already orders the pair or
    ``after`` already reaches ``before`` (the edge would close a cycle)."""
    if any(edge.dst is after for edge in state.out_edges(before)):
        return
    if any(node is before for node in state.bfs_nodes(after)):
        return
    state.add_nedge(before, after, Memlet())


def order_competing_writes(state: SDFGState, lifted: List[LiftedCopy], order: DispatchOrder) -> None:
    """Pin each lifted copy against every other write to its region with a happens-before edge.

    Nothing in the graph orders two writes to one region that reach a node on separate edges, so
    their order is the order plain codegen emitted them in. A lifted copy is sorted afresh as a node,
    which silently swaps which value survives (measured on npbench ``vadv``: a dead ``dcol`` write
    moved after the tasklet that supersedes it). The edge keeps the emission order.

    :param state: the state the copies were lifted in.
    :param lifted: the copies lifted in ``state``.
    :param order: the state's emission order before lifting.
    """
    keys = {lift.node: lift.key for lift in lifted}
    for lift in lifted:
        for other in state.in_edges(lift.target):
            if other.src is lift.node or other.data.is_empty() or other.data.data != lift.name:
                continue
            other_subset = other.data.get_dst_subset(other, state) or other.data.subset
            if subsets.intersects(other_subset, lift.subset) is False:
                continue
            anchor = writer_anchor(state, other, order, keys)
            if anchor is None:
                continue
            key, tail, head = anchor
            if key < lift.key:
                add_ordering_edge(state, tail, lift.node)
            elif key > lift.key and other.src not in keys:
                # A later lifted copy draws this edge itself, from its own side.
                add_ordering_edge(state, lift.node, head)


def _carry_write_ordering(state: SDFGState, written: nodes.AccessNode, libnode: nodes.Node) -> None:
    """Repeat onto ``libnode`` the ordering edges that sequenced writes to ``written``.

    An empty memlet is a happens-before edge, and the write it constrained is no longer the access
    node's own -- it is the libnode's. Left behind, it orders a node that no longer writes anything,
    and the libnode is free to be scheduled ahead of the write it was supposed to follow.

    :param state: the state both nodes live in.
    :param written: the access node the libnode now writes.
    :param libnode: the inserted copy node.
    """
    for edge in state.in_edges(written):
        if not edge.data.is_empty() or edge.src is libnode:
            continue
        if any(existing.src is edge.src for existing in state.in_edges(libnode)):
            continue
        state.add_edge(edge.src, None, libnode, None, Memlet())


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitCopies(ppl.Pass):
    """Replaces implicit copy patterns with ``CopyLibraryNode`` instances.

    Detected patterns:
    - ``AccessNode -> AccessNode`` (direct copy edge).
    - ``AccessNode <-> View <-> AccessNode`` data-movement edge -- View treated as a normal array endpoint.
    - ``AccessNode -> (MapEntry)+ -> AccessNode`` (stage-in) -- libnode placed inside the innermost map
      scope, wired to the MapEntry output connector.
    - ``AccessNode -> (MapExit)+ -> AccessNode`` (stage-out) -- symmetric, wired to the outermost MapExit.

    A lifted copy that shares a region with another write to the same node is ordered against it
    (:func:`order_competing_writes`).
    """

    # Storages whose copies CopyLibraryNode can lower. Other storages
    # (e.g. TensorCore_*, FPGA_*, Snitch_*) belong to custom codegen
    # targets that handle copies via their own ``copy_memory`` hook.
    _STANDARD_STORAGES = (CPU_RESIDENT_STORAGES | GPU_RESIDENT_STORAGES
                          | {dtypes.StorageType.Default, dtypes.StorageType.Register})

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Lift every implicit copy in ``sdfg`` (and nested SDFGs) to a ``CopyLibraryNode``.

        :param sdfg: The SDFG to transform, recursively including nested SDFGs.
        :param pipeline_results: Results of previously applied passes (unused).
        :returns: The number of copy nodes inserted, or ``None`` if none.
        """
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                order = DispatchOrder(state)
                lifted = self._replace_direct_copies(state, order)
                lifted.extend(self._replace_map_staging_copies(state, order))
                order_competing_writes(state, lifted, order)
                count += len(lifted)
        return count if count > 0 else None

    def _replace_direct_copies(self, state: SDFGState, order: DispatchOrder) -> List[LiftedCopy]:
        """Replace direct ``AccessNode -> AccessNode`` edges with ``CopyLibraryNode`` instances.

        :param state: The state to scan for direct copy edges (owning SDFG is ``state.sdfg``).
        :param order: the state's emission order before lifting.
        :returns: The copies lifted in ``state``.
        """
        sdfg = state.sdfg
        edges = list(state.edges())
        lifted: List[LiftedCopy] = []
        for edge in edges:
            if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                continue

            src_node: nodes.AccessNode = edge.src
            dst_node: nodes.AccessNode = edge.dst
            memlet: Memlet = edge.data

            if memlet.is_empty():
                continue

            # WCR edges aren't copies.
            if memlet.wcr is not None:
                continue

            # A set binds a pointer, not data: lifting it drops the ``set`` connector.
            if edge.dst_conn == 'set':
                continue

            src_desc = sdfg.arrays[src_node.data]
            dst_desc = sdfg.arrays[dst_node.data]

            # A view's alias (view-defining) edge references the underlying
            # buffer rather than moving data -- skip it.
            if any(
                    isinstance(sdfg.arrays[an.data], data.View) and sdutils.get_view_edge(state, an) is edge
                    for an in (src_node, dst_node)):
                continue

            # We only copy array-like data (Array / Scalar), not streams.
            if not isinstance(src_desc, (data.Array, data.Scalar)) \
                    or not isinstance(dst_desc, (data.Array, data.Scalar)):
                continue

            # Custom-target storages are handled by their own codegen, not CopyLibraryNode.
            if (src_desc.storage not in self._STANDARD_STORAGES or dst_desc.storage not in self._STANDARD_STORAGES):
                continue

            src_name = src_node.data
            dst_name = dst_node.data

            # Resolve src and dst subset. ``get_src_subset`` / ``get_dst_subset`` are the only correct
            # readers of the ``subset`` / ``other_subset`` pair: which side ``subset`` names is carried
            # by the memlet's own ``_is_data_src`` flag, NOT derivable from the endpoint names. A
            # self-copy is no exception -- both endpoints match ``memlet.data``, so the flag is what
            # decides, and ``try_initialize`` defaults it to src-relative. This is what the legacy
            # generator lowers a copy edge with (``cpp.memlet_copy_to_absolute_strides``); reading
            # the pair positionally instead reverses every src-relative self-copy.
            src_subset = memlet.get_src_subset(edge, state)
            dst_subset = memlet.get_dst_subset(edge, state)

            # Derive any side the memlet did not carry from the array shape (handles
            # implicit copies between different-shaped but same-volume arrays).
            if src_subset is None:
                src_subset = _derive_matching_dst_subset(dst_subset, src_desc)
            if dst_subset is None:
                dst_subset = _derive_matching_dst_subset(src_subset, dst_desc)

            # A copy of zero elements moves nothing, and plain copy-edge codegen emits nothing for
            # it. Lifting it would put a node in the state that has no work to do.
            if symbolic.equal(src_subset.num_elements(), 0, is_length=False) is True:
                continue

            in_memlet = Memlet(data=src_name, subset=copy.deepcopy(src_subset))
            in_memlet.dynamic = memlet.dynamic
            out_memlet = Memlet(data=dst_name, subset=copy.deepcopy(dst_subset))
            out_memlet.dynamic = memlet.dynamic
            # ``allow_oob`` is the author's waiver of the src/dst volume check (``validation.py``
            # honours it the same way); dropping it here turns a legal copy into an expansion error.
            in_memlet.allow_oob = memlet.allow_oob
            out_memlet.allow_oob = memlet.allow_oob

            label = f"copy_{src_name}_to_{dst_name}"
            libnode = CopyLibraryNode(name=label)
            # Instrumentation providers decide a copy edge's instrumentation from the state it is in
            # (``on_copy_begin``); as a node the copy needs its own setting to stay measured.
            libnode.instrument = state.instrument

            key = order.copy_key(edge)
            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, in_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_node, None, out_memlet)
            _carry_write_ordering(state, dst_node, libnode)
            lifted.append(LiftedCopy(libnode, dst_node, dst_name, dst_subset, key))

        return lifted

    def _replace_map_staging_copies(self, state: SDFGState, order: DispatchOrder) -> List[LiftedCopy]:
        """Lift stage-in / stage-out copies through ``MapEntry`` / ``MapExit`` to ``CopyLibraryNode``.

        The libnode sits inside the map scope; chained MapEntries / MapExits are followed via
        ``memlet_path``.

        :param state: The state to scan (owning SDFG is ``state.sdfg``).
        :param order: the state's emission order before lifting.
        :returns: The copies lifted in ``state``.
        """
        lifted: List[LiftedCopy] = []
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                candidates, stage_in = list(state.out_edges(node)), True
            elif isinstance(node, nodes.MapExit):
                candidates, stage_in = list(state.in_edges(node)), False
            else:
                continue
            for edge in candidates:
                lift = self._lift_staging_edge(state, edge, stage_in, order)
                if lift is not None:
                    lifted.append(lift)
        return lifted

    def _lift_staging_edge(self, state: SDFGState, edge, stage_in: bool, order: DispatchOrder) -> Optional[LiftedCopy]:
        """Lift one stage-in (``stage_in=True``) or stage-out copy edge to a libnode.

        :returns: the lifted copy, or ``None`` if the edge was left as it is.
        """
        sdfg = state.sdfg
        # Inner side: edge.dst for stage-in, edge.src for stage-out.
        inner_node = edge.dst if stage_in else edge.src
        if not isinstance(inner_node, nodes.AccessNode) or edge.data.is_empty():
            return None
        # A reference-set edge binds a POINTER rather than moving data; lifting it would drop the
        # ``set`` connector and leave the Reference unbound.
        if edge.dst_conn == 'set':
            return None
        inner_desc = sdfg.arrays[inner_node.data]
        if isinstance(inner_desc, data.View):
            return None
        find_outer = sdutils.find_input_arraynode if stage_in else sdutils.find_output_arraynode
        try:
            outer = find_outer(state, edge)
        except RuntimeError:
            return None
        outer_desc = sdfg.arrays[outer.data]
        # A dtype change is fine: the copy node's selector lowers a converting copy to a casting tasklet.
        if outer_desc.storage not in self._STANDARD_STORAGES or inner_desc.storage not in self._STANDARD_STORAGES:
            return None
        # A WCR edge isn't a copy -- it's a reduction (e.g. AccumulateTransient's tile merge back
        # into the real output), and the memcpy expansions store unconditionally, so lifting one
        # turns the accumulate into an overwrite. A host single-element stage-out is the exception:
        # ``Auto`` has nothing but ``Tasklet`` to pick for it, and a tasklet keeps the WCR on its
        # output edge for the generator's own conflict resolution to lower. Left implicit that shape
        # has no explicit spelling at all -- ``cpu.py`` falls back to ``dace::CopyND::Accumulate``.
        if edge.data.wcr is not None:
            liftable = (not stage_in and not GPU_RESIDENT_STORAGES & {outer_desc.storage, inner_desc.storage}
                        and all(sbs is None or sbs.num_elements_exact() == 1
                                for sbs in (edge.data.subset, edge.data.other_subset)))
            if not liftable:
                return None

        outer_memlet = edge.data
        # The memlet may be dst-relative (subset in ``other_subset``); resolve it in the
        # outer array's index space via ``get_src/dst_subset``.
        if stage_in:
            outer_subset = outer_memlet.get_src_subset(edge, state) or outer_memlet.subset
        else:
            outer_subset = outer_memlet.get_dst_subset(edge, state) or outer_memlet.subset
        outer_side_memlet = Memlet(data=outer.data, subset=copy.deepcopy(outer_subset))
        outer_side_memlet.dynamic = outer_memlet.dynamic
        outer_side_memlet.wcr = outer_memlet.wcr
        # When the memlet names both sides that mapping IS the copy; deriving one retargets the write.
        if stage_in:
            inner_subset = outer_memlet.get_dst_subset(edge, state)
        else:
            inner_subset = outer_memlet.get_src_subset(edge, state)
        if inner_subset is None or outer_memlet.other_subset is None:
            inner_subset = _derive_matching_dst_subset(outer_subset, inner_desc)
        else:
            inner_subset = copy.deepcopy(inner_subset)
        # One container over one subset on both sides of the scope is the same memory: nothing moves.
        if outer.data == inner_node.data and inner_subset == outer_subset:
            return None
        key = order.copy_key(edge)
        inner_memlet = Memlet(data=inner_node.data, subset=inner_subset)
        label = (f"copy_{outer.data}_to_{inner_node.data}" if stage_in else f"copy_{inner_node.data}_to_{outer.data}")
        libnode = CopyLibraryNode(name=label)
        libnode.instrument = state.instrument
        state.add_node(libnode)
        if stage_in:
            map_node = edge.src
            state.add_edge(map_node, edge.src_conn, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, outer_side_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, inner_node, None, inner_memlet)
            _carry_write_ordering(state, inner_node, libnode)
            boundary_conn = 'IN_' + edge.src_conn[len('OUT_'):]
            boundary_edges = list(state.in_edges_by_connector(map_node, boundary_conn))
            lift = LiftedCopy(libnode, inner_node, inner_node.data, inner_subset, key)
        else:
            map_node = edge.dst
            state.add_edge(inner_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, inner_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, map_node, edge.dst_conn, outer_side_memlet)
            boundary_conn = 'OUT_' + edge.dst_conn[len('IN_'):]
            boundary_edges = list(state.out_edges_by_connector(map_node, boundary_conn))
            lift = LiftedCopy(libnode, map_node, outer.data, outer_subset, key)
        state.remove_edge(edge)

        # The scope-boundary edge on this connector may still carry a memlet whose ``.data``
        # names the inner array, relying on memlet_path continuing through the scope entry/exit
        # straight to inner_node for validation (validation.py resolves src/dst from the full
        # path, not the edge's immediate neighbours). That continuation broke: the libnode now
        # sits between the scope node and inner_node, so the path ends at a non-AccessNode and
        # the boundary edge needs its own outer-relative memlet instead.
        for bedge in boundary_edges:
            if bedge.data.data != outer.data:
                bedge.data = Memlet(data=outer.data, subset=copy.deepcopy(outer_subset))
        return lift
