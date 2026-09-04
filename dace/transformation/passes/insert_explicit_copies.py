# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing implicit copy patterns with explicit ``CopyLibraryNode`` instances."""
import copy
from typing import Any, Dict, Optional

from dace import data, dtypes, nodes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutils
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


def _competing_writer(state: SDFGState, target: nodes.Node, edge, name: str, subset: subsets.Subset) -> bool:
    """True if another edge into ``target`` writes a region of ``name`` that may overlap ``subset``.

    Nothing in the graph orders two writes to the same region that reach a node on separate edges:
    plain copy-edge codegen emits a copy when its SOURCE access node is visited, so the copy lands
    before every other consumer of that node. Lifting the copy to a node of its own re-sorts it
    against the competing write, which silently swaps which value survives (measured on npbench
    ``vadv``: a dead ``dcol`` write moved after the tasklet that supersedes it). Where the order
    cannot be shown to be irrelevant, leave the copy implicit.

    :param state: the state holding ``target``.
    :param target: the node the copy writes through (access node, or map exit when staging out).
    :param edge: the copy edge itself, excluded from the scan.
    :param name: data name the copy writes.
    :param subset: region the copy writes.
    :returns: ``True`` when a possibly-overlapping competing write exists.
    """
    for other in state.in_edges(target):
        if other is edge or other.data.is_empty() or other.data.data != name:
            continue
        other_subset = other.data.get_dst_subset(other, state) or other.data.subset
        if subsets.intersects(other_subset, subset) is not False:
            return True
    return False


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
                count += self._replace_direct_copies(state)
                count += self._replace_map_staging_copies(state)
        return count if count > 0 else None

    def _replace_direct_copies(self, state: SDFGState) -> int:
        """Replace direct ``AccessNode -> AccessNode`` edges with ``CopyLibraryNode`` instances.

        :param state: The state to scan for direct copy edges (owning SDFG is ``state.sdfg``).
        :returns: The number of copy nodes inserted in ``state``.
        """
        sdfg = state.sdfg
        edges = list(state.edges())
        count = 0
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

            if _competing_writer(state, dst_node, edge, dst_name, dst_subset):
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

            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, in_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_node, None, out_memlet)
            _carry_write_ordering(state, dst_node, libnode)
            count += 1

        return count

    def _replace_map_staging_copies(self, state: SDFGState) -> int:
        """Lift stage-in / stage-out copies through ``MapEntry`` / ``MapExit`` to ``CopyLibraryNode``.

        The libnode sits inside the map scope; chained MapEntries / MapExits are followed via
        ``memlet_path``.

        :param state: The state to scan (owning SDFG is ``state.sdfg``).
        :returns: Number of libnodes inserted.
        """
        count = 0
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                for edge in list(state.out_edges(node)):
                    if self._lift_staging_edge(state, edge, stage_in=True):
                        count += 1
            elif isinstance(node, nodes.MapExit):
                for edge in list(state.in_edges(node)):
                    if self._lift_staging_edge(state, edge, stage_in=False):
                        count += 1
        return count

    def _lift_staging_edge(self, state: SDFGState, edge, stage_in: bool) -> bool:
        """Lift one stage-in (``stage_in=True``) or stage-out copy edge to a libnode.

        :returns: True iff the edge was lifted.
        """
        sdfg = state.sdfg
        # Inner side: edge.dst for stage-in, edge.src for stage-out.
        inner_node = edge.dst if stage_in else edge.src
        if not isinstance(inner_node, nodes.AccessNode) or edge.data.is_empty():
            return False
        # A WCR edge isn't a copy -- it's a reduction (e.g. AccumulateTransient's tile merge back
        # into the real output). CopyLibraryNode's expansions (ExpandMemcpyCPU et al.) always emit
        # an unconditional store; lifting a WCR edge here would silently turn the accumulate into
        # an overwrite. Mirrors the same guard in ``_replace_direct_copies``.
        if edge.data.wcr is not None:
            return False
        # A reference-set edge binds a POINTER rather than moving data; lifting it would drop the
        # ``set`` connector and leave the Reference unbound.
        if edge.dst_conn == 'set':
            return False
        inner_desc = sdfg.arrays[inner_node.data]
        if isinstance(inner_desc, data.View):
            return False
        find_outer = sdutils.find_input_arraynode if stage_in else sdutils.find_output_arraynode
        try:
            outer = find_outer(state, edge)
        except RuntimeError:
            return False
        outer_desc = sdfg.arrays[outer.data]
        if (outer_desc.storage not in self._STANDARD_STORAGES or inner_desc.storage not in self._STANDARD_STORAGES
                or outer_desc.dtype != inner_desc.dtype):
            return False

        outer_memlet = edge.data
        # The memlet may be dst-relative (subset in ``other_subset``); resolve it in the
        # outer array's index space via ``get_src/dst_subset``.
        if stage_in:
            outer_subset = outer_memlet.get_src_subset(edge, state) or outer_memlet.subset
        else:
            outer_subset = outer_memlet.get_dst_subset(edge, state) or outer_memlet.subset
            if _competing_writer(state, edge.dst, edge, outer.data, outer_subset):
                return False
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
        if stage_in and _competing_writer(state, inner_node, edge, inner_node.data, inner_subset):
            return False
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
        else:
            map_node = edge.dst
            state.add_edge(inner_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, inner_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, map_node, edge.dst_conn, outer_side_memlet)
            boundary_conn = 'OUT_' + edge.dst_conn[len('IN_'):]
            boundary_edges = list(state.out_edges_by_connector(map_node, boundary_conn))
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
        return True
