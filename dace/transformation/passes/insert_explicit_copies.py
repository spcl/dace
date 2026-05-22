# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing implicit copy patterns (e.g. a path between two access nodes
without an intermediate tasklet) with explicit ``CopyLibraryNode`` instances.
"""
import copy
from typing import Any, Dict, Optional

from dace import data, dtypes, nodes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutils
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


def _derive_matching_dst_subset(src_subset: subsets.Range, dst_desc: data.Data) -> subsets.Range:
    """Pick the destination subset for a copy memlet that omits its dst side.

    Implicit copy edges may carry only the source subset. The implicit
    destination is the whole array whenever the two volumes are equal -- or
    cannot be proven unequal, which covers symbolic stepped ranges and
    consecutive-dimension reshapes (e.g. ``[8, 12] -> [96]``, or singleton dims
    present on one side only). A provably different volume falls back to
    ``src_subset``, correct when both sides share a shape.

    :param src_subset: the known side of the implicit copy.
    :param dst_desc: descriptor of the side whose subset is derived.
    :returns: a :class:`~dace.subsets.Range` for the destination side.
    """
    dst_range = subsets.Range.from_array(dst_desc)
    if symbolic.equal(src_subset.num_elements(), dst_range.num_elements()) is not False:
        return dst_range
    return src_subset


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitCopies(ppl.Pass):
    """Replaces implicit copy patterns with ``CopyLibraryNode`` instances.

    Detected patterns:
    - ``AccessNode -> AccessNode`` (direct copy edge) -- lifted to a libnode.
    - a ``View <-> AccessNode`` data-movement edge -- lifted to a libnode with
      the View as a normal endpoint (a View is an ``Array`` subclass with its
      own shape / strides). The View's alias (view-defining) edge to its
      underlying buffer is left untouched, so a round-trip ``AN -> View -> AN``
      becomes ``AN -> View -> Copy -> AN`` (or the mirror for a dst-side View).
    - ``AccessNode -> MapEntry -> AccessNode`` (stage-in) -- libnode placed
      inside the map scope, wired directly to the MapEntry's output
      connector. Chained MapEntries are followed via ``memlet_path``.
      Views on the outer side stay in place.
    - ``AccessNode -> MapExit -> AccessNode`` (stage-out) -- symmetric;
      libnode inside the map scope, output connector wired directly to
      MapExit.
    """

    # Storages whose copies CopyLibraryNode can lower. Other storages
    # (e.g. TensorCore_*, FPGA_*, Snitch_*) belong to custom codegen
    # targets that handle copies via their own ``copy_memory`` hook.
    _STANDARD_STORAGES = frozenset({
        dtypes.StorageType.Default,
        dtypes.StorageType.Register,
        dtypes.StorageType.CPU_Heap,
        dtypes.StorageType.CPU_Pinned,
        dtypes.StorageType.CPU_ThreadLocal,
        dtypes.StorageType.GPU_Global,
        dtypes.StorageType.GPU_Shared,
    })

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Lift every implicit copy in ``sdfg`` to a ``CopyLibraryNode``.

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

            # WCR edges aren't copies -- they're reductions. Lifting them
            # into a ``CopyLibraryNode`` would lose the conflict-resolution
            # semantics (write-without-merge). They're left in place so a
            # later pass can lower them to a proper reduction node.
            if memlet.wcr is not None:
                continue

            src_desc = sdfg.arrays[src_node.data]
            dst_desc = sdfg.arrays[dst_node.data]

            # A view's alias (view-defining) edge references the underlying
            # buffer rather than moving data -- skip it, and leave it intact so
            # ``get_view_edge`` keeps resolving the buffer. Any other
            # View<->Array edge is a real copy: a view is an ``Array`` subclass
            # with its own shape / strides, so it lifts like any endpoint.
            if any(
                    isinstance(sdfg.arrays[an.data], data.View) and sdutils.get_view_edge(state, an) is edge
                    for an in (src_node, dst_node)):
                continue

            # CopyLibraryNode expansion queries ``shape`` / ``strides`` /
            # ``is_packed_fortran_strides`` on both endpoints. ``Array`` and
            # ``Scalar`` both satisfy that contract (Scalar reports ``shape
            # = (1,)``, ``strides = [1]``). ``Stream`` (queue) and other
            # non-shape data classes do not -- leave the natural memlet for
            # the codegen's stream / custom paths.
            if not isinstance(src_desc, (data.Array, data.Scalar)) \
                    or not isinstance(dst_desc, (data.Array, data.Scalar)):
                continue

            # Custom-target storages (e.g. TensorCore_A/B/Accumulator from
            # the tensor_cores sample) are handled by their own
            # ``TargetCodeGenerator.copy_memory`` hook via wmma intrinsics.
            # Lifting them into a CopyLibraryNode emits scalar tasklet
            # assignments that don't compile against opaque fragment types.
            if (src_desc.storage not in self._STANDARD_STORAGES or dst_desc.storage not in self._STANDARD_STORAGES):
                continue

            src_name = src_node.data
            dst_name = dst_node.data

            # ``Memlet`` carries ``data`` (which array ``subset`` refers to) plus an
            # optional ``other_subset``. For self-copies (src_name == dst_name)
            # ``memlet.data`` matches both endpoints; the DaCe convention there
            # is that ``subset`` is the destination range, so check dst first.
            if memlet.data == dst_name:
                dst_subset = memlet.subset
                src_subset = memlet.other_subset
            elif memlet.data == src_name:
                src_subset = memlet.subset
                dst_subset = memlet.other_subset
            else:
                src_subset = memlet.subset
                dst_subset = memlet.other_subset

            # Fill in either side that wasn't carried by the memlet, deriving
            # a matching range on the absent side from the array shape when
            # the volumes line up (common for implicit copies between
            # different-shaped but same-volume arrays).
            if src_subset is None:
                src_subset = _derive_matching_dst_subset(dst_subset, src_desc)
            if dst_subset is None:
                dst_subset = _derive_matching_dst_subset(src_subset, dst_desc)

            in_memlet = Memlet(data=src_name, subset=copy.deepcopy(src_subset))
            in_memlet.dynamic = memlet.dynamic
            out_memlet = Memlet(data=dst_name, subset=copy.deepcopy(dst_subset))
            out_memlet.dynamic = memlet.dynamic

            label = f"copy_{src_name}_to_{dst_name}"
            libnode = CopyLibraryNode(name=label)

            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, in_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_node, None, out_memlet)
            count += 1

        return count

    def _replace_map_staging_copies(self, state: SDFGState) -> int:
        """Lift stage-in / stage-out copies through ``MapEntry`` / ``MapExit`` to ``CopyLibraryNode``.

        The libnode is placed inside the map scope: for stage-in it keeps the
        per-iteration memlet on the MapEntry side and a descriptor-derived
        memlet on the inner AccessNode; stage-out is symmetric. Chained
        MapEntries / MapExits are followed via ``memlet_path``.

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
        # For stage-in the inner side is edge.dst (AccessNode), for stage-out edge.src.
        inner_node = edge.dst if stage_in else edge.src
        if not isinstance(inner_node, nodes.AccessNode) or edge.data.is_empty():
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
        inner_subset = _derive_matching_dst_subset(outer_memlet.subset, inner_desc)
        inner_memlet = Memlet(data=inner_node.data, subset=inner_subset)
        label = (f"copy_{outer_memlet.data}_to_{inner_node.data}"
                 if stage_in else f"copy_{inner_node.data}_to_{outer_memlet.data}")
        libnode = CopyLibraryNode(name=label)
        state.add_node(libnode)
        if stage_in:
            map_node = edge.src
            state.add_edge(map_node, edge.src_conn, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                           copy.deepcopy(outer_memlet))
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, inner_node, None, inner_memlet)
        else:
            map_node = edge.dst
            state.add_edge(inner_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, inner_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, map_node, edge.dst_conn,
                           copy.deepcopy(outer_memlet))
        state.remove_edge(edge)
        return True
