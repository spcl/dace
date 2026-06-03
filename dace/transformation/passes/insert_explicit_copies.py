# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing implicit copy patterns (a path between two access nodes
without an intermediate tasklet) with explicit ``CopyLibraryNode`` instances.
"""
import copy as _copy
from typing import Any, Dict, Optional

import dace
from dace import dtypes, nodes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutils
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


def _resolve_subset_for(memlet, an_node):
    """Return the subset of ``an_node`` referenced by ``memlet``.

    :param memlet: The memlet on the edge incident to ``an_node``.
    :param an_node: The access node whose range is being resolved.
    :returns: ``memlet.subset`` when ``memlet`` targets ``an_node`` directly;
        ``memlet.other_subset`` when the memlet targets something else on the
        same edge (e.g. a View); otherwise a 0-anchored range matching the
        volume shape (the DaCe memlet-propagation convention).
    """
    import dace.subsets as _ss
    if memlet.data == an_node.data:
        return memlet.subset
    if memlet.other_subset is not None:
        return memlet.other_subset
    sizes = memlet.subset.size()
    return _ss.Range([(0, s - 1, 1) for s in sizes])


def _derive_matching_dst_subset(src_subset, dst_desc):
    """Pick a destination subset for a memlet that omits ``other_subset``/``dst_subset``.

    Convention used by implicit copy edges: if the destination array's shape
    matches the source subset's volume shape (either directly, or after
    squeezing singleton dimensions), the implicit destination range is the
    full destination (offset 0 on every dimension). Otherwise it falls back
    to ``src_subset`` -- correct only when the two arrays share the same shape.

    :param src_subset: The known side of the implicit copy.
    :param dst_desc: Data descriptor of the side whose subset is being derived.
    :returns: A :class:`~dace.subsets.Range` for the destination side.
    """
    from dace import subsets as _subsets

    src_size = list(src_subset.size())
    dst_shape = list(dst_desc.shape)

    # DaCe symbols are interned by name but ``sympy.simplify`` can leave
    # ``N - N`` un-simplified when the two ``N`` instances belong to different
    # symbol objects. Compare via string repr, which yields a canonical form
    # for symbols sharing a name.
    def _eq(a, b):
        if a is b or a == b:
            return True
        try:
            return str(a) == str(b)
        except Exception:
            return False

    def _shapes_match(a, b):
        return len(a) == len(b) and all(_eq(s, d) for s, d in zip(a, b))

    # Direct match: ranks and per-dim sizes line up exactly.
    if _shapes_match(src_size, dst_shape):
        return _subsets.Range.from_array(dst_desc)

    # Rank-reducing case: drop singleton dims from the source subset and
    # try again. Pattern: ``A[i, j, 0:N]`` -> 1-D destination of size ``N``
    # where the singleton index dims are implicit on the source side only.
    src_size_squeezed = [s for s in src_size if not _eq(s, 1)]
    if _shapes_match(src_size_squeezed, dst_shape):
        return _subsets.Range.from_array(dst_desc)

    # Rank match (after squeezing) but per-dim sizes differ symbolically.
    # This catches cases where the subset's stepped range produces a size
    # symbolically different from the destination's declared shape (e.g.
    # ``1:N-1:2`` -> ``ceiling(N/2) - 1`` vs ``floor(N/2) - 1``) even
    # though they agree at runtime for valid ``N``. Trust the user's
    # intent that the volumes line up and pick the destination's full
    # natural range.
    if len(src_size_squeezed) == len(dst_shape):
        return _subsets.Range.from_array(dst_desc)

    # Consecutive-rank reshape: a contiguous run of source dims may collapse
    # into a single destination dim (or vice versa, splitting a source dim
    # into a run of destination dims). Walk both shapes left-to-right with
    # two pointers, accumulating the running product on whichever side has
    # the smaller current value, and advance both when products meet.
    # Example: ``[8, 12, 5, 3] -> [96, 5, 3]`` collapses dims 0-1; ``[80, 12]
    # -> [8, 10, 12]`` splits dim 0. Squeeze 1s on both sides first so a
    # ``[8, 1, 12]`` source matches a ``[8, 12]`` destination through this
    # path even though squeezing the source alone would have caught it
    # earlier.
    dst_shape_squeezed = [s for s in dst_shape if not _eq(s, 1)]
    if _is_consecutive_reshape(src_size_squeezed, dst_shape_squeezed):
        return _subsets.Range.from_array(dst_desc)

    return src_subset


def _is_consecutive_reshape(src_size, dst_shape):
    """Test whether ``dst_shape`` is reachable from ``src_size`` by reshaping.

    Reachable means by collapsing contiguous runs of dimensions, or splitting
    one dimension into a contiguous run. Both inputs must be 1-squeezed.

    :param src_size: Per-dimension sizes of the source subset.
    :param dst_shape: Per-dimension sizes of the destination array.
    :returns: ``True`` iff the running products coincide.
    """
    i = j = 0
    src_acc = 1
    dst_acc = 1
    while i < len(src_size) and j < len(dst_shape):
        if src_acc == dst_acc:
            src_acc *= src_size[i]
            dst_acc *= dst_shape[j]
            i += 1
            j += 1
            continue
        # Symbolic ``<`` may be indeterminate -- fall back to advancing dst (safe under the equal-product check below).
        try:
            advance_src = bool((src_acc - dst_acc) < 0)
        except Exception:
            advance_src = False
        if advance_src:
            src_acc *= src_size[i]
            i += 1
        else:
            dst_acc *= dst_shape[j]
            j += 1
    while i < len(src_size):
        src_acc *= src_size[i]
        i += 1
    while j < len(dst_shape):
        dst_acc *= dst_shape[j]
        j += 1
    try:
        return bool((src_acc - dst_acc) == 0)
    except Exception:
        return str(src_acc) == str(dst_acc)


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitCopies(ppl.Pass):
    """Replaces implicit copy patterns with ``CopyLibraryNode`` instances.

    Detected patterns:
    - ``AccessNode -> AccessNode`` (direct copy edge) -- lifted to a libnode.
    - ``AccessNode -> View -> AccessNode`` (round-trip through a View) --
      first collapsed into a direct ``AN -> AN`` edge composing the two
      memlets, then lifted by the direct-copy path.
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
                self._collapse_round_trip_views(nsdfg, state)
                count += self._replace_direct_copies(nsdfg, state)
                count += self._replace_map_staging_copies(nsdfg, state)
        return count if count > 0 else None

    def _collapse_round_trip_views(self, sdfg: SDFG, state: SDFGState):
        """Collapse ``AN_src -> View -> AN_dst`` aliasing paths into a direct edge.

        Such a path is an implicit copy through the View's alias. The two edges
        are replaced with a single ``AN_src -> AN_dst`` edge carrying the
        composed memlet (underlying-side subset, access-side subset); the View
        access node stays in the state but is detached from this dataflow path.
        :meth:`_replace_direct_copies` then lifts the new direct edge.

        :param sdfg: The (possibly nested) SDFG owning ``state``.
        :param state: The state to scan for round-trip View paths.
        """
        from dace import data as dt
        from dace.sdfg.utils import get_view_edge

        for view_node in list(state.nodes()):
            if not isinstance(view_node, nodes.AccessNode):
                continue
            if not isinstance(sdfg.arrays[view_node.data], dt.View):
                continue
            in_edges = [e for e in state.in_edges(view_node) if not e.data.is_empty()]
            out_edges = [e for e in state.out_edges(view_node) if not e.data.is_empty()]
            if len(in_edges) != 1 or len(out_edges) != 1:
                continue
            in_e, out_e = in_edges[0], out_edges[0]
            if not isinstance(in_e.src, nodes.AccessNode) or not isinstance(out_e.dst, nodes.AccessNode):
                continue
            src_node, dst_node = in_e.src, out_e.dst
            if isinstance(sdfg.arrays[src_node.data], dt.View) or isinstance(sdfg.arrays[dst_node.data], dt.View):
                continue
            view_edge = get_view_edge(state, view_node)
            if view_edge is None:
                continue

            src_subset = _resolve_subset_for(in_e.data, src_node)
            dst_subset = _resolve_subset_for(out_e.data, dst_node)

            new_memlet = Memlet(data=src_node.data,
                                subset=_copy.deepcopy(src_subset),
                                other_subset=_copy.deepcopy(dst_subset))
            new_memlet.dynamic = in_e.data.dynamic or out_e.data.dynamic
            state.remove_edge(in_e)
            state.remove_edge(out_e)
            state.add_edge(src_node, in_e.src_conn, dst_node, out_e.dst_conn, new_memlet)
            # The View is now disconnected from this dataflow path; drop it if
            # nothing else references it (SDFG validation rejects isolated nodes).
            if state.degree(view_node) == 0:
                state.remove_node(view_node)

    def _replace_direct_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """Replace direct ``AccessNode -> AccessNode`` edges with ``CopyLibraryNode`` instances.

        :param sdfg: The (possibly nested) SDFG owning ``state``.
        :param state: The state to scan for direct copy edges.
        :returns: The number of copy nodes inserted in ``state``.
        """
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

            # Views alias their underlying array; an Array<->View edge is an
            # aliasing reference, not a copy. Lifting it into a CopyLibraryNode
            # would (a) emit a memcpy between two pointers into the same buffer
            # and (b) break ``sdutil.get_view_edge``, which requires the View's
            # neighbor on at least one side to be an AccessNode -- it walks the
            # adjacent edge to the underlying buffer.
            if isinstance(src_desc, dace.data.View) or isinstance(dst_desc, dace.data.View):
                continue

            # CopyLibraryNode expansion queries ``shape`` / ``strides`` /
            # ``is_packed_fortran_strides`` on both endpoints. ``Array`` and
            # ``Scalar`` both satisfy that contract (Scalar reports ``shape
            # = (1,)``, ``strides = [1]``). ``Stream`` (queue) and other
            # non-shape data classes do not -- leave the natural memlet for
            # the codegen's stream / custom paths.
            if not isinstance(src_desc, (dace.data.Array, dace.data.Scalar)) \
                    or not isinstance(dst_desc, (dace.data.Array, dace.data.Scalar)):
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

            in_memlet = Memlet(data=src_name, subset=_copy.deepcopy(src_subset))
            in_memlet.dynamic = memlet.dynamic
            out_memlet = Memlet(data=dst_name, subset=_copy.deepcopy(dst_subset))
            out_memlet.dynamic = memlet.dynamic

            label = f"copy_{src_name}_to_{dst_name}"
            libnode = CopyLibraryNode(name=label)

            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, in_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_node, None, out_memlet)
            count += 1

        return count

    def _replace_map_staging_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """Lift stage-in / stage-out copies through ``MapEntry`` / ``MapExit`` to ``CopyLibraryNode``.

        Stage-in is a ``MapEntry -> AccessNode`` edge whose ``memlet_path`` traces
        back to an outer AccessNode; the libnode is inserted between MapEntry
        and the inner AN, wired to MapEntry's existing output connector with
        the original (per-iteration) memlet preserved, and to the inner AN
        with a memlet derived for the inner array's descriptor. Stage-out is
        symmetric. Chained MapEntries / MapExits are followed by
        ``memlet_path``; downstream content (tasklets, NestedSDFGs, nested
        maps) is irrelevant to the lift.

        :param sdfg: The (possibly nested) SDFG owning ``state``.
        :param state: The state to scan.
        :returns: Number of libnodes inserted.
        """
        count = 0
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                for edge in list(state.out_edges(node)):
                    if self._lift_staging_edge(sdfg, state, edge, stage_in=True):
                        count += 1
            elif isinstance(node, nodes.MapExit):
                for edge in list(state.in_edges(node)):
                    if self._lift_staging_edge(sdfg, state, edge, stage_in=False):
                        count += 1
        return count

    def _lift_staging_edge(self, sdfg: SDFG, state: SDFGState, edge, stage_in: bool) -> bool:
        """Lift one stage-in (``stage_in=True``) or stage-out copy edge to a libnode.

        :returns: True iff the edge was lifted.
        """
        # For stage-in the inner side is edge.dst (AccessNode), for stage-out edge.src.
        inner_node = edge.dst if stage_in else edge.src
        if not isinstance(inner_node, nodes.AccessNode) or edge.data.is_empty():
            return False
        inner_desc = sdfg.arrays[inner_node.data]
        if isinstance(inner_desc, dace.data.View):
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
                           _copy.deepcopy(outer_memlet))
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, inner_node, None, inner_memlet)
        else:
            map_node = edge.dst
            state.add_edge(inner_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, inner_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, map_node, edge.dst_conn,
                           _copy.deepcopy(outer_memlet))
        state.remove_edge(edge)
        return True
