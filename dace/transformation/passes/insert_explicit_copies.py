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
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


def _derive_matching_dst_subset(src_subset: subsets.Range, dst_desc: data.Data) -> subsets.Range:
    """Derive the absent side of a copy memlet: the full array when volumes are not
    provably unequal, else ``src_subset``.

    :param src_subset: the known (source) side of the copy.
    :param dst_desc: descriptor whose subset is being derived.
    :returns: the destination :class:`~dace.subsets.Range`.
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

            # A dtype-converting copy is a cast, not a byte move: CopyLibraryNode (memcpy)
            # cannot express it, so leave it for tasklet lowering (mirrors _lift_staging_edge).
            if src_desc.dtype != dst_desc.dtype:
                continue

            src_name = src_node.data
            dst_name = dst_node.data

            # Resolve src and dst subset. Self-copy: subset is the dst side;
            # otherwise the memlet path maps ``data`` to an endpoint.
            if src_name == dst_name:
                src_subset, dst_subset = memlet.other_subset, memlet.subset
            else:
                src_subset = memlet.get_src_subset(edge, state)
                dst_subset = memlet.get_dst_subset(edge, state)

            # Derive any side the memlet did not carry from the array shape (handles
            # implicit copies between different-shaped but same-volume arrays).
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
        outer_side_memlet = Memlet(data=outer.data, subset=copy.deepcopy(outer_subset))
        outer_side_memlet.dynamic = outer_memlet.dynamic
        outer_side_memlet.wcr = outer_memlet.wcr
        inner_subset = _derive_matching_dst_subset(outer_subset, inner_desc)
        inner_memlet = Memlet(data=inner_node.data, subset=inner_subset)
        label = (f"copy_{outer.data}_to_{inner_node.data}" if stage_in else f"copy_{inner_node.data}_to_{outer.data}")
        libnode = CopyLibraryNode(name=label)
        state.add_node(libnode)
        if stage_in:
            map_node = edge.src
            state.add_edge(map_node, edge.src_conn, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, outer_side_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, inner_node, None, inner_memlet)
        else:
            map_node = edge.dst
            state.add_edge(inner_node, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, inner_memlet)
            state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, map_node, edge.dst_conn, outer_side_memlet)
        state.remove_edge(edge)
        return True
