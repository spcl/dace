# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass that replaces implicit copy patterns -- direct ``AccessNode ->
AccessNode`` edges and map-boundary staging (AN -> MapEntry -> AN
transient; AN transient -> MapExit -> AN) -- with explicit
``CopyLibraryNode`` instances.  ``src_locations`` / ``dst_locations``
restrict lifting to specific storage-pair filters (empty means any);
the GPU-specific wrapper ``InsertExplicitGPUGlobalMemoryCopies`` uses
this to target GPU_Global / CPU_Pinned copies only.
"""
import copy as _copy
from typing import Any, Dict, Iterable, Optional

import dace
from dace import dtypes, nodes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitCopies(ppl.Pass):
    """
    Replaces implicit copy patterns with explicit ``CopyLibraryNode``
    instances.

    Detected patterns:

    - ``AccessNode -> AccessNode`` (direct copy edge)
    - ``AccessNode -> MapEntry -> AccessNode(transient)`` (stage-in)
    - ``AccessNode(transient) -> MapExit -> AccessNode`` (stage-out)

    The pass sets ``src_storage`` and ``dst_storage`` on each new node
    from the array descriptors.

    The ``src_locations`` / ``dst_locations`` properties restrict which
    copies are lifted.  Empty / ``None`` means "any storage".  When set,
    only copies whose source storage is in ``src_locations`` *and* whose
    destination storage is in ``dst_locations`` are replaced.
    """

    CATEGORY = "Optimization Preparation"

    src_locations = properties.SetProperty(
        element_type=dtypes.StorageType,
        default=set(),
        desc="Only lift copies whose source storage is in this set. "
        "Empty set means any source storage is accepted.",
    )
    dst_locations = properties.SetProperty(
        element_type=dtypes.StorageType,
        default=set(),
        desc="Only lift copies whose destination storage is in this set. "
        "Empty set means any destination storage is accepted.",
    )

    def __init__(self,
                 src_locations: Optional[Iterable[dtypes.StorageType]] = None,
                 dst_locations: Optional[Iterable[dtypes.StorageType]] = None):
        super().__init__()
        self.src_locations = set(src_locations) if src_locations else set()
        self.dst_locations = set(dst_locations) if dst_locations else set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _storage_allowed(self, src_storage: dtypes.StorageType, dst_storage: dtypes.StorageType) -> bool:
        """Return True when the (src, dst) storage pair passes the configured filter."""
        if self.src_locations and src_storage not in self.src_locations:
            return False
        if self.dst_locations and dst_storage not in self.dst_locations:
            return False
        return True

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """
        :return: Number of copy nodes inserted, or None if nothing changed.
        """
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                count += self._replace_direct_copies(nsdfg, state)
                count += self._replace_map_staging_copies(nsdfg, state)
        return count if count > 0 else None

    def _replace_direct_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """Replace direct ``AccessNode -> AccessNode`` edges with ``CopyLibraryNode`` instances."""
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

            src_desc = sdfg.arrays[src_node.data]
            dst_desc = sdfg.arrays[dst_node.data]

            if not self._storage_allowed(src_desc.storage, dst_desc.storage):
                continue

            src_name = src_node.data
            dst_name = dst_node.data
            src_subset = memlet.src_subset or memlet.subset
            dst_subset = memlet.dst_subset or memlet.other_subset or src_subset

            in_memlet = Memlet(data=src_name, subset=_copy.deepcopy(src_subset))
            in_memlet.dynamic = memlet.dynamic
            out_memlet = Memlet(data=dst_name, subset=_copy.deepcopy(dst_subset))
            out_memlet.dynamic = memlet.dynamic

            label = f"copy_{src_name}_to_{dst_name}"
            libnode = CopyLibraryNode(
                name=label,
                src_storage=src_desc.storage,
                dst_storage=dst_desc.storage,
            )

            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, "_in", in_memlet)
            state.add_edge(libnode, "_out", dst_node, None, out_memlet)
            count += 1

        return count

    def _replace_map_staging_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """
        Replace map-boundary staging paths with ``CopyLibraryNode`` instances.

        Two staging directions are handled:

          * ``AccessNode -> MapEntry -> AccessNode(transient)`` (stage-in)
          * ``AccessNode(transient) -> MapExit -> AccessNode`` (stage-out)

        The map scope must be entered from (or exited to) an outer access
        node through a pass-through memlet; the inserted copy node
        materializes the transfer at the boundary.
        """
        edges_to_process = []

        for e in state.edges():
            if e.data.is_empty():
                continue
            # Stage-in: MapEntry -> AccessNode(transient)
            # Require an outer AccessNode feeding the MapEntry with the
            # same data name (i.e. a proper pass-through path).
            if (isinstance(e.src, nodes.MapEntry) and isinstance(e.dst, nodes.AccessNode)):
                desc = sdfg.arrays.get(e.dst.data)
                if desc is None or not desc.transient:
                    continue
                if e.src_conn is None or not e.src_conn.startswith("OUT_"):
                    continue
                in_conn = "IN_" + e.src_conn[len("OUT_"):]
                outer_an = None
                for oe in state.in_edges(e.src):
                    if (oe.dst_conn == in_conn and isinstance(oe.src, nodes.AccessNode)):
                        outer_an = oe.src
                        break
                if outer_an is None:
                    continue
                edges_to_process.append(('in', e))
            # Stage-out: AccessNode(transient) -> MapExit
            # Require the MapExit to feed an outer AccessNode on the
            # matching OUT_ connector.
            elif (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.MapExit)):
                desc = sdfg.arrays.get(e.src.data)
                if desc is None or not desc.transient:
                    continue
                if e.dst_conn is None or not e.dst_conn.startswith("IN_"):
                    continue
                out_conn = "OUT_" + e.dst_conn[len("IN_"):]
                outer_an = None
                for oe in state.out_edges(e.dst):
                    if (oe.src_conn == out_conn and isinstance(oe.dst, nodes.AccessNode)):
                        outer_an = oe.dst
                        break
                if outer_an is None:
                    continue
                edges_to_process.append(('out', e))

        count = 0
        for direction, edge in edges_to_process:
            if edge not in state.edges():
                continue  # already removed by earlier iteration

            if direction == 'in':
                # ME(OUT_X) --[outer_data[per_iter_subset]]--> AN(local)
                me = edge.src
                local_an = edge.dst
                outer_memlet = edge.data
                local_desc = sdfg.arrays[local_an.data]
                outer_desc = sdfg.arrays[outer_memlet.data]

                if not self._storage_allowed(outer_desc.storage, local_desc.storage):
                    continue

                local_subset = dace.subsets.Range.from_array(local_desc)
                local_memlet = Memlet(data=local_an.data, subset=local_subset)

                libnode = CopyLibraryNode(name=f"copy_{outer_memlet.data}_to_{local_an.data}",
                                          src_storage=outer_desc.storage,
                                          dst_storage=local_desc.storage)

                state.remove_edge(edge)
                state.add_node(libnode)
                state.add_edge(me, edge.src_conn, libnode, "_in",
                               Memlet(data=outer_memlet.data, subset=_copy.deepcopy(outer_memlet.subset)))
                state.add_edge(libnode, "_out", local_an, None, local_memlet)

            else:  # direction == 'out'
                # AN(local) --[outer_data[per_iter_subset]]--> MX(IN_Y)
                local_an = edge.src
                mx = edge.dst
                outer_memlet = edge.data
                local_desc = sdfg.arrays[local_an.data]
                outer_desc = sdfg.arrays[outer_memlet.data]

                if not self._storage_allowed(local_desc.storage, outer_desc.storage):
                    continue

                local_subset = dace.subsets.Range.from_array(local_desc)
                local_memlet = Memlet(data=local_an.data, subset=local_subset)

                libnode = CopyLibraryNode(name=f"copy_{local_an.data}_to_{outer_memlet.data}",
                                          src_storage=local_desc.storage,
                                          dst_storage=outer_desc.storage)

                state.remove_edge(edge)
                state.add_node(libnode)
                state.add_edge(local_an, None, libnode, "_in", local_memlet)
                state.add_edge(libnode, "_out", mx, edge.dst_conn,
                               Memlet(data=outer_memlet.data, subset=_copy.deepcopy(outer_memlet.subset)))

            count += 1

        return count
