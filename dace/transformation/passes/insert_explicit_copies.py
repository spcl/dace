# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing implicit copy patterns (path between two access nodes without
and intermediate tasklet) with explicit ``CopyLibraryNode`` instances.
"""
import copy as _copy
from typing import Any, Dict, Iterable, Optional

import dace
from dace import dtypes, nodes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitCopies(ppl.Pass):
    """Replaces implicit copy patterns with ``CopyLibraryNode`` instances.

    Detected patterns:
    - ``AccessNode -> AccessNode`` (direct copy edge)
    - ``AccessNode -> MapEntries -> AccessNode`` (stage-in)
    - ``AccessNode -> MapExits -> AccessNode`` (stage-out)
    """

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
    skip_inside_device_scope = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, copies whose endpoints sit inside a GPU device-level scope are left "
        "alone (cudaMemcpyAsync cannot be issued from device code, and the codegen handles "
        "intra-kernel copies through other paths).",
    )
    inside_device_impl = properties.Property(
        dtype=str,
        default="",
        allow_none=True,
        desc="When set, copies that land inside a GPU device-level scope are lowered with this "
        "implementation (e.g. 'DirectAssignment' or 'pure') and a Sequential schedule, so they "
        "expand to inline code instead of an unsupported cudaMemcpyAsync from device code. "
        "Ignored when ``skip_inside_device_scope`` is True.",
    )

    def __init__(self,
                 src_locations: Optional[Iterable[dtypes.StorageType]] = None,
                 dst_locations: Optional[Iterable[dtypes.StorageType]] = None,
                 skip_inside_device_scope: bool = False,
                 inside_device_impl: Optional[str] = None):
        super().__init__()
        self.src_locations = set(src_locations) if src_locations else set()
        self.dst_locations = set(dst_locations) if dst_locations else set()
        self.skip_inside_device_scope = skip_inside_device_scope
        self.inside_device_impl = inside_device_impl or ""

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
        """Return the number of copy nodes inserted, or ``None`` if nothing changed."""
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

            in_device = (is_devicelevel_gpu(sdfg, state, src_node) or is_devicelevel_gpu(sdfg, state, dst_node))
            if in_device and self.skip_inside_device_scope:
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
            libnode = CopyLibraryNode(name=label)
            self._configure_for_scope(libnode, in_device)

            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, "_in", in_memlet)
            state.add_edge(libnode, "_out", dst_node, None, out_memlet)
            count += 1

        return count

    def _configure_for_scope(self, libnode: 'CopyLibraryNode', in_device_scope: bool) -> None:
        """Pin schedule + implementation when the copy lands inside a GPU device scope."""
        if in_device_scope and self.inside_device_impl:
            libnode.implementation = self.inside_device_impl
            libnode.schedule = dtypes.ScheduleType.Sequential

    def _replace_map_staging_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """Replace map-boundary staging paths with ``CopyLibraryNode`` instances.

        Both staging directions are handled at any nesting depth:
          * ``AccessNode -> MapEntry (-> MapEntry)* -> AccessNode`` (stage-in)
          * ``AccessNode -> (MapExit ->)* MapExit -> AccessNode`` (stage-out)

        The outer AccessNode is resolved via ``state.memlet_path(edge)`` so
        intermediate map scopes between the two access nodes are transparent.
        """
        edges_to_process = []

        for e in state.edges():
            if e.data.is_empty():
                continue
            if isinstance(e.src, nodes.MapEntry) and isinstance(e.dst, nodes.AccessNode):
                direction = 'in'
            elif isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.MapExit):
                direction = 'out'
            else:
                continue

            mpath = state.memlet_path(e)
            # stage-in: innermost edge is (innermost MapEntry -> AccessNode); source of the path is the outer AccessNode.
            # stage-out: innermost edge is (AccessNode -> innermost MapExit); sink of the path is the outer AccessNode.
            outer_an = mpath[0].src if direction == 'in' else mpath[-1].dst
            if not isinstance(outer_an, nodes.AccessNode):
                continue

            edges_to_process.append((direction, e, outer_an))

        count = 0
        for direction, edge, outer_an in edges_to_process:
            if edge not in state.edges():
                continue

            outer_desc = sdfg.arrays[outer_an.data]
            if direction == 'in':
                scope_node, local_an = edge.src, edge.dst
                local_desc = sdfg.arrays[local_an.data]
                src_storage, dst_storage = outer_desc.storage, local_desc.storage
            else:
                local_an, scope_node = edge.src, edge.dst
                local_desc = sdfg.arrays[local_an.data]
                src_storage, dst_storage = local_desc.storage, outer_desc.storage

            if not self._storage_allowed(src_storage, dst_storage):
                continue

            in_device = (is_devicelevel_gpu(sdfg, state, local_an) or is_devicelevel_gpu(sdfg, state, scope_node))
            if in_device and self.skip_inside_device_scope:
                continue

            outer_memlet = edge.data
            local_memlet = Memlet(data=local_an.data, subset=dace.subsets.Range.from_array(local_desc))
            outer_copy = Memlet(data=outer_memlet.data, subset=_copy.deepcopy(outer_memlet.subset))
            name = (f"copy_{outer_an.data}_to_{local_an.data}"
                    if direction == 'in' else f"copy_{local_an.data}_to_{outer_an.data}")
            libnode = CopyLibraryNode(name=name)
            self._configure_for_scope(libnode, in_device)

            state.remove_edge(edge)
            state.add_node(libnode)
            if direction == 'in':
                state.add_edge(scope_node, edge.src_conn, libnode, "_in", outer_copy)
                state.add_edge(libnode, "_out", local_an, None, local_memlet)
            else:
                state.add_edge(local_an, None, libnode, "_in", local_memlet)
                state.add_edge(libnode, "_out", scope_node, edge.dst_conn, outer_copy)

            count += 1

        return count
