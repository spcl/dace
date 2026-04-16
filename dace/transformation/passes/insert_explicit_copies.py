# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass that replaces implicit copy patterns with explicit
``CopyLibraryNode`` library nodes.

Handles two patterns:

1. **Direct copy edges**: ``AccessNode -> AccessNode``
2. **Map boundary staging**: ``AccessNode -> MapEntry -> AccessNode(transient)`` and
   ``AccessNode(transient) -> MapExit -> AccessNode`` where data is staged in/out of
   a map scope through a transient buffer.

After this pass every data copy in the SDFG is represented by a library
node that can be independently expanded (Layer 3) and specialized.
"""
import copy as _copy
from typing import Any, Dict, Optional

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
    - ``MapEntry -> AccessNode(transient)`` /
      ``AccessNode(transient) -> MapExit`` (map boundary staging)

    The pass sets ``src_storage`` and ``dst_storage`` on each new node
    from the array descriptors.
    """

    CATEGORY = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

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

    # -----------------------------------------------------------------
    # Pattern 1: direct AccessNode -> AccessNode edges
    # -----------------------------------------------------------------

    @staticmethod
    def _replace_direct_copies(sdfg: SDFG, state: SDFGState) -> int:
        edges = list(state.edges())
        count = 0
        for edge in edges:
            if not (isinstance(edge.src, nodes.AccessNode)
                    and isinstance(edge.dst, nodes.AccessNode)):
                continue

            src_node: nodes.AccessNode = edge.src
            dst_node: nodes.AccessNode = edge.dst
            memlet: Memlet = edge.data

            if memlet.is_empty():
                continue

            src_desc = sdfg.arrays[src_node.data]
            dst_desc = sdfg.arrays[dst_node.data]

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

    # -----------------------------------------------------------------
    # Pattern 2: map boundary staging (ME -> AN or AN -> MX)
    # -----------------------------------------------------------------

    @staticmethod
    def _replace_map_staging_copies(sdfg: SDFG, state: SDFGState) -> int:
        """
        Detect edges ``MapEntry -> AccessNode(transient)`` and
        ``AccessNode(transient) -> MapExit`` that stage data in/out of
        a map scope.  Insert a CopyLibraryNode at each such boundary.
        """
        edges_to_process = []

        for e in state.edges():
            if e.data.is_empty():
                continue
            # MapEntry -> AccessNode(transient)
            if (isinstance(e.src, nodes.MapEntry)
                    and isinstance(e.dst, nodes.AccessNode)):
                desc = sdfg.arrays.get(e.dst.data)
                if desc is not None and desc.transient:
                    edges_to_process.append(('in', e))
            # AccessNode(transient) -> MapExit
            elif (isinstance(e.src, nodes.AccessNode)
                  and isinstance(e.dst, nodes.MapExit)):
                desc = sdfg.arrays.get(e.src.data)
                if desc is not None and desc.transient:
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

                local_subset = dace.subsets.Range.from_array(local_desc)
                local_memlet = Memlet(data=local_an.data,
                                      subset=local_subset)

                libnode = CopyLibraryNode(
                    name=f"copy_{outer_memlet.data}_to_{local_an.data}",
                    src_storage=outer_desc.storage,
                    dst_storage=local_desc.storage)

                state.remove_edge(edge)
                state.add_node(libnode)
                state.add_edge(me, edge.src_conn, libnode, "_in",
                               Memlet(data=outer_memlet.data,
                                      subset=_copy.deepcopy(
                                          outer_memlet.subset)))
                state.add_edge(libnode, "_out", local_an, None,
                               local_memlet)

            else:  # direction == 'out'
                # AN(local) --[outer_data[per_iter_subset]]--> MX(IN_Y)
                local_an = edge.src
                mx = edge.dst
                outer_memlet = edge.data
                local_desc = sdfg.arrays[local_an.data]
                outer_desc = sdfg.arrays[outer_memlet.data]

                local_subset = dace.subsets.Range.from_array(local_desc)
                local_memlet = Memlet(data=local_an.data,
                                      subset=local_subset)

                libnode = CopyLibraryNode(
                    name=f"copy_{local_an.data}_to_{outer_memlet.data}",
                    src_storage=local_desc.storage,
                    dst_storage=outer_desc.storage)

                state.remove_edge(edge)
                state.add_node(libnode)
                state.add_edge(local_an, None, libnode, "_in",
                               local_memlet)
                state.add_edge(libnode, "_out", mx, edge.dst_conn,
                               Memlet(data=outer_memlet.data,
                                      subset=_copy.deepcopy(
                                          outer_memlet.subset)))

            count += 1

        return count