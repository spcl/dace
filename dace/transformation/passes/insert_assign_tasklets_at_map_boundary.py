# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Insert plain assignment tasklets at map-boundary staging edges.

A vectorization-focused variant of ``InsertExplicitCopies``: it detects
the stage-in (``AccessNode -> MapEntry* -> AccessNode``) and stage-out
(``AccessNode -> MapExit* -> AccessNode``) staging patterns and replaces
the boundary edge with a 3-node chain whose middle is an ``_out = _in``
tasklet. This splits memlets carrying ``other_subset`` into two simple
subsets so downstream vectorization emitters see uniform shapes. GPU
storage/device-scope handling is intentionally omitted.
"""
import copy
from typing import Any, Dict, Optional

import dace
from dace import nodes
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class InsertAssignTaskletsAtMapBoundary(ppl.Pass):
    """Insert ``_out = _in`` tasklets at map-boundary staging edges."""
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Insert assignment tasklets at every matching staging edge.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results from previously run passes (unused).
        :returns: Number of tasklets inserted, or ``None`` if none.
        """
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                count += self._replace_map_staging(nsdfg, state)
                count += self._replace_other_subset_an_edges(nsdfg, state)
        return count if count > 0 else None

    @staticmethod
    def _replace_other_subset_an_edges(sdfg: SDFG, state: SDFGState) -> int:
        """Split ``AccessNode -[other_subset]-> AccessNode`` edges via an intermediate assign tasklet.

        The inserted ``_out = _in`` tasklet splits the memlet into two
        clean halves so downstream passes never see ``other_subset`` set.

        :param sdfg: The SDFG owning the array descriptors.
        :param state: The state whose edges are scanned and rewritten.
        :returns: Number of edges replaced.
        """
        edges_to_process = []
        for e in state.edges():
            if e.data is None or e.data.is_empty():
                continue
            if not (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)):
                continue
            if e.data.other_subset is None:
                continue
            edges_to_process.append(e)

        count = 0
        for edge in edges_to_process:
            if edge not in state.edges():
                continue
            src_an = edge.src
            dst_an = edge.dst
            mem = edge.data
            # Identify which side of the memlet matches each AccessNode.
            if mem.data == src_an.data:
                src_subset = mem.subset
                dst_subset = mem.other_subset
            else:
                assert mem.data == dst_an.data
                src_subset = mem.other_subset
                dst_subset = mem.subset
            tasklet = state.add_tasklet(
                name=f"_assign_{src_an.data}_to_{dst_an.data}",
                inputs={"_in"},
                outputs={"_out"},
                code="_out = _in",
            )
            state.remove_edge(edge)
            state.add_edge(src_an, edge.src_conn, tasklet, "_in",
                           Memlet(data=src_an.data, subset=copy.deepcopy(src_subset)))
            # Preserve the original edge's WCR on the OUTPUT side. The split
            # turns ``AccessNode -[wcr]-> AccessNode`` into ``... -> tasklet
            # -[wcr]-> AccessNode``; the WCR semantics still apply at the
            # write into ``dst_an``. Dropping the WCR here corrupts reductions
            # (LoopToReduce's wcr-scalar emit lands the WCR on the
            # AccessNode-to-AccessNode edge that this method splits).
            state.add_edge(tasklet, "_out", dst_an, edge.dst_conn,
                           Memlet(data=dst_an.data, subset=copy.deepcopy(dst_subset), wcr=mem.wcr))
            count += 1
        return count

    @staticmethod
    def _replace_map_staging(sdfg: SDFG, state: SDFGState) -> int:
        """Replace map-boundary stage-in/stage-out paths with an intermediate assign tasklet.

        The innermost memlet-path edge identifies the staging boundary; the
        outer AccessNode is the path's source (stage-in) or sink (stage-out).

        :param sdfg: The SDFG owning the array descriptors.
        :param state: The state whose staging edges are rewritten.
        :returns: Number of staging edges replaced.
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
            # WCR is reduction semantics on the WRITE side and is only valid on
            # stage-out edges (``... -> MapExit``). A WCR-carrying stage-in
            # edge (``MapEntry -> AccessNode``) has no defined meaning — fail
            # loudly rather than silently strip the WCR.
            if direction == 'in' and e.data.wcr is not None:
                raise ValueError(f"InsertAssignTaskletsAtMapBoundary: stage-in edge "
                                 f"{e.src} -> {e.dst} (data {e.data.data!r}) carries WCR "
                                 f"{e.data.wcr!r}; WCR is only valid on stage-out edges "
                                 f"into MapExit.")

            mpath = state.memlet_path(e)
            outer_an = mpath[0].src if direction == 'in' else mpath[-1].dst
            if not isinstance(outer_an, nodes.AccessNode):
                continue

            edges_to_process.append((direction, e, outer_an))

        count = 0
        for direction, edge, outer_an in edges_to_process:
            if edge not in state.edges():
                continue

            outer_memlet: Memlet = edge.data
            if direction == 'in':
                scope_node = edge.src  # innermost MapEntry
                local_an = edge.dst  # AccessNode inside the scope
                local_desc = sdfg.arrays[local_an.data]
                local_memlet = Memlet(data=local_an.data, subset=dace.subsets.Range.from_array(local_desc))
                outer_copy = Memlet(data=outer_memlet.data, subset=copy.deepcopy(outer_memlet.subset))
                tasklet = state.add_tasklet(name=f"_assign_in_{outer_an.data}_to_{local_an.data}",
                                            inputs={"_in"},
                                            outputs={"_out"},
                                            code="_out = _in")
                state.remove_edge(edge)
                state.add_edge(scope_node, edge.src_conn, tasklet, "_in", outer_copy)
                state.add_edge(tasklet, "_out", local_an, None, local_memlet)
                count += 1
            else:
                local_an = edge.src  # AccessNode inside the scope
                scope_node = edge.dst  # innermost MapExit
                local_desc = sdfg.arrays[local_an.data]
                local_memlet = Memlet(data=local_an.data, subset=dace.subsets.Range.from_array(local_desc))
                # Preserve the original stage-out edge's WCR on the output
                # side. The split turns ``AccessNode -[wcr]-> MapExit`` into
                # ``... -> tasklet -[wcr]-> MapExit``; dropping the WCR here
                # corrupts reductions whose privatised scalar reaches the map
                # exit via an AccessNode-to-MapExit edge with WCR.
                outer_copy = Memlet(data=outer_memlet.data,
                                    subset=copy.deepcopy(outer_memlet.subset),
                                    wcr=outer_memlet.wcr)
                tasklet = state.add_tasklet(name=f"_assign_out_{local_an.data}_to_{outer_an.data}",
                                            inputs={"_in"},
                                            outputs={"_out"},
                                            code="_out = _in")
                state.remove_edge(edge)
                state.add_edge(local_an, None, tasklet, "_in", local_memlet)
                state.add_edge(tasklet, "_out", scope_node, edge.dst_conn, outer_copy)
                count += 1

        return count
