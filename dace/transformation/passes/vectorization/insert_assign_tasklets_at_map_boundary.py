# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``InsertAssignTaskletsAtMapBoundary``: a vectorization-focused special case of
``InsertExplicitCopies`` (from ``new-gpu-codegen-dev``). Detects the two
map-boundary staging patterns and inserts a plain ``_out = _in`` tasklet
instead of a ``CopyLibraryNode``.

Patterns handled (same detection as ``InsertExplicitCopies._replace_map_staging_copies``):
  * Stage-in:  ``AccessNode -> MapEntry (-> MapEntry)* -> AccessNode``
  * Stage-out: ``AccessNode -> (MapExit ->)* MapExit -> AccessNode``

For each match, the inner ``MapEntry -> AccessNode`` (or ``AccessNode -> MapExit``)
edge is replaced with a 3-node chain whose middle is a plain assignment
tasklet. The outer-side memlet keeps its data + subset; the inner-side
memlet becomes a single-element write/read on the local AccessNode.

Why a vectorization-specific variant:
- The python frontend's parser emits ``MapEntry -> AccessNode`` edges with
  ``other_subset`` set (for shifted reads like ``b[i+1]``). The vectorize
  pass raises on ``other_subset != None``. The assignment tasklet
  inserted by this pass splits the memlet into two simple subsets, so
  downstream emitters see uniform shapes.
- Unlike the lib-node version, this emits a Python tasklet that the
  vectorizer's existing tasklet classifier / emitter recognises and
  lowers to ``vector_copy(...)`` per the unified pipeline.

GPU storage filters and device-scope handling from ``InsertExplicitCopies``
are intentionally omitted; this pass runs unconditionally as a vectorization
precursor.
"""
import copy as _copy
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
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                count += self._replace_map_staging(nsdfg, state)
                count += self._replace_other_subset_an_edges(nsdfg, state)
        return count if count > 0 else None

    @staticmethod
    def _replace_other_subset_an_edges(sdfg: SDFG, state: SDFGState) -> int:
        """Replace ``AccessNode -[other_subset]→ AccessNode`` edges with an
        intermediate assign tasklet so downstream passes never see
        ``other_subset`` set.

        The frontend lowers a scalar slice like ``a0 = a[0]`` into
        ``a -[data=a, subset=0, other_subset=0]→ a0`` (a single edge that
        carries both ends' subsets). The vectorize pipeline's
        ``no_other_subset`` invariant rejects this shape. Inserting a plain
        ``_out = _in`` tasklet between the two AccessNodes splits the
        memlet into two clean halves (``a -[subset=0]→ tasklet`` +
        ``tasklet -[a0, subset=0]→ a0``).
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
                           Memlet(data=src_an.data, subset=_copy.deepcopy(src_subset)))
            state.add_edge(tasklet, "_out", dst_an, edge.dst_conn,
                           Memlet(data=dst_an.data, subset=_copy.deepcopy(dst_subset)))
            count += 1
        return count

    @staticmethod
    def _replace_map_staging(sdfg: SDFG, state: SDFGState) -> int:
        """Replace ``AccessNode -> MapEntry -> AccessNode`` (stage-in) and
        ``AccessNode -> MapExit -> AccessNode`` (stage-out) paths with an
        intermediate assignment tasklet.

        Detection mirrors ``InsertExplicitCopies._replace_map_staging_copies``:
        the innermost edge of the memlet_path identifies the staging boundary;
        the outer AccessNode is the path's source (stage-in) or sink (stage-out).
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
                outer_copy = Memlet(data=outer_memlet.data, subset=_copy.deepcopy(outer_memlet.subset))
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
                outer_copy = Memlet(data=outer_memlet.data, subset=_copy.deepcopy(outer_memlet.subset))
                tasklet = state.add_tasklet(name=f"_assign_out_{local_an.data}_to_{outer_an.data}",
                                            inputs={"_in"},
                                            outputs={"_out"},
                                            code="_out = _in")
                state.remove_edge(edge)
                state.add_edge(local_an, None, tasklet, "_in", local_memlet)
                state.add_edge(tasklet, "_out", scope_node, edge.dst_conn, outer_copy)
                count += 1

        return count
