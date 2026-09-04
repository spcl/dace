# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Insert plain assignment tasklets at unit map-boundary / copy edges.

A canonicalization cleanup: it detects the stage-in
(``AccessNode -> MapEntry* -> AccessNode``) and stage-out
(``AccessNode -> MapExit* -> AccessNode``) staging patterns, plus bare
``other_subset`` ``AccessNode -> AccessNode`` copies, and replaces the boundary
edge with a 3-node chain whose middle is an ``_out = _in`` tasklet. This removes
``other_subset`` from the edge (splitting it into two simple src/dst subsets) so
downstream subset-substituting passes never have to reason about copy memlets.

Only **unit** (single-element) copies are rewritten: an ``_out = _in`` tasklet is
a value-preserving copy solely for one element -- for a multi-element slice the
connectors are pointers and ``_out = _in`` is a no-op that silently drops the
copy, so those are left as real (``CopyNDDynamic``) copies. A View's defining
edge is never split. GPU storage/device-scope handling is intentionally omitted.
"""
import copy
from typing import Any, Dict, Optional

import dace
from dace import nodes
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.insert_unit_copy_assign_tasklets import _is_unit_subset


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
            # A View's AccessNode->AccessNode edge (other_subset maps the viewed
            # array onto the view's shape) is the view's DEFINING edge, not a copy
            # to split; inserting a tasklet here breaks get_view_edge ("Ambiguous
            # or invalid edge to/from a View access node").
            if (isinstance(sdfg.arrays[e.src.data], dace.data.View)
                    or isinstance(sdfg.arrays[e.dst.data], dace.data.View)):
                continue
            if e.data.other_subset is None:
                continue
            # An ``_out = _in`` tasklet is a value-preserving copy ONLY for a
            # single element. For a multi-element slice the connectors are typed
            # as pointers, so ``_out = _in`` is a no-op pointer reassignment that
            # silently drops the copy (e.g. vadv's loop-carried
            # ``data_col[:] = dcol[:,:,k]`` became dead, killing the k-recurrence).
            # This pass only handles unit copies; leave multi-element AN->AN
            # copies as real (CopyNDDynamic) copies. Reuse the companion predicate.
            if not (_is_unit_subset(e.data.subset) and _is_unit_subset(e.data.other_subset)):
                continue
            edges_to_process.append(e)

        count = 0
        for edge in edges_to_process:
            if edge not in state.edges():
                continue
            src_an = edge.src
            dst_an = edge.dst
            mem = edge.data
            # Which of ``subset`` / ``other_subset`` names the source is carried by the memlet's own
            # ``_is_data_src`` flag, NOT by the endpoint names: a self-copy ``A -> A`` matches
            # ``mem.data`` on BOTH sides, so a name test picks the source arbitrarily and reverses
            # half of them. ``get_src_subset`` / ``get_dst_subset`` are the only correct readers.
            src_subset = mem.get_src_subset(edge, state)
            dst_subset = mem.get_dst_subset(edge, state)
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
            # A View AccessNode's boundary edge (``MapEntry -> View`` stage-in or
            # ``View -[views]-> MapExit`` stage-out) is the View's DEFINING edge,
            # not a copy to stage. Splitting it with an ``_out=_in`` tasklet
            # destroys the ``views`` connector and leaves the View with code nodes
            # on both sides, so ``get_view_edge`` can no longer identify the
            # defining edge ("Ambiguous or invalid edge to/from a View access
            # node"). Mirrors the guard in ``_replace_other_subset_an_edges``.
            scope_an = e.dst if direction == 'in' else e.src
            if isinstance(sdfg.arrays[scope_an.data], dace.data.View):
                continue
            # The staging split copies the FULL local array via an ``_out = _in``
            # tasklet, which is value-preserving only for a single element -- a
            # multi-element local is typed as a pointer, so ``_out = _in`` is a
            # no-op. Only split unit staging edges (the pass handles unit copies);
            # leave larger ones as real copies.
            if not _is_unit_subset(dace.subsets.Range.from_array(sdfg.arrays[scope_an.data])):
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
