# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Resolve residual ``AccessNode -[other_subset]-> AccessNode`` edges.

Auto-vectorization does not support ``other_subset`` on a memlet: the
downstream classify / promote / fan-out machinery walks tasklet in/out-edges,
so any AccessNode -> AccessNode edge that re-shapes the access via
``other_subset`` is invisible to it. The forward cleaner
:class:`CleanAccessNodeToScalarSliceToTaskletPattern` plus
:class:`RemoveRedundantAssignmentTasklets` are expected to canonicalise
trivial scalar-slice intermediaries to a clean tasklet-bridged form; the
latter, however, may collapse the bridging assign tasklet into a single
AN -> AN edge with ``other_subset`` set whenever the in / out subsets
disagree (vag-style ``a[i] = b[ip[i]]``: removing the tasklet leaves
``b -[data=a, subset=[i], other_subset=[ip_index]]-> a``).

This pass is the safety net that runs after both cleaners:

- If the edge carries 1-element subsets on both sides, reinsert an
  ``_out = _in`` assign tasklet between the two access nodes so the
  classify / promote walkers can see the read and the write.
- Otherwise (a genuinely multi-element ``other_subset`` view copy) raise
  ``NotImplementedError`` — the vectorization pipeline cannot lower it.

Scoped to inner-body NSDFGs ONLY by ``apply_pass``: we deliberately leave
outer-map staging edges alone (some legacy 1D paths rely on them).
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
class ResolveOtherSubsetANEdges(ppl.Pass):
    """Reinsert assign tasklets for 1-element residual ``AN -[os]-> AN`` edges; raise otherwise.

    Walks every inner-body NestedSDFG and inspects ``AccessNode ->
    AccessNode`` edges whose memlet has ``other_subset`` set. Bails out
    loudly on any edge whose source or destination subset has > 1 element
    (the vectorization pipeline does not support such shapes); otherwise
    reinserts an ``_out = _in`` tasklet to keep the data flow visible
    to the descent's classify / promote / fan-out walkers.
    """
    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    @staticmethod
    def _subsets_for(edge, src_an: nodes.AccessNode, dst_an: nodes.AccessNode):
        """Return ``(src_subset, dst_subset)`` deep copies for an AN -> AN edge.

        The memlet's ``data`` selects which side carries ``subset`` and which
        carries ``other_subset``; normalise both halves so the caller can
        thread them into a tasklet-bridged pair without re-classifying.
        """
        mem = edge.data
        if mem.data == src_an.data:
            return mem.subset, mem.other_subset
        return mem.other_subset, mem.subset

    def _resolve_state(self, sdfg: SDFG, state: SDFGState) -> int:
        """Process one state; raise on multi-element shapes, rewrite 1-element ones."""
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
            src_an, dst_an = edge.src, edge.dst
            src_subset, dst_subset = self._subsets_for(edge, src_an, dst_an)
            try:
                src_n = src_subset.num_elements()
                dst_n = dst_subset.num_elements()
                src_one = (dace.symbolic.simplify(src_n) == 1)
                dst_one = (dace.symbolic.simplify(dst_n) == 1)
                # 1-dim guard: a multi-dim point access (icon-style
                # ``z_kin_hor_e[edge_blk_idx, jk, edge_idx_idx]`` with both
                # subset and other_subset 1-element) is *already* the form the
                # descent's mixed-gather path consumes through the AN -> AN
                # pattern. Reinserting a tasklet around it would re-expose the
                # multi-dim subset to ``_collapse_tile_gathers``, which only
                # emits 1-dim index memlets (lib-node ``_idx_k`` carries the
                # widened ``(W,)`` index tile). The 1-dim guard limits the
                # rewrite to the genuine vag-style ``a[i] = b[ip[i]]`` shape
                # (both subsets are 1-dim 1-element).
                src_one_dim = (len(src_subset) == 1)
                dst_one_dim = (len(dst_subset) == 1)
            except (TypeError, AttributeError):
                src_one = dst_one = src_one_dim = dst_one_dim = False
            if not (src_one and dst_one):
                raise NotImplementedError(
                    f"ResolveOtherSubsetANEdges: cannot lower AccessNode->AccessNode edge "
                    f"{src_an.data}[{src_subset}] -> {dst_an.data}[{dst_subset}] via other_subset; "
                    f"auto vectorization does not support multi-element other_subset edges "
                    f"(state {state.label!r}). Refactor the source SDFG to route the copy "
                    f"through an assign tasklet, or extend the cleaner family.")
            if not (src_one_dim and dst_one_dim):
                # Multi-dim point access — leave the AN -> AN edge for the
                # descent's mixed-gather path. (No rewrite, no raise.)
                continue
            # Reinsert the assign tasklet so classify/promote walkers see it.
            tasklet = state.add_tasklet(name=f"_assign_{src_an.data}_to_{dst_an.data}",
                                        inputs={"_in"},
                                        outputs={"_out"},
                                        code="_out = _in")
            state.remove_edge(edge)
            state.add_edge(src_an, edge.src_conn, tasklet, "_in",
                           Memlet(data=src_an.data, subset=copy.deepcopy(src_subset)))
            state.add_edge(tasklet, "_out", dst_an, edge.dst_conn,
                           Memlet(data=dst_an.data, subset=copy.deepcopy(dst_subset)))
            count += 1
        return count

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Resolve residual AN -[other_subset]-> AN edges inside body NestedSDFGs.

        Only walks SDFGs reached through a :class:`NestedSDFG` node — the
        outer SDFG's AN -> AN edges may be scatter / gather staging that the
        legacy 1D detect passes (``DetectGather`` / ``DetectScatter``) handle,
        and rewriting them here would orphan the per-lane fan-outs they
        produce. The tile-path descent only descends into body NSDFGs, so
        scoping the residual cleanup the same way avoids double-handling
        edges those legacy detectors already understand.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Unused.
        :returns: Number of edges rewritten, or ``None`` if zero.
        :raises NotImplementedError: When an edge with multi-element subsets remains.
        """
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            if nsdfg is sdfg:
                continue
            for state in nsdfg.states():
                count += self._resolve_state(nsdfg, state)
        return count if count > 0 else None
