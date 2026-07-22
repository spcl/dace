# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split single-element ``AccessNode -> AccessNode`` copies into assign tasklets.

A general-purpose companion to ``InsertAssignTaskletsAtMapBoundary``: while
that pass targets map-boundary staging edges, this one rewrites every plain
``AccessNode -> AccessNode`` copy edge whose moved region is exactly one
element into an ``AccessNode -> (_out = _in) -> AccessNode`` chain. Restricting
to unit copies keeps the rewrite a value-preserving scalar assignment and
removes ``other_subset`` from such edges so subset-substituting passes (e.g.
``NormalizeLoopsAndMaps``) never have to reason about copy memlets.
"""
import copy as _copy
from typing import Any, Dict, Optional

from dace import subsets
from dace import nodes
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


def _is_unit_subset(subset: Optional[subsets.Subset]) -> bool:
    """Return whether ``subset`` provably addresses exactly one element.

    Conservative: a symbolic extent that is not structurally ``1`` counts as
    non-unit, so only copies that are unambiguously single-element qualify.

    :param subset: The subset to test, or ``None``.
    :returns: ``True`` iff every dimension has extent 1 and the volume is 1.
    """
    if subset is None:
        return False
    if subset.num_elements() != 1:
        return False
    return all(sz == 1 for sz in subset.size())


@transformation.explicit_cf_compatible
class InsertAssignTaskletsForUnitCopies(ppl.Pass):
    """Rewrite single-element ``AccessNode -> AccessNode`` copies as assign tasklets."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Split every qualifying unit copy edge in ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results from previously run passes (unused).
        :returns: Number of edges rewritten, or ``None`` if none.
        """
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                count += self._split_unit_copies(nsdfg, state)
        return count if count > 0 else None

    @staticmethod
    def _split_unit_copies(sdfg: SDFG, state: SDFGState) -> int:
        """Rewrite single-element ``AccessNode -> AccessNode`` edges in ``state``.

        Only edges moving exactly one element (on both the subset and, if
        present, ``other_subset``) are rewritten; ``other_subset`` is dropped
        in favour of two clean half-memlets. WCR edges are left to
        ``WCRToAugAssign``.

        :param sdfg: The SDFG owning the array descriptors.
        :param state: The state whose edges are scanned and rewritten.
        :returns: Number of edges replaced.
        """
        edges_to_process = []
        for e in state.edges():
            if e.data is None or e.data.is_empty() or e.data.wcr is not None:
                continue
            if not (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)):
                continue
            mem = e.data
            if not _is_unit_subset(mem.subset):
                continue
            if mem.other_subset is not None and not _is_unit_subset(mem.other_subset):
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
            # half of them. A memlet carrying only one side leaves the other ``None``; derive it from
            # the endpoint's (single-element) array exactly as before.
            src_side = mem.get_src_subset(edge, state)
            dst_side = mem.get_dst_subset(edge, state)
            src_subset = (_copy.deepcopy(src_side)
                          if src_side is not None else subsets.Range.from_array(sdfg.arrays[src_an.data]))
            dst_subset = (_copy.deepcopy(dst_side)
                          if dst_side is not None else subsets.Range.from_array(sdfg.arrays[dst_an.data]))
            tasklet = state.add_tasklet(
                name=f"_assign_{src_an.data}_to_{dst_an.data}",
                inputs={"_in"},
                outputs={"_out"},
                code="_out = _in",
            )
            state.remove_edge(edge)
            state.add_edge(src_an, edge.src_conn, tasklet, "_in", Memlet(data=src_an.data, subset=src_subset))
            state.add_edge(tasklet, "_out", dst_an, edge.dst_conn, Memlet(data=dst_an.data, subset=dst_subset))
            count += 1
        return count
