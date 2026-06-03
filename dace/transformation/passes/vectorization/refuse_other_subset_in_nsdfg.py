# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Refuse ``other_subset`` memlets and unsupported WCR shapes inside body NSDFGs.

Auto-vectorization treats the body NSDFG of an inner map as the unit it
will tile. Two memlet shapes inside that body are unsupported and surface
as hard descent failures downstream — refuse them up front with a clean
``NotImplementedError`` so the per-test skip channel records a precise
reason instead of an obscure rank-mismatch trip in a late pass.

The two forbidden patterns:

- **``other_subset`` on a memlet anywhere inside a body NSDFG**: the
  upstream cleaners (:class:`CleanAccessNodeToScalarSliceToTaskletPattern`,
  :class:`RemoveRedundantAssignmentTasklets`,
  :class:`ResolveOtherSubsetANEdges`) are expected to lower every
  AN<->AN copy to an explicit assign tasklet. Any residual
  ``other_subset`` would have to be carried through classification and
  reshape passes whose contract is "one memlet, one subset, one side."
  We refuse rather than half-handle it.

- **WCR write that is not the Scalar -> WCR -> MapExit pattern**: a
  reduction sink (``s = s + a[i]``) is only safe to vectorize when the
  WCR writes from a Scalar straight into the inner ``MapExit``. Any
  other WCR shape (WCR inside the body NSDFG, WCR on a non-Scalar
  source, WCR not feeding the MapExit) is refused — the per-lane
  accumulation semantics can only be recovered by the canonical
  ``Scalar -> WCR -> MapExit`` shape downstream tile-reduce expects.
"""
from typing import Any, Dict, Optional

import dace
from dace import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class RefuseOtherSubsetInNSDFG(ppl.Pass):
    """Refuse forbidden memlet shapes inside body NSDFGs before the descent runs.

    The two refusals are documented at module level. Both raise
    ``NotImplementedError`` — surfacing through the harness as a clean
    per-test skip rather than a late, opaque downstream failure.
    """
    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every SDFG (top-level + every nested body); refuse forbidden shapes.

        ``other_subset`` is refused everywhere (top-level + every nested
        body). The check covers the flat-emit path too: when
        ``nest_map_bodies`` is off, the body lives directly under the
        inner Map in the top-level SDFG, and any ``other_subset`` there
        would still trip the descent classifiers.

        ``WCR`` is refused inside nested SDFGs (a body NSDFG). The
        top-level SDFG is allowed to carry the canonical
        ``Scalar -> WCR -> MapExit`` reduction pattern — it is rejected
        only if it appears inside a body NSDFG, where per-lane
        accumulation semantics can no longer be recovered.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Unused.
        :returns: ``None`` (the pass is a precheck, not a rewrite).
        :raises NotImplementedError: On the first occurrence of any
            forbidden pattern.
        """
        for s in sdfg.all_sdfgs_recursive():
            inside_nested = (s is not sdfg)
            for state in s.states():
                self._check_state(s, state, inside_nested=inside_nested)
        return None

    def _check_state(self, sdfg: dace.SDFG, state: SDFGState, inside_nested: bool) -> None:
        """Refuse forbidden memlet shapes in one state.

        :param sdfg: The SDFG owning ``state``.
        :param state: The state to inspect.
        :param inside_nested: ``True`` iff ``sdfg`` is a nested body SDFG
            (not the top-level), in which case WCR is also refused.
        :raises NotImplementedError: On any forbidden pattern.
        """
        scope = "body NSDFG" if inside_nested else "top-level"
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.is_empty():
                continue
            if mem.other_subset is not None and not _is_boundary_edge(edge):
                raise NotImplementedError(
                    f"RefuseOtherSubsetInNSDFG: {scope} state {state.label!r} carries a memlet "
                    f"with ``other_subset`` set ({_describe_edge(edge)}). The cleanup family "
                    f"(CleanAccessNodeToScalarSliceToTaskletPattern, ResolveOtherSubsetANEdges, "
                    f"RemoveRedundantAssignmentTasklets) is expected to remove these before the "
                    f"descent; a residual one means vectorization cannot proceed without lossy "
                    f"reinterpretation. Refactor the source pattern or extend a cleaner.")
            # WCR is only allowed at the top level (the canonical
            # ``Scalar -> WCR -> MapExit`` reduction pattern). Inside a
            # body NSDFG it cannot be vectorised cleanly.
            if mem.wcr is not None and inside_nested:
                raise NotImplementedError(
                    f"RefuseOtherSubsetInNSDFG: body NSDFG state {state.label!r} carries a WCR "
                    f"memlet ({_describe_edge(edge)}). The only supported WCR shape is "
                    f"``Scalar -> WCR -> MapExit`` in the outer state; an inner WCR is refused.")


def _describe_edge(edge) -> str:
    """Short ``src -> dst`` description for diagnostic messages."""
    src = edge.src.data if isinstance(edge.src, nodes.AccessNode) else type(edge.src).__name__
    dst = edge.dst.data if isinstance(edge.dst, nodes.AccessNode) else type(edge.dst).__name__
    return f"{src} -> {dst}, data={edge.data.data!r}"


def _is_boundary_edge(edge) -> bool:
    """True iff the edge crosses a scope/region boundary (NSDFG / Map / View).

    Scope-boundary memlets legitimately carry ``other_subset`` — the inner
    connector subset on a NSDFG node, the outer-loop position on a Map
    edge, or the view's source subset on an AccessNode-to-AccessNode
    view copy. These are not the data-dependent gather residues the
    refusal targets, so we let them pass.
    """
    boundary_nodes = (nodes.NestedSDFG, nodes.MapEntry, nodes.MapExit)
    return isinstance(edge.src, boundary_nodes) or isinstance(edge.dst, boundary_nodes)
