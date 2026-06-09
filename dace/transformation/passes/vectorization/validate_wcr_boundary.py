# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass-level validator for the design section 3.5 WCR boundary contract.

WCR (write-conflict-resolution) memlets are locked to a single canonical
shape: a memlet on an edge ``AccessNode -> MapExit`` whose source
AccessNode descriptor is a length-1 array (or :class:`dace.data.Scalar`).
Anywhere else -- inside a body NSDFG, on a non-boundary edge, on a
non-single-element AN -- the WCR would race or break the design's
reduction semantics. The audit walks the full SDFG (recursing through
nested SDFGs) and raises :class:`NotImplementedError` on the first
violation, naming the edge and the failure mode.
"""
from typing import Any, Dict, Optional

import dace
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapExit
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class ValidateWCRBoundary(ppl.Pass):
    """Refuse any WCR memlet outside the locked ``AN -> MapExit`` shape.

    The check fires at any expand-time / orchestrator-exit hook. Idempotent
    and never modifies the SDFG; on failure raises ``NotImplementedError``.
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every edge in every state of every nested SDFG; refuse WCR
        memlets outside the locked shape.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused).
        :returns: ``None`` (audit-only).
        :raises NotImplementedError: On the first non-locked WCR.
        """
        for inner in sdfg.all_sdfgs_recursive():
            for state in inner.states():
                for edge in state.edges():
                    if edge.data is None or edge.data.wcr is None:
                        continue
                    if not isinstance(edge.src, AccessNode):
                        raise NotImplementedError(
                            f"ValidateWCRBoundary: WCR memlet on edge {edge.src} -> {edge.dst} has "
                            f"non-AccessNode source (got {type(edge.src).__name__}); design section 3.5 "
                            f"locks WCR to AN -> MapExit boundary.")
                    if not isinstance(edge.dst, MapExit):
                        raise NotImplementedError(
                            f"ValidateWCRBoundary: WCR memlet on edge {edge.src.data} -> {edge.dst} has "
                            f"non-MapExit destination (got {type(edge.dst).__name__}); design section 3.5 "
                            f"locks WCR to AN -> MapExit boundary.")
                    desc = inner.arrays.get(edge.src.data)
                    if desc is None:
                        raise NotImplementedError(
                            f"ValidateWCRBoundary: WCR memlet source {edge.src.data!r} has no descriptor.")
                    if not _is_single_element(desc):
                        raise NotImplementedError(
                            f"ValidateWCRBoundary: WCR memlet source {edge.src.data!r} has descriptor "
                            f"shape {tuple(desc.shape)} -- design section 3.5 requires a single-element "
                            f"AccessNode (Scalar or length-1 Array).")
        return None


def _is_single_element(desc) -> bool:
    """True iff ``desc`` describes a single addressable element (Scalar or
    length-1 Array)."""
    if isinstance(desc, dace.data.Scalar):
        return True
    if isinstance(desc, dace.data.Array):
        try:
            return all(bool(dace.symbolic.simplify(s - 1) == 0) for s in desc.shape)
        except Exception:  # noqa: BLE001 -- symbolic shape that can't be simplified to 1 is rejected.
            return False
    return False
