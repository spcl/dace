# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-emit audit pass that refuses any per-lane index symbol leak.

Per TILIFICATION_TRANSFORMATION_DESIGN.md section 10.6: after the
multi-dim tile vectorization pipeline finishes, the K-dim path must
have replaced every per-lane symbol with index tiles materialised by
:class:`PreparePerLaneIndices` (G8). A residual ``<base>_laneid_<n>``
(legacy 1D form) or ``<base>_lane<d>id_<n>`` (canonical multi-dim
form) anywhere in the SDFG indicates a prep pass accidentally fanned
out to per-lane scalars; the pure expansion would then mis-lower.

This pass walks the entire SDFG (recursing into nested SDFGs) and
raises ``AssertionError`` on the first leak detected. Designed to
run as the final step in :class:`VectorizeCPUMultiDim`'s pipeline so
any contract violation surfaces loudly at orchestrator exit.

Reuse: the audit logic lives in
:func:`utils.name_schemes.assert_no_laneid_in_tile_path` (existing
helper); this module just wraps it as a :class:`ppl.Pass` so the
pipeline can consume it.
"""
from typing import Any, Dict, Optional

from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.name_schemes import assert_no_laneid_in_tile_path


@transformation.explicit_cf_compatible
class ClearPerLaneIndexSymbols(ppl.Pass):
    """Post-emit audit pass (design section 10.6).

    Refuses any per-lane scalar symbol in the post-emit SDFG. The check
    is a delegation to :func:`assert_no_laneid_in_tile_path`, which
    walks every array name, SDFG symbol, tasklet connector, and
    interstate-edge assignment target (recursively through nested
    SDFGs) and classifies each via
    :meth:`LaneIdScheme.is_lane_fanned`. Any leak raises
    ``AssertionError`` naming the offending symbols.

    Idempotent / does not modify the SDFG.
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the audit; raise on any leak.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused).
        :returns: ``None`` (no rewrites; audit-only).
        :raises AssertionError: If any per-lane symbol survives in the
            SDFG. The message lists the offending names.
        """
        assert_no_laneid_in_tile_path(sdfg)
        return None
