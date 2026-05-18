# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Permute perfectly-nested loops to maximize unit-stride array accesses.

The ``LoopRegion`` analogue of ``MinimizeStridePermutation``, intended to run
before ``LoopToMap`` so the parallel maps it produces already have the
contiguous axis innermost. Like the map version it must stay conservative:
if the per-loop stride order cannot be deduced from concrete (symbol-free)
strides it must do nothing rather than guess. Not implemented yet -- this is
a structurally-complete no-op placeholder so the pipeline has a real, named
slot before ``LoopToMap``.
"""
from typing import Any, Dict, Optional

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class LoopStridePermutation(ppl.Pass):
    """Reorder loop nests for unit stride (no-op until implemented).

    When implemented it must reuse the conservative scoring of
    ``MinimizeStridePermutation``: undeducible/symbolic strides => no
    permutation (a safe, idempotent no-op rather than a guess).
    """
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """No-op until loop interchange for unit-stride is implemented.

        :param sdfg: The SDFG (unmodified).
        :param pipeline_results: Results from previous passes (unused).
        :returns: ``None`` (nothing changed).
        """
        return None
