# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Permute perfectly-nested loops to maximize unit-stride array accesses.

The ``LoopRegion`` analogue of ``MinimizeStridePermutation``, slotted before
``LoopToMap``. Intentionally a **no-op**: a sound loop-level interchange
would need a loop-interchange primitive (none exists) plus loop-carried
dependence analysis. The canonicalization pipeline instead permutes the
loops that *can* become maps as maps, right after ``LoopToMap``, via the
proven, symbolic-safe ``MinimizeStridePermutation`` (dependence-free by the
Map contract). This named slot is kept so a future direct loop-level
implementation has an honest place; until then the work happens post-L2M.
"""
from typing import Any, Dict, Optional, Set

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class LoopStridePermutation(ppl.Pass):
    """Reorder loop nests for unit stride -- intentional no-op (see module
    docstring): the pipeline permutes map-eligible loops as maps post-L2M
    via the symbolic-safe ``MinimizeStridePermutation`` instead."""
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Intentional no-op; stride permutation runs post-LoopToMap on maps.

        :param sdfg: The SDFG (unmodified).
        :param pipeline_results: Results from previous passes (unused).
        :returns: ``None`` (nothing changed).
        """
        return None
