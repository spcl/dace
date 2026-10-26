# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrites copy memlets into the canonical form named by ``copy_memlet_owner``."""

from typing import Any

from dace import properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.utils import normalize_copy_memlet
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeCopyMemlets(ppl.Pass):
    """Makes every copy memlet name the container ``copy_memlet_owner`` asks for: the one outside the scope its
    path crosses, or the destination of a direct copy. ``subset`` then belongs to it, ``other_subset`` to the
    other end."""

    CATEGORY: str = "Cleanup"

    __slots__ = ()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        for edge, state in sdfg.all_edges_recursive():
            if isinstance(state, SDFGState):
                count += normalize_copy_memlet(state, edge)
        return count or None
