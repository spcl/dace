# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop fusion: the LoopRegion analogue of MapFusion, for the residual sequential loops left after ``LoopToMap``."""

from typing import Any

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.loop_fusion import LoopFusion


@transformation.explicit_cf_compatible
class FuseLoops(ppl.Pass):
    """Fuse consecutive same-range sequential sibling loops into one loop."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Fuse every qualifying consecutive-loop pair in ``sdfg`` and its nested SDFGs, repeating until no pair matches
        (a chain collapses one adjacency per sweep)."""
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                for cfg in list(sd.all_control_flow_regions()):
                    if isinstance(cfg, ControlFlowRegion) and self._fuse_one(sd, cfg):
                        fused += 1
                        changed = True
                        break
        return fused or None

    @staticmethod
    def _fuse_one(sdfg: SDFG, cfg: ControlFlowRegion) -> bool:
        """Find and fuse one legal consecutive same-range loop pair inside ``cfg`` via ``LoopFusion``."""
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            second = out_edges[0].dst
            if not isinstance(second, LoopRegion) or second is first:
                continue
            if LoopFusion.can_be_applied_to(sdfg, first=first, second=second):
                LoopFusion.apply_to(sdfg, first=first, second=second, verify=False, annotate=False, save=False)
                return True
        return False


__all__ = ["FuseLoops"]
