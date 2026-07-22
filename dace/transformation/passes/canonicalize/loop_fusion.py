# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop fusion: the LoopRegion analogue of MapFusion, for the residual
sequential loops left after ``LoopToMap``.

``MapFusionVertical`` / ``MapFusionHorizontal`` only ever fuse ``MapEntry``
nodes, so two consecutive sibling loops that ``LoopToMap`` refused (recurrences,
Thomas sweeps, sequential scans) are never fused. This pass fuses such a pair --
same iteration space, single-compute-state bodies -- into one loop whose body
runs the first body then the second per iteration. It is the exact inverse of
``LoopFission`` and shares its soundness kernel (``break_anti_dependence``'s
``_dep_class`` and ``loop_fission``'s per-iteration-subset reasoning), so a pair
that fission would have separated is a pair that fusion may legally rejoin.

Designed to run POST-``LoopToMap`` (every DOALL loop is already a Map, hence not
a candidate), so it only ever sees sequential residual loops and cannot serialize
parallel work. A defensive ``LoopToMap.can_be_applied_to`` guard refuses to fuse
a loop that is independently DOALL, keeping the pass correct even out of position.

Legality (per array touched by both bodies, under the shared iterator):
  * flow (write in body1, read in body2): illegal if the read is READ-AHEAD
    (``a[i+k]``, k>0) -- the fused loop reads a not-yet-produced value.
  * anti (read in body1, write in body2): illegal if the read is READ-BEHIND
    (``a[i-k]``, k>0) of what body2 overwrites earlier in the fused sweep.
  * output (write in both): illegal unless the two writes hit the same index.
Symbolic / indirected / complex offsets are refused conservatively (v1).
"""
from typing import Any, Dict, Optional

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.fuse_loops import FuseLoops


@transformation.explicit_cf_compatible
class LoopFusion(ppl.Pass):
    """Fuse consecutive same-range sequential sibling loops into one loop.

    The deterministic, fuse-everything-legal form of the ``FuseLoops`` transformation: it applies
    ``FuseLoops`` to every qualifying consecutive-loop pair until a fixpoint. All the legality (same
    iteration space, single-compute-state bodies, not independently DOALL, value-preserving dependence
    classes) and the merge itself live on ``FuseLoops`` -- this pass owns only the traversal, so the pass
    and the transformation can never disagree.
    """
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Fuse every qualifying consecutive-loop pair in ``sdfg`` and its nested
        SDFGs, repeating until no pair matches (a chain collapses one adjacency
        per sweep).

        :param sdfg: The SDFG to transform in place.
        :returns: Number of fusions performed, or ``None`` if none.
        """
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if self._fuse_one(sd, cfg):
                        fused += 1
                        changed = True
                        break
        return fused or None

    @staticmethod
    def _fuse_one(sdfg: SDFG, cfg: ControlFlowRegion) -> bool:
        """Find and fuse one legal consecutive same-range loop pair inside ``cfg`` via ``FuseLoops``.

        Enumerates candidate adjacencies (a ``LoopRegion`` with a single out-edge to a ``LoopRegion``) in
        node order and delegates the legality decision and the merge to :class:`FuseLoops` -- the shared
        source of truth. First applicable pair wins, matching the original one-adjacency-per-sweep order.

        :param sdfg: The owning SDFG (for the DOALL oracle FuseLoops consults).
        :param cfg: The control-flow region to search (one level; deeper loops are reached via
                    ``all_control_flow_regions``).
        :returns: ``True`` if a pair was fused.
        """
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            second = out_edges[0].dst
            if not isinstance(second, LoopRegion) or second is first:
                continue
            if FuseLoops.can_be_applied_to(sdfg, first=first, second=second):
                FuseLoops.apply_to(sdfg, first=first, second=second, verify=False, annotate=False, save=False)
                return True
        return False


__all__ = ['LoopFusion']
