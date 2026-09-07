# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Condition fusion: the ``ConditionalBlock`` analogue of the loop- and state-fusion passes.

``ConditionFusion`` merges two guarded blocks that share a guard (``if c: A`` then ``if c: B``
becomes ``if c: A; B``), turns opposite guards into one if/else, and flattens a conditional nested
alone inside a branch of another. Left split, the two guards keep their bodies in separate
ConditionalBlocks, and every consumer that works within one block -- map fusion above all -- sees
two halves it cannot join.

The pass is the deterministic, fuse-everything-legal form of that transformation: it applies
``ConditionFusion`` to every qualifying block until a fixpoint. All the legality (the guard
comparison, the "no branch condition reads a symbol the first block assigns" rule) and the merge
itself live on the transformation -- this pass owns only the traversal, so the pass and the
transformation can never disagree.
"""
from typing import Any, Dict, Optional

from dace import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.condition_fusion import ConditionFusion

#: ``ConditionFusion.expressions()`` index for the consecutive-pair match.
CONSECUTIVE = 0
#: ``ConditionFusion.expressions()`` index for the single, nested-alone-in-a-branch match.
NESTED = 1


@transformation.explicit_cf_compatible
class FuseConditions(ppl.Pass):
    """Fuse consecutive and nested ConditionalBlocks until no pair matches."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.CFG | ppl.Modifies.States))

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Fuse every qualifying conditional in ``sdfg`` and its nested SDFGs, to a fixpoint.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of fusions performed, or ``None`` if none.
        """
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if self.fuse_one(sd, cfg):
                        fused += 1
                        changed = True
                        break  # the region list is stale once a block is merged away
        return fused or None

    @staticmethod
    def fuse_one(sdfg: SDFG, cfg: ControlFlowRegion) -> bool:
        """Find and fuse one legal conditional inside ``cfg`` via ``ConditionFusion``.

        Enumerates candidates in node order and delegates every legality decision and the merge to
        :class:`ConditionFusion` -- the shared source of truth. First applicable match wins.

        The consecutive pair is tried before the nested form: collapsing a pair leaves fewer blocks
        for the nested match to walk, and the nested match cannot expose a new pair (it only
        flattens a conditional into the branch that already contained it).

        :param sdfg: The owning SDFG.
        :param cfg: The control-flow region to search (one level; deeper ones are reached by the
                    caller through ``all_control_flow_regions``).
        :returns: ``True`` if a fusion was applied.
        """
        for first in cfg.nodes():
            if not isinstance(first, ConditionalBlock):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            second = out_edges[0].dst
            if not isinstance(second, ConditionalBlock) or second is first:
                continue
            if ConditionFusion.can_be_applied_to(sdfg, expr_index=CONSECUTIVE, cblck1=first, cblck2=second):
                ConditionFusion.apply_to(sdfg,
                                         expr_index=CONSECUTIVE,
                                         cblck1=first,
                                         cblck2=second,
                                         verify=False,
                                         annotate=False,
                                         save=False)
                return True

        for block in cfg.nodes():
            if not isinstance(block, ConditionalBlock):
                continue
            if ConditionFusion.can_be_applied_to(sdfg, expr_index=NESTED, cblck1=block):
                ConditionFusion.apply_to(sdfg,
                                         expr_index=NESTED,
                                         cblck1=block,
                                         verify=False,
                                         annotate=False,
                                         save=False)
                return True
        return False


__all__ = ['FuseConditions']
