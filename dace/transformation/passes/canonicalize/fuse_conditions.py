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
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dace import SDFG
from dace.config import Config
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.passes.pattern_matching import child_regions

#: ``ConditionFusion.expressions()`` index for the consecutive-pair match.
CONSECUTIVE = 0
#: ``ConditionFusion.expressions()`` index for the single, nested-alone-in-a-branch match.
NESTED = 1


def matcher_candidates(region: AbstractControlFlowRegion) -> Iterator[Tuple[int, Dict[Any, ConditionalBlock]]]:
    """``(expr_index, binding)`` of every ``ConditionFusion`` candidate in ``region``, in the order
    ``match_patterns`` enumerates them: every consecutive pair in the collapsed graph's edge order
    (source by node order, then first appearance of the edge), then every block for the nested form.
    """
    order = {block: i for i, block in enumerate(region.nodes())}
    pairs: Dict[Tuple[Any, Any], None] = {}
    for edge in region.edges():
        pairs.setdefault((edge.src, edge.dst), None)
    for first, second in sorted(pairs, key=lambda pair: order[pair[0]]):
        if first is not second and isinstance(first, ConditionalBlock) and isinstance(second, ConditionalBlock):
            yield CONSECUTIVE, {ConditionFusion.cblck1: first, ConditionFusion.cblck2: second}
    for block in region.nodes():
        if isinstance(block, ConditionalBlock):
            yield NESTED, {ConditionFusion.cblck1: block}


def chain_candidates(block: AbstractControlFlowRegion) -> Iterator[Tuple[AbstractControlFlowRegion, int, Dict]]:
    """The candidates naming ``block`` or one of its ancestors, in the region holding each, up to the SDFG."""
    while not isinstance(block, SDFG) and block.parent_graph is not None:
        region = block.parent_graph
        if isinstance(block, ConditionalBlock):
            for pred in region.predecessors(block):
                if isinstance(pred, ConditionalBlock) and pred is not block:
                    yield region, CONSECUTIVE, {ConditionFusion.cblck1: pred, ConditionFusion.cblck2: block}
            for succ in region.successors(block):
                if isinstance(succ, ConditionalBlock) and succ is not block:
                    yield region, CONSECUTIVE, {ConditionFusion.cblck1: block, ConditionFusion.cblck2: succ}
            yield region, NESTED, {ConditionFusion.cblck1: block}
        block = region


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

    def __init__(self, matcher_order: bool = False) -> None:
        """
        :param matcher_order: Fuse in ``PatternApplyOnceEverywhere([ConditionFusion()])`` order, the
                              order the canonicalization recipe was built on, instead of per SDFG.
        """
        self.matcher_order = matcher_order

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Fuse every qualifying conditional in ``sdfg`` and its nested SDFGs, to a fixpoint.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of fusions performed, or ``None`` if none.
        """
        if self.matcher_order:
            return self.apply_in_matcher_order(sdfg)
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

    def apply_in_matcher_order(self, sdfg: SDFG) -> Optional[int]:
        """Fuse every conditional ``PatternApplyOnceEverywhere([ConditionFusion()])`` would, in its order.

        The matcher walks every region of ``all_control_flow_regions(recursive=True)`` again after every
        application: quadratic in the region count (warpx_field_gather: 1.5M probes for 978 fusions). What
        ``ConditionFusion`` reads is local -- a pair's own region and the pair's subtrees, a nested block's
        region and that region's parent -- and what it writes is the fused region (a pair) or the parent
        conditional (the nested form). So every candidate the walk passed before that region is unchanged
        and refused, except a candidate naming the region or one of its ancestors: the conditions and
        subtree assignments it reads can move. Those few are probed again, and if one is now accepted the
        walk restarts from the root, as the matcher's would; otherwise it resumes at the rewritten region.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of fusions performed, or ``None`` if none.
        """
        xform = ConditionFusion()
        fused = 0
        # The walk as a stack of ``[region, child regions or None while its own candidates are probed, next
        # child]`` frames, so it resumes in place instead of listing the whole tree after each fusion. A
        # fusion rewrites only the resumed region's subtree, so every frame above it stays exact.
        stack: List[list] = [[sdfg, None, 0]]
        while stack:
            frame = stack[-1]
            region, children, index = frame
            if children is not None:
                if index == len(children):
                    stack.pop()
                else:
                    frame[2] = index + 1
                    stack.append([children[index], None, 0])
                continue
            match = next((m for m in matcher_candidates(region) if self.accepts(xform, region, *m)), None)
            if match is None:
                frame[1] = child_regions(region)
                continue
            expr_index, binding = match
            self.bind(xform, region, expr_index, binding)
            # The pair fuses inside ``region``; the nested form rewrites the conditional holding it.
            resume = region if expr_index == CONSECUTIVE else region.parent_graph
            xform.apply(region, region.sdfg)
            fused += 1
            if expr_index != CONSECUTIVE:
                stack.pop()
            earlier = any(self.accepts(xform, owner, e, b) for owner, e, b in chain_candidates(resume))
            if earlier or not stack or stack[-1][0] is not resume:
                stack = [[sdfg, None, 0]]
            else:
                stack[-1] = [resume, None, 0]
        return fused or None

    @staticmethod
    def bind(xform: ConditionFusion, region: AbstractControlFlowRegion, expr_index: int, binding: Dict) -> None:
        xform.setup_match(region.sdfg, -1, -1, binding, expr_index, override=True)

    @staticmethod
    def accepts(xform: ConditionFusion, region: AbstractControlFlowRegion, expr_index: int, binding: Dict) -> bool:
        FuseConditions.bind(xform, region, expr_index, binding)
        try:
            return xform.can_be_applied(region, expr_index, region.sdfg, permissive=False)
        except Exception as e:  # noqa: BLE001 -- the matcher's own policy, see ``_try_to_match_transformation``
            if Config.get_bool('optimizer', 'match_exception'):
                raise
            print(f'WARNING: ConditionFusion::can_be_applied triggered a {e.__class__.__name__} exception: {e}')
            return False


__all__ = ['FuseConditions']
