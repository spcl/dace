# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Desugar an N-way ``ConditionalBlock`` into a chain of single-arm blocks.

``SameWriteSetIfElseToITECFG`` + :class:`BranchNormalization` lower one-arm ``if`` and
two-arm ``if/else`` into per-lane ``ITE`` masks, but neither touches a block with ≥3
branches (or a two-arm ``if/elif`` with no ``else``). Such a block survives into the
tiled body as scalar control flow evaluated once at the TILE BASE (``cond[_loop_it_0] > 0``
gates all lanes by lane 0), mis-predicating every other lane. The motivating shape is
condition fusion: canon fuses two independent guards into one block enumerating the
cartesian product of the atomic predicates; this pass flattens it back to straight-line
single-arm ``if``s so the single-arm ``ITE`` lowering fires per-lane.

Standard ``if/elif/else`` → sequential-``if`` rewrite, each arm accumulating the negations
of earlier arms so first-match (elif) semantics hold exactly::

    if c0: b0            ->    if c0: b0
    elif c1: b1               if (not c0) and c1: b1
    elif c2: b2               if (not c0) and (not c1) and c2: b2
    else:  b3                 if (not c0) and (not c1) and (not c2): b3

Each emitted block is single-arm, so :class:`BranchNormalization` lowers it to
``arr = ITE(eff_cond, expr, arr)`` masked writes; a data-dependent ``cond[i]`` in the
accumulated condition becomes a per-lane mask. Two-arm ``if/else`` is LEFT for
``SameWriteSetIfElseToITECFG`` / ``BranchNormalization`` (a tighter ``ITE(c, t, e)``
blend); this pass only fires where those give up.
"""
from typing import List, Optional, Tuple

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


@properties.make_properties
class FlattenBranches(ppl.Pass):
    """Desugar ``>=3``-way (and ``elif``-no-``else``) ``ConditionalBlock``s into
    a chain of single-arm blocks with accumulated-negation conditions."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Flatten every eligible ``ConditionalBlock`` to fixed point.

        :param sdfg: SDFG to transform in place.
        :returns: number of blocks flattened, or ``None`` if none.
        """
        flattened = 0
        progress = True
        while progress:
            progress = False
            for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
                for block in list(cfg.nodes()):
                    if isinstance(block, ConditionalBlock) and self._should_flatten(block):
                        self._flatten(block)
                        flattened += 1
                        progress = True
        return flattened or None

    @staticmethod
    def _should_flatten(cb: ConditionalBlock) -> bool:
        """A block is flattened iff the existing 1-arm / 2-arm if-else passes
        cannot: ``>=3`` branches, or a 2-arm block whose SECOND arm also carries
        a condition (``if/elif`` with no ``else``)."""
        branches = cb.branches
        if len(branches) >= 3:
            return True
        if len(branches) == 2:
            return branches[0][0] is not None and branches[1][0] is not None
        return False

    @staticmethod
    def _cond_text(cond) -> str:
        return cond.as_string if isinstance(cond, CodeBlock) else str(cond)

    def _flatten(self, cb: ConditionalBlock) -> None:
        """Replace ``cb`` with a sequential chain of single-arm blocks."""
        parent = cb.parent_graph
        # Snapshot the arms in order, then detach every body from ``cb`` so each
        # can be re-parented onto its own single-arm block.
        arms: List[Tuple[Optional[CodeBlock], ControlFlowRegion]] = list(cb.branches)
        for _cond, body in arms:
            cb.remove_branch(body)

        in_edges = list(parent.in_edges(cb))
        out_edges = list(parent.out_edges(cb))

        prior_neg: List[str] = []  # negations of earlier arms' conditions
        new_blocks: List[ConditionalBlock] = []
        for i, (cond, body) in enumerate(arms):
            terms = list(prior_neg)
            if cond is not None:
                terms.append(f"({self._cond_text(cond)})")
            # Empty term list = unconditional arm (a leading bare ``else`` as the sole
            # branch); guard as ``True`` so the block stays a well-formed single-arm cond.
            eff = " and ".join(terms) if terms else "True"
            blk = ConditionalBlock(label=f"{cb.label}_flat{i}", sdfg=parent.sdfg, parent=parent)
            blk.add_branch(CodeBlock(eff), body)
            parent.add_node(blk)
            new_blocks.append(blk)
            if cond is not None:
                prior_neg.append(f"(not ({self._cond_text(cond)}))")

        # Stitch: in-edges -> first block, chain the blocks, last -> cb's out.
        for ie in in_edges:
            parent.add_edge(ie.src, new_blocks[0], ie.data)
        for a, b in zip(new_blocks, new_blocks[1:]):
            parent.add_edge(a, b, dace.InterstateEdge())
        for oe in out_edges:
            parent.add_edge(new_blocks[-1], oe.dst, oe.data)

        parent.remove_node(cb)  # drops cb and its now-dangling in/out edges
        parent.reset_cfg_list()
