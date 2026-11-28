# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reorder a state sitting between two loops PAST the second loop, so the loops become adjacent."""

from typing import Any

from dace import SDFG
from dace import data as dt
from dace.sdfg import nodes as nd
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
    is_written,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.analysis import names_read_by_text

#: A ``(loop1, state, loop2)`` triple matching the ``loop1 -> state -> loop2`` chain.
Candidate = tuple[LoopRegion, SDFGState, LoopRegion]

#: Blocks through which control can leave ``loop2`` other than by its normal successor (item 7).
ESCAPE_BLOCKS: tuple[type[ControlFlowBlock], ...] = (ReturnBlock, BreakBlock, ContinueBlock)

#: Descriptor types for which two different names can name the same memory (item 4).
ALIASING_TYPES: tuple[type, ...] = (dt.View, dt.Reference)


@transformation.explicit_cf_compatible
class ReorderStateForLoopFusion(ppl.Pass):
    """Move a state between two loops past the second loop when that is legal and unlocks ``LoopFusion``."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Reorder every legal, fusion-unlocking between-loops state in ``sdfg``."""
        reordered = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not isinstance(cfg, ControlFlowRegion):
                    continue
                # A move rewires edges of ``cfg`` only; a refused triple stays refused.
                refused: set[Candidate] = set()
                while self.reorder_one(sd, cfg, refused):
                    reordered += 1
        return reordered or None

    @staticmethod
    def candidates(cfg: ControlFlowRegion) -> list[Candidate]:
        """Every ``loop1 -> state -> loop2`` chain directly inside ``cfg`` whose edges allow the move (items 8,
        9, 10, 12)."""
        found: list[Candidate] = []
        start = cfg.start_block
        for state in cfg.nodes():
            if not isinstance(state, SDFGState) or state is start:  # item 12
                continue
            in_edges, out_edges = cfg.in_edges(state), cfg.out_edges(state)
            if len(in_edges) != 1 or len(out_edges) != 1:  # item 10
                continue
            first, second = in_edges[0].src, out_edges[0].dst
            if not isinstance(first, LoopRegion) or not isinstance(second, LoopRegion) or first is second:
                continue
            if second is start:  # item 12
                continue
            if len(cfg.in_edges(second)) != 1:  # item 10
                continue
            if in_edges[0].data.assignments or out_edges[0].data.assignments:  # item 8
                continue
            if not in_edges[0].data.is_unconditional() or not out_edges[0].data.is_unconditional():  # item 9
                continue
            found.append((first, state, second))
        return found

    @staticmethod
    def escapes(second: LoopRegion) -> bool:
        """Item 7: whether ``second`` holds a return, break or continue block at any depth."""
        return any(isinstance(b, ESCAPE_BLOCKS) for b in second.all_control_flow_blocks(recursive=True))

    @staticmethod
    def opaque_code(block: SDFGState | LoopRegion, sdfg: SDFG) -> bool:
        """Items 5 and 11: whether ``block`` holds a nested SDFG or a code node with side effects."""
        states = [block] if isinstance(block, SDFGState) else block.states()
        return any(
            isinstance(n, nd.NestedSDFG) or (isinstance(n, nd.CodeNode) and n.has_side_effects(sdfg))
            for s in states
            for n in s.nodes()
        )

    @staticmethod
    def access_sets(block: SDFGState | LoopRegion, arrays: Any) -> tuple[set[str], set[str]]:
        """Containers ``block`` reads and writes: access nodes, plus data read by its loop headers, branch
        conditions and inner interstate edges."""
        reads: set[str] = set()
        writes: set[str] = set()
        states = [block] if isinstance(block, SDFGState) else block.states()
        for s in states:
            for an in s.data_nodes():
                if is_written(s, an):
                    writes.add(an.data)
                if s.out_degree(an):
                    reads.add(an.data)
        if isinstance(block, SDFGState):
            return reads, writes
        texts: list[str] = []
        for b in [block, *block.all_control_flow_blocks()]:
            if isinstance(b, LoopRegion):
                texts.append(b.loop_condition.as_string)
                texts.extend(
                    t
                    for t in (
                        loop_analysis.assignment_text(b.init_statement, b.loop_variable),
                        loop_analysis.assignment_text(b.update_statement, b.loop_variable),
                    )
                    if t
                )
            elif isinstance(b, ConditionalBlock):
                texts.extend(c.as_string for c, _ in b.branches if c is not None)
        for t in texts:
            reads |= names_read_by_text(t) & arrays
        for e in block.all_interstate_edges(recursive=True):
            reads |= e.data.free_symbols & arrays
        return reads, writes

    @staticmethod
    def reorder_legal(state: SDFGState, second: LoopRegion) -> bool:
        """Whether running ``state`` after ``second`` changes nothing (items 7, 5, 11, 4, 1-3, cheapest first)."""
        sdfg = state.sdfg
        if ReorderStateForLoopFusion.escapes(second):
            return False
        if ReorderStateForLoopFusion.opaque_code(state, sdfg) or ReorderStateForLoopFusion.opaque_code(second, sdfg):
            return False
        arrays = sdfg.arrays.keys()
        reads_s, writes_s = ReorderStateForLoopFusion.access_sets(state, arrays)
        reads_b, writes_b = ReorderStateForLoopFusion.access_sets(second, arrays)
        if any(isinstance(sdfg.arrays[n], ALIASING_TYPES) for n in reads_s | writes_s | reads_b | writes_b):
            return False
        return not (writes_s & reads_b or reads_s & writes_b or writes_s & writes_b)

    @staticmethod
    def reorder_past(
        cfg: ControlFlowRegion, first: LoopRegion, state: SDFGState, second: LoopRegion
    ) -> tuple[list, list]:
        """Rewire ``first -> state -> second`` into ``first -> second -> state`` in place."""
        removed = [cfg.edges_between(first, state)[0], cfg.edges_between(state, second)[0], *cfg.out_edges(second)]
        for e in removed:
            cfg.remove_edge(e)
        added = [cfg.add_edge(first, second, InterstateEdge()), cfg.add_edge(second, state, InterstateEdge())]
        added += [cfg.add_edge(state, e.dst, e.data) for e in removed[2:]]
        return removed, added

    @staticmethod
    def undo_reorder(cfg: ControlFlowRegion, removed: list, added: list) -> None:
        """Undo :meth:`reorder_past`."""
        for e in added:
            cfg.remove_edge(e)
        for e in removed:
            cfg.add_edge(e.src, e.dst, e.data)

    def reorder_one(self, sdfg: SDFG, cfg: ControlFlowRegion, refused: set[Candidate]) -> bool:
        """Move one legal, fusion-unlocking candidate inside ``cfg``; record the refused ones in ``refused``."""
        for cand in self.candidates(cfg):
            if cand in refused:
                continue
            first, state, second = cand
            if not LoopFusion._same_iteration_space(first, second) or not self.reorder_legal(state, second):
                refused.add(cand)
                continue
            removed, added = self.reorder_past(cfg, first, state, second)
            if LoopFusion.can_be_applied_to(sdfg, first=first, second=second):
                return True
            self.undo_reorder(cfg, removed, added)
            refused.add(cand)
        return False


__all__ = ["ReorderStateForLoopFusion"]
