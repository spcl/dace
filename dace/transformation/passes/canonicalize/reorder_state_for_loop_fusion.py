# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reorder a state sitting between two loops PAST the second loop, so the loops become adjacent.

``LoopFusion`` (and the ``FuseLoops`` pass built on it) matches a two-node path graph, so
``loop1 -> state -> loop2`` produces no candidate pair at all and the loops are never offered for
fusion. ``state`` stays a SIBLING block at the same nesting level the whole time -- it is never
nested into ``loop2``'s body, only reordered relative to it: ``loop1 -> loop2 -> state``. This is
sound only when nothing about the reorder is left undecided; a permissive check here is a silent
miscompile, not a crash, so every condition below is either PROVEN or the candidate is refused.

Numbered against the checklist this pass must clear (see ``reorder_legal``/``escapes`` for where each
one is enforced):

  1. RAW  -- ``state`` writes data ``loop2`` reads.
  2. WAR  -- ``state`` reads data ``loop2`` writes.
  3. WAW  -- ``state`` writes data ``loop2`` also writes.
  4. Aliasing -- a ``View``/``Reference`` makes name-disjointness meaningless (two names, one
     memory); resolving an alias soundly in general needs data-flow this pass does not have, so ANY
     touch of a ``View``/``Reference``-typed descriptor by either side refuses the whole candidate.
  5. Nested SDFGs -- an access inside a ``NestedSDFG`` can escape its connectors' declared footprint
     in ways this pass cannot cheaply re-verify; ANY ``NestedSDFG`` in ``state`` or in ``loop2``'s
     body refuses the whole candidate (this also conservatively subsumes 4 and 6 for anything hidden
     behind that boundary).
  6. WCR / dynamic memlets / streams -- never given subset-level credit for being "probably fine";
     the RAW/WAR/WAW checks are container-name-only (no subset reasoning at all), so a WCR or dynamic
     write is exactly as disqualifying as a plain one, and a stream push/pop is just another access
     node with degree -- no special case needed, no special case trusted.
  7. Early exit -- a ``ReturnBlock``/``BreakBlock``/``ContinueBlock`` anywhere in ``loop2``'s body
     (at any nesting depth) can land control somewhere other than ``loop2``'s normal successor; were
     ``state`` moved to run only after that successor, such a path would never run ``state`` at all.
     Refused outright, unconditionally on reachability (this pass does not attempt to prove a given
     escape block is dead).
  8. Symbols -- an assignment on the ``state -> loop2`` edge could define a symbol ``loop2``'s
     bounds/condition/subsets consume; refused by requiring that edge (and the ``loop1 -> state``
     edge) to carry no assignments at all, full stop.
  9. Conditions -- a non-trivial condition on either edge around ``state`` is itself a control
     dependence this pass does not attempt to re-home; both edges must be unconditional.
  10. Topology -- ``state`` must be the ONLY path from ``loop1`` to ``loop2``: exactly one in-edge
      (from ``loop1``) and one out-edge (to ``loop2``) on ``state`` itself, and ``loop2`` must have no
      OTHER predecessor (which also catches a direct ``loop1 -> loop2`` edge running in parallel).
  11. Side effects -- checked with ``node.has_side_effects(sdfg)`` (a method, not the ``Tasklet``
      ``side_effects`` Property, which defaults to ``None`` and answers nothing). Scanned on BOTH
      sides: a side-effecting node is one whose effects are NOT described by its memlets, so the
      name-level RAW/WAR/WAW checks say nothing about it -- a side-effecting node in ``loop2`` could
      read data ``state`` writes without that read appearing in any access set. ``NestedSDFG``'s own
      override already recurses into arbitrarily deep nesting, so this scan is complete on its own
      merits -- independent of the blanket nested-SDFG refusal in 5.
  12. Entry block -- the reorder rewires the edges around ``state`` and ``loop2``, so if either is
      ``cfg``'s start block the entry path itself changes shape (execution would begin partway
      through the new chain). Refused; only a candidate whose blocks are all interior is considered.

Checks run cheapest first: the edge shape (``candidates``), then the two loops' iteration spaces, then the
bodies, then the name-level dependences. Only a reorder that is legal AND that ``LoopFusion.can_be_applied_to``
accepts is kept: it is tried in place and undone on a refusal. The pass itself never fuses; ``FuseLoops`` does.
"""

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
        """Reorder every legal, fusion-unlocking between-loops state in ``sdfg``.

        :returns: Number of states moved, or ``None`` if none.
        """
        reordered = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions(recursive=True)):
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
                if s.in_degree(an):
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
        """Whether running ``state`` after ``second`` changes nothing (items 7, 5, 11, 4, 1-3, cheapest first).

        Dependences are by container name: a shared name is a conflict whatever the subsets.
        """
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
        """Rewire ``first -> state -> second`` into ``first -> second -> state`` in place.

        :returns: The removed and the added edges, for :meth:`undo_reorder`.
        """
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
