# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reorder a state sitting between two loops PAST the second loop, so the loops become adjacent.

``FuseLoops`` (and the ``LoopFusion`` pass built on it) matches a two-node path graph, so
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

Even a legal reorder is pointless on its own: the point is only to unlock ``FuseLoops``. So the
reorder fires only when it is BOTH dependence-legal AND ``FuseLoops.can_be_applied_to`` would accept
the resulting ``loop1 -> loop2`` adjacency -- decided on a throwaway deep copy so a refusal never
mutates the real SDFG. The pass itself never fuses; it only restores adjacency and leaves fusing to
``LoopFusion``/``FuseLoops``, the same division of labour ``SinkStateIntoLoop`` uses.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG
from dace import data as dt
from dace.sdfg import nodes as nd
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import BreakBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.fuse_loops import FuseLoops
from dace.transformation.passes.analysis.analysis import AccessSets

#: A ``(loop1, state, loop2)`` triple matching the ``loop1 -> state -> loop2`` chain, structurally
#: eligible to be considered for the reorder (legality and fusability still to be decided).
Candidate = Tuple[LoopRegion, SDFGState, LoopRegion]

#: Block types whose presence means control can leave ``loop2`` somewhere other than its normal
#: successor edge -- see checklist item 7.
ESCAPE_BLOCKS = (ReturnBlock, BreakBlock, ContinueBlock)

#: Descriptor types for which two different names can name the same memory -- see checklist item 4.
ALIASING_TYPES = (dt.View, dt.Reference)


@transformation.explicit_cf_compatible
class ReorderStateForLoopFusion(ppl.Pass):
    """Reorder a between-loops SIBLING state past the second loop when doing so is dependence-legal
    AND unlocks ``FuseLoops``. ``state`` is never nested into ``loop2``; it stays a sibling block
    before and after, just moved from ``loop1 -> state -> loop2`` to ``loop1 -> loop2 -> state``.
    Fires on neither condition alone, and refuses outright on anything undecidable -- see the module
    docstring for the full checklist."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def depends_on(self) -> List[type]:
        return [AccessSets]

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Reorder every legal, fusion-unlocking between-loops state in ``sdfg``, to a fixpoint.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of states sunk, or ``None`` if none.
        """
        access_sets = pipeline_results.get(AccessSets.__name__)
        if access_sets is None:
            access_sets = AccessSets().apply_pass(sdfg, {})

        reordered = 0
        changed = True
        while changed:
            changed = False
            for sd in sdfg.all_sdfgs_recursive():
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    # `sd`, not the top SDFG: the throwaway copy in `would_fuse` only has to carry the
                    # SDFG the candidate actually lives in, and FuseLoops is handed that same scope.
                    if self.reorder_one(sd, cfg, access_sets):
                        reordered += 1
                        changed = True
                        break
                if changed:
                    break
            if changed:
                # The move can change which pairs of blocks are adjacent to which edges, so a stale
                # access-set entry could under- or over-count what an endpoint now touches -- recompute.
                access_sets = AccessSets().apply_pass(sdfg, {})
        return reordered or None

    @staticmethod
    def candidates(cfg: ControlFlowRegion) -> List[Candidate]:
        """Every structurally-eligible ``loop1 -> state -> loop2`` chain directly inside ``cfg``
        (one level; nested chains are reached separately via ``all_control_flow_regions``).

        Enforces checklist items 8/9/10/12: both edges around ``state`` unconditional and
        assignment-free, ``state`` the sole path in and out, ``loop2`` with no other predecessor, and
        neither of the two reordered blocks is the entry.
        """
        found: List[Candidate] = []
        start = cfg.start_block
        for state in cfg.nodes():
            if not isinstance(state, SDFGState) or state is start:  # item 12
                continue
            in_edges, out_edges = cfg.in_edges(state), cfg.out_edges(state)
            if len(in_edges) != 1 or len(out_edges) != 1:  # item 10: state is the only path in/out
                continue
            first, second = in_edges[0].src, out_edges[0].dst
            if not isinstance(first, LoopRegion) or not isinstance(second, LoopRegion) or first is second:
                continue
            if second is start:  # item 12: second inherits state's out-edges, so it must not be the entry
                continue
            # item 10: second must have no OTHER way in (also catches a parallel loop1 -> loop2 edge).
            if len(cfg.in_edges(second)) != 1:
                continue
            if in_edges[0].data.assignments or out_edges[0].data.assignments:  # item 8
                continue
            if not in_edges[0].data.is_unconditional() or not out_edges[0].data.is_unconditional():  # item 9
                continue
            found.append((first, state, second))
        return found

    @staticmethod
    def escapes(second: LoopRegion) -> bool:
        """Item 7: whether ``second``'s body contains a ``ReturnBlock``/``BreakBlock``/``ContinueBlock``
        anywhere (any nesting depth; ``all_control_flow_blocks`` also follows a ``NestedSDFG`` down into
        its own regions). Presence alone refuses -- reachability is not attempted."""
        return any(isinstance(b, ESCAPE_BLOCKS) for b in second.all_control_flow_blocks(recursive=True))

    @staticmethod
    def nested_sdfgs_within(block: Any) -> bool:
        """Item 5: whether ``block`` (a plain state, or a region such as ``second``) contains a
        ``NestedSDFG`` node anywhere in its own states. Proving RAW/WAR/WAW disjointness, WCR/dynamic-
        memlet safety, and aliasing THROUGH an SDFG-scope boundary needs data-flow this pass does not
        have -- any ``NestedSDFG`` here refuses the whole candidate rather than guessing.
        """
        states = [block] if isinstance(block, SDFGState) else list(block.all_states())
        return any(isinstance(n, nd.NestedSDFG) for s in states for n in s.nodes())

    @staticmethod
    def has_side_effecting_code(block: Any, sdfg: SDFG) -> bool:
        """Item 11: whether ``block`` (a plain state, or a region such as ``second``) holds a code node
        whose effects are not described by its memlets. Such a node is invisible to the name-level
        RAW/WAR/WAW checks, so it has to disqualify the candidate from whichever side it sits on."""
        states = [block] if isinstance(block, SDFGState) else list(block.all_states())
        return any(isinstance(n, nd.CodeNode) and n.has_side_effects(sdfg) for s in states for n in s.nodes())

    @staticmethod
    def touches_aliasing_data(sdfg: SDFG, names: Any) -> bool:
        """Item 4: whether any of ``names`` is a ``View``/``Reference`` descriptor in ``sdfg``. A name
        not in ``sdfg.arrays`` (e.g. a free symbol AccessSets folded in from an edge) is not a data
        descriptor and cannot alias anything, so it is skipped rather than raising."""
        return any(isinstance(sdfg.arrays[name], ALIASING_TYPES) for name in names if name in sdfg.arrays)

    @staticmethod
    def reorder_legal(state: SDFGState, second: LoopRegion, access_sets: Dict[Any, Tuple[Any, Any]]) -> bool:
        """Whether running ``state`` after ``second`` instead of before it changes nothing. Checks items
        7, 11, 5 (+6), 4, then 1/2/3, cheapest first.

        RAW/WAR/WAW are container-name-level, never subset-level: if two accesses name different data
        descriptors they provably cannot conflict regardless of subset, so this is sound; a shared name
        (WCR or not, dynamic or not) is always treated as a conflict even if the actual subsets might not
        overlap -- an undecided overlap is not legal.
        """
        if ReorderStateForLoopFusion.escapes(second):  # item 7
            return False
        sdfg = state.sdfg
        if ReorderStateForLoopFusion.has_side_effecting_code(
                state, sdfg) or ReorderStateForLoopFusion.has_side_effecting_code(second, sdfg):
            return False  # item 11
        if ReorderStateForLoopFusion.nested_sdfgs_within(state) or ReorderStateForLoopFusion.nested_sdfgs_within(
                second):  # item 5 (+6)
            return False
        reads_s, writes_s = access_sets[state]
        reads_b, writes_b = access_sets[second]
        if ReorderStateForLoopFusion.touches_aliasing_data(sdfg, reads_s | writes_s | reads_b | writes_b):  # item 4
            return False
        if writes_s & reads_b:  # item 1, RAW: state writes what second reads
            return False
        if reads_s & writes_b:  # item 2, WAR: state reads what second writes
            return False
        if writes_s & writes_b:  # item 3, WAW: both write the same container
            return False
        return True

    @staticmethod
    def would_fuse(sdfg: SDFG, cfg: ControlFlowRegion, first: LoopRegion, state: SDFGState, second: LoopRegion) -> bool:
        """Whether ``FuseLoops`` would accept ``first``/``second`` once ``state`` no longer sits between
        them. Decided on a throwaway deep copy of the whole SDFG (memoized by object id, so the copies of
        ``first``/``state``/``second`` are recoverable) -- a refusal here must never touch the real graph.
        """
        memo: Dict[int, Any] = {}
        sdfg_copy = copy.deepcopy(sdfg, memo)
        ReorderStateForLoopFusion.reorder_past(memo[id(cfg)], memo[id(first)], memo[id(state)], memo[id(second)])
        return FuseLoops.can_be_applied_to(sdfg_copy, first=memo[id(first)], second=memo[id(second)])

    @staticmethod
    def reorder_past(cfg: ControlFlowRegion, first: LoopRegion, state: SDFGState, second: LoopRegion) -> None:
        """Rewire ``first -> state -> second`` into ``first -> second -> state``, in place.

        ``state`` stays a SIBLING node of ``cfg`` throughout -- it is never added into ``second``'s own
        node set, only re-targeted by edges, so its nesting level is unchanged by this reorder. It
        inherits ``second``'s original outgoing edges (deep-copied, never the same edge-data object
        shared across two edges); the two new connecting edges are fresh, plain, unconditional splices --
        exactly the shape ``candidates`` already required of the edges being removed.
        """
        tail_edges = list(cfg.out_edges(second))
        cfg.remove_edge(cfg.edges_between(first, state)[0])
        cfg.remove_edge(cfg.edges_between(state, second)[0])
        for e in tail_edges:
            cfg.remove_edge(e)
        cfg.add_edge(first, second, InterstateEdge())
        cfg.add_edge(second, state, InterstateEdge())
        for e in tail_edges:
            cfg.add_edge(state, e.dst, copy.deepcopy(e.data))

    def reorder_one(self, sdfg: SDFG, cfg: ControlFlowRegion, access_sets: Dict[Any, Tuple[Any, Any]]) -> bool:
        """Find and sink one legal, fusion-unlocking candidate inside ``cfg``."""
        for first, state, second in self.candidates(cfg):
            if not self.reorder_legal(state, second, access_sets):
                continue
            if not self.would_fuse(sdfg, cfg, first, state, second):
                continue
            self.reorder_past(cfg, first, state, second)
            return True
        return False


__all__ = ['ReorderStateForLoopFusion']
