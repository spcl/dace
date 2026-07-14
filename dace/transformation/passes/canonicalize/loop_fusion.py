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
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import _symbolically_equal
from dace.transformation.passes.loop_fission import _linear_blocks, _single_compute_state


@transformation.explicit_cf_compatible
class LoopFusion(ppl.Pass):
    """Fuse consecutive same-range sequential sibling loops into one loop.

    A no-op on any pair whose fusion is not provably value-preserving, and on any
    loop that is independently parallel (left to ``LoopToMap``).
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

    def _fuse_one(self, sdfg: SDFG, cfg: ControlFlowRegion) -> bool:
        """Find and fuse one legal consecutive same-range loop pair inside ``cfg``.

        :param sdfg: The owning SDFG (for the DOALL oracle).
        :param cfg: The control-flow region to search (one level; deeper loops are
                    reached via ``all_control_flow_regions``).
        :returns: ``True`` if a pair was fused.
        """
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            link = out_edges[0]
            second = link.dst
            if not isinstance(second, LoopRegion) or second is first:
                continue
            if len(cfg.in_edges(second)) != 1:
                continue
            # The connecting edge must be pure sequencing: no assignments, trivial
            # condition, so nothing runs (or is decided) between the two loops.
            if link.data.assignments:
                continue
            if link.data.condition is not None and link.data.condition.as_string not in ('1', 'True', '(1)'):
                continue
            if not self._same_iteration_space(first, second):
                continue
            s1 = _single_compute_state(first)
            s2 = _single_compute_state(second)
            if s1 is None or s2 is None:
                continue
            # Never serialize a loop that could parallelize on its own -- leave it
            # to LoopToMap. Post-LoopToMap this is already true of every candidate;
            # the guard keeps the pass correct if it is ever moved earlier.
            if self._is_doall(sdfg, first) or self._is_doall(sdfg, second):
                continue
            if not self._fusion_legal(first, second, s1, s2):
                continue
            self._merge(sdfg, cfg, first, second)
            return True
        return False

    @staticmethod
    def _is_doall(sdfg: SDFG, loop: LoopRegion) -> bool:
        """``True`` iff ``LoopToMap`` would parallelize ``loop`` (the same oracle
        the ``parallelize`` stage uses)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        try:
            return LoopToMap.can_be_applied_to(sdfg, loop=loop)
        except Exception:  # noqa: BLE001 -- oracle refuses exotic shapes -> not provably DOALL
            return False

    @staticmethod
    def _same_iteration_space(first: LoopRegion, second: LoopRegion) -> bool:
        """Both loops iterate the same range with the same stride (iterator names
        may differ; they are unified during the merge)."""
        for loop in (first, second):
            if not loop.loop_variable:
                return False
        s1 = loop_analysis.get_loop_stride(first)
        s2 = loop_analysis.get_loop_stride(second)
        if s1 is None or s2 is None or not _symbolically_equal(s1, s2):
            return False
        i1 = loop_analysis.get_init_assignment(first)
        i2 = loop_analysis.get_init_assignment(second)
        e1 = loop_analysis.get_loop_end(first)
        e2 = loop_analysis.get_loop_end(second)
        if None in (i1, i2, e1, e2):
            return False
        return _symbolically_equal(i1, i2) and _symbolically_equal(e1, e2)

    @staticmethod
    def _accesses(state: SDFGState) -> Tuple[Dict[str, List], Dict[str, List]]:
        """``(reads, writes)`` -- per data container, the list of accessed subsets
        into that container. In-edges of an AccessNode write it (its ``dst`` side),
        out-edges read it (its ``src`` side). A subset that cannot be resolved is
        recorded as ``None`` so the legality check can refuse conservatively rather
        than silently drop a dependence."""
        reads: Dict[str, List] = {}
        writes: Dict[str, List] = {}
        for n in state.nodes():
            if not isinstance(n, nodes.AccessNode):
                continue
            for e in state.in_edges(n):
                sub = e.data.get_dst_subset(e, state) if e.data is not None else None
                writes.setdefault(n.data, []).append(sub)
            for e in state.out_edges(n):
                sub = e.data.get_src_subset(e, state) if e.data is not None else None
                reads.setdefault(n.data, []).append(sub)
        return reads, writes

    def _fusion_legal(self, first: LoopRegion, second: LoopRegion, s1: SDFGState, s2: SDFGState) -> bool:
        """Whether running ``s1`` then ``s2`` per iteration preserves the value of
        the two loops run in sequence. See the module docstring for the rule."""
        ivar = first.loop_variable
        classifier = BreakAntiDependence()
        r1, w1 = self._accesses(s1)
        r2, w2 = self._accesses(s2)
        arrays = (set(r1) | set(w1)) & (set(r2) | set(w2))
        for arr in arrays:
            # A dependence whose subset could not be resolved (None) is refused
            # rather than dropped -- otherwise an unseen edge could hide a
            # fusion-preventing dependence.
            if any(sub is None for d in (r1, w1, r2, w2) for sub in d.get(arr, [])):
                return False
            # flow: a value written in body1 and read in body2 must not be read
            # ahead of its production (read-ahead WAR = fused reads stale value).
            for write in w1.get(arr, []):
                for read in r2.get(arr, []):
                    cls, _ = classifier._dep_class(read, write, ivar)
                    if cls != 'RAW' and cls != 'none':
                        return False
            # anti: a value read in body1 must not have been overwritten earlier in
            # the fused sweep by body2 (read-behind of body2's write).
            for read in r1.get(arr, []):
                for write in w2.get(arr, []):
                    cls, _ = classifier._dep_class(read, write, ivar)
                    if cls != 'WAR' and cls != 'none':
                        return False
            # output: writes from both bodies must hit the same cell each iteration,
            # else the last-writer order differs between fused and unfused.
            for wa in w1.get(arr, []):
                for wb in w2.get(arr, []):
                    if not self._same_point(wa, wb):
                        return False
        return True

    @staticmethod
    def _same_point(a, b) -> bool:
        """Whether two subsets are the same point access on every dimension."""
        ra, rb = list(a.ndrange()), list(b.ndrange())
        if len(ra) != len(rb):
            return False
        for (sa, _ea, _sa), (sb, _eb, _sb) in zip(ra, rb):
            if not _symbolically_equal(sa, sb):
                return False
        return True

    @staticmethod
    def _merge(sdfg: SDFG, cfg: ControlFlowRegion, first: LoopRegion, second: LoopRegion) -> None:
        """Append ``second``'s body to ``first``'s and splice ``second`` out.

        ``first`` keeps its header and iterator; ``second``'s compute state is
        renamed to ``first``'s iterator and appended after ``first``'s last body
        block. ``second``'s successors are re-homed onto ``first``. The adjacent
        body states are merged by the ``StateFusionExtended`` in the next cleanup.
        """
        v1 = first.loop_variable
        v2 = second.loop_variable
        s2 = _single_compute_state(second)
        second.remove_node(s2)  # detach the state object from second (its dataflow survives)
        if v2 != v1:
            s2.replace(v2, v1)  # unify the iterator so s2's memlets index first's variable
        order = _linear_blocks(first) or [_single_compute_state(first)]
        last = order[-1]
        # ensure_unique_name: ``s2`` (from ``second``) may share the frontend's
        # auto-generated block name with one of ``first``'s blocks; without a
        # rename the fused loop carries two identically-named states, which trips
        # a later fission/clone ("multiple blocks with the same name").
        first.add_node(s2, ensure_unique_name=True)
        first.add_edge(last, s2, InterstateEdge())

        out_edges = list(cfg.out_edges(second))
        cfg.remove_node(second)  # also drops the first -> second sequencing edge
        for e in out_edges:
            cfg.add_edge(first, e.dst, copy.deepcopy(e.data))
        set_nested_sdfg_parent_references(sdfg)


__all__ = ['LoopFusion']
