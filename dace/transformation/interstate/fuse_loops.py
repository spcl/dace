# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseLoops``: the single-pair transformation form of loop fusion -- the LoopRegion analogue of
``MapFusionVertical``, for the residual sequential loops left after ``LoopToMap``.

``MapFusionVertical`` / ``MapFusionHorizontal`` only ever fuse ``MapEntry`` nodes, so two consecutive
sibling loops that ``LoopToMap`` refused (recurrences, Thomas sweeps, sequential scans) are never fused.
This transformation fuses ONE such pair -- same iteration space, single-compute-state bodies -- into one
loop whose body runs the first body then the second per iteration. The ``LoopFusion`` PASS is exactly this
transformation applied to every legal pair until a fixpoint; the transformation is the source of truth for
the legality kernel, so the pass and the transformation can never disagree.

It is the exact inverse of ``LoopFission`` and shares its soundness kernel (``break_anti_dependence``'s
``_dep_class`` and ``loop_fission``'s per-iteration-subset reasoning), so a pair that fission would have
separated is a pair that fusion may legally rejoin.

Designed to run POST-``LoopToMap`` (every DOALL loop is already a Map, hence not a candidate), so it only
ever sees sequential residual loops and cannot serialize parallel work. A defensive
``LoopToMap.can_be_applied_to`` guard refuses to fuse a loop that is independently DOALL, keeping it
correct even out of position.

Legality (per array touched by both bodies, under the shared iterator):
  * flow (write in body1, read in body2): illegal if the read is READ-AHEAD (``a[i+k]``, k>0) -- the fused
    loop reads a not-yet-produced value.
  * anti (read in body1, write in body2): illegal if the read is READ-BEHIND (``a[i-k]``, k>0) of what
    body2 overwrites earlier in the fused sweep.
  * output (write in both): illegal unless the two writes hit the same index.
Symbolic / indirected / complex offsets are refused conservatively (v1).

After a legal merge it also CONTRACTS intermediates: a transient the fused body produces and consumes at
the same point ``tmp[i]`` (no cross-iteration history, no outside use) is shrunk to a reused ``[1]`` slot --
the buffer-reclaim ``MapFusionVertical`` does for maps, decided here on the loop iteration. See
``_contract_localized_intermediates``.
"""
import copy
from typing import Dict, List, Optional, Tuple

from dace import SDFG
from dace import data as dt
from dace import subsets
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import _symbolically_equal
from dace.transformation.passes.loop_fission import _linear_blocks, _single_compute_state


@transformation.explicit_cf_compatible
class FuseLoops(transformation.MultiStateTransformation):
    """Fuse one consecutive same-range sequential sibling loop pair into a single loop.

    Applicable only where the fusion is provably value-preserving, and never on a loop that is
    independently parallel (left to ``LoopToMap``). ``first`` is the surviving loop; ``second``'s body is
    appended to it per iteration and ``second`` is spliced out.
    """

    first = transformation.PatternNode(LoopRegion)
    second = transformation.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first, cls.second)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        first, second = self.first, self.second
        if second is first:
            return False
        # Both loops must be sole neighbours across the connecting edge: nothing else runs before/after
        # either within this adjacency, so appending body2 to body1 does not reorder a third block.
        if len(graph.out_edges(first)) != 1 or len(graph.in_edges(second)) != 1:
            return False
        link = graph.edges_between(first, second)
        if len(link) != 1:
            return False
        link = link[0]
        # The connecting edge must be pure sequencing: no assignments, trivial condition, so nothing runs
        # (or is decided) between the two loops.
        if link.data.assignments:
            return False
        if link.data.condition is not None and link.data.condition.as_string not in ('1', 'True', '(1)'):
            return False
        if not self._same_iteration_space(first, second):
            return False
        s1 = _single_compute_state(first)
        s2 = _single_compute_state(second)
        if s1 is None or s2 is None:
            return False
        # Never serialize a loop that could parallelize on its own -- leave it to LoopToMap. Post-LoopToMap
        # this is already true of every candidate; the guard keeps it correct if it runs earlier.
        if self._is_doall(sdfg, first) or self._is_doall(sdfg, second):
            return False
        return self._fusion_legal(first, second, s1, s2)

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        self._merge(sdfg, graph, self.first, self.second)
        self._contract_localized_intermediates(sdfg, self.first)

    # ----- legality kernel (shared source of truth; the LoopFusion pass calls this transformation) -----

    @staticmethod
    def _is_doall(sdfg: SDFG, loop: LoopRegion) -> bool:
        """``True`` iff ``LoopToMap`` would parallelize ``loop`` (the same oracle the ``parallelize`` stage
        uses).

        The oracle is asked about the loop's OWN SDFG, not the caller's ``sdfg``: the two differ for a loop
        inside a NestedSDFG, and asking about the wrong graph makes ``LoopToMap`` raise, which the guard
        below would read as "not DOALL" -- silently disabling itself exactly where it still applies. They
        are the same object for a top-level loop, so this costs nothing there.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap
        try:
            return LoopToMap.can_be_applied_to(loop.sdfg, loop=loop)
        except Exception:  # noqa: BLE001 -- oracle refuses exotic shapes -> not provably DOALL
            return False

    @staticmethod
    def _same_iteration_space(first: LoopRegion, second: LoopRegion) -> bool:
        """Both loops iterate the same range with the same stride (iterator names may differ; they are
        unified during the merge)."""
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
        """``(reads, writes)`` -- per data container, the list of accessed subsets into that container.
        In-edges of an AccessNode write it (its ``dst`` side), out-edges read it (its ``src`` side). A
        subset that cannot be resolved is recorded as ``None`` so the legality check can refuse
        conservatively rather than silently drop a dependence."""
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
        """Whether running ``s1`` then ``s2`` per iteration preserves the value of the two loops run in
        sequence. See the module docstring for the rule."""
        ivar = first.loop_variable
        classifier = BreakAntiDependence()
        r1, w1 = self._accesses(s1)
        r2, w2 = self._accesses(s2)
        arrays = (set(r1) | set(w1)) & (set(r2) | set(w2))
        for arr in arrays:
            # A dependence whose subset could not be resolved (None) is refused rather than dropped --
            # otherwise an unseen edge could hide a fusion-preventing dependence.
            if any(sub is None for d in (r1, w1, r2, w2) for sub in d.get(arr, [])):
                return False
            # flow: a value written in body1 and read in body2 must not be read ahead of its production
            # (read-ahead WAR = fused reads stale value). 'invariant' is refused: body1 writes a
            # loop-invariant location that body2 reads, so unfused body2 sees the FINAL value body1 left
            # there while fused it sees the RUNNING one (`for i: s = a[i]` then `for i: d[i] = d[i-1] + s`).
            for write in w1.get(arr, []):
                for read in r2.get(arr, []):
                    cls, _ = classifier._dep_class(read, write, ivar)
                    if cls not in ('RAW', 'none'):
                        return False
            # anti: a value read in body1 must not have been overwritten earlier in the fused sweep by
            # body2 (read-behind of body2's write). 'invariant' is refused for the mirror reason: fused,
            # body1's read at iteration i sees body2's write from iteration i-1 instead of the value the
            # location held before the second loop ran at all.
            for read in r1.get(arr, []):
                for write in w2.get(arr, []):
                    cls, _ = classifier._dep_class(read, write, ivar)
                    if cls not in ('WAR', 'none'):
                        return False
            # output: writes from both bodies must hit the same cell each iteration, else the last-writer
            # order differs between fused and unfused.
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
    def _merge(sdfg: SDFG, cfg: ControlFlowRegion, first: LoopRegion, second: LoopRegion):
        """Append ``second``'s body to ``first``'s and splice ``second`` out.

        ``first`` keeps its header and iterator; ``second``'s compute state is renamed to ``first``'s
        iterator and appended after ``first``'s last body block. ``second``'s successors are re-homed onto
        ``first``. The adjacent body states are merged by the ``StateFusionExtended`` in the next cleanup.
        """
        v1 = first.loop_variable
        v2 = second.loop_variable
        s2 = _single_compute_state(second)
        second.remove_node(s2)  # detach the state object from second (its dataflow survives)
        if v2 != v1:
            s2.replace(v2, v1)  # unify the iterator so s2's memlets index first's variable
        order = _linear_blocks(first) or [_single_compute_state(first)]
        last = order[-1]
        # ensure_unique_name: ``s2`` (from ``second``) may share the frontend's auto-generated block name
        # with one of ``first``'s blocks; without a rename the fused loop carries two identically-named
        # states, which trips a later fission/clone ("multiple blocks with the same name").
        first.add_node(s2, ensure_unique_name=True)
        first.add_edge(last, s2, InterstateEdge())

        out_edges = list(cfg.out_edges(second))
        cfg.remove_node(second)  # also drops the first -> second sequencing edge
        for e in out_edges:
            cfg.add_edge(first, e.dst, copy.deepcopy(e.data))
        set_nested_sdfg_parent_references(sdfg)

    # ----- intermediate contraction (the buffer-shrink MapFusion does, for the loop path) -------------

    def _contract_localized_intermediates(self, sdfg: SDFG, loop: LoopRegion) -> None:
        """Shrink every transient the fused loop produces-and-consumes within a single iteration to a
        per-iteration scalar -- the loop analogue of ``MapFusionVertical`` building a smaller intermediate.

        Fusing ``body1;body2`` per iteration does not, by itself, reclaim the intermediate array between
        them: ``tmp[i]`` written in body1 and read in body2 is still a full ``[N]`` buffer. But once both
        bodies share the iteration, a transient touched ONLY at the SAME point ``tmp[i]`` -- no ``tmp[i-1]``
        history, no use outside the loop -- lives entirely inside iteration ``i``, so a single reused
        ``[1]`` slot preserves every value. This is exactly ``MapFusion``'s exclusive-intermediate case,
        decided on the loop iteration instead of the map iteration; it is orthogonal to parallelism, so it
        fires on the sequential recurrences ``LoopToMap`` left behind.

        Conservative (v1): 1-D transient, all accesses a single identical point of the loop iterator, not
        under a map scope, and touched nowhere outside the fused loop. Anything else is left untouched --
        the fusion already happened; contraction is opportunistic cleanup that never changes a value.
        """
        body_states = _linear_blocks(loop) or [_single_compute_state(loop)]
        body_states = [s for s in body_states if s is not None]
        if not body_states:
            return
        body_set = set(body_states)
        ivar = loop.loop_variable
        for arr in list(sdfg.arrays.keys()):
            desc = sdfg.arrays[arr]
            # Only a genuine internal 1-D array is a contraction target: a Scalar is already minimal, a
            # non-transient is a program in/out, a multi-dim carries per-map footprint (v1 skips it).
            if not desc.transient or not isinstance(desc, dt.Array) or len(desc.shape) != 1:
                continue
            # Exclusive: any access outside the fused loop's body means the buffer outlives one iteration.
            if any(
                    isinstance(n, nodes.AccessNode) and n.data == arr for s in sdfg.all_states() if s not in body_set
                    for n in s.nodes()):
                continue
            point = self._localized_point(arr, body_states, ivar)
            if point is not None:
                self._rewrite_array_to_scalar(sdfg, body_states, arr, desc)

    @staticmethod
    def _localized_point(arr: str, body_states: List[SDFGState], ivar: str) -> Optional[object]:
        """Return the single point subset all accesses to ``arr`` share (referencing ``ivar``), or ``None``
        if the accesses are not one identical iterator-indexed point -- an offset (``arr[i-1]``), a second
        distinct index, an unresolved subset, an access under a map scope, or no iterator all disqualify."""
        ref = None
        saw_write = saw_read = False
        for s in body_states:
            for n in s.nodes():
                if not isinstance(n, nodes.AccessNode) or n.data != arr:
                    continue
                if s.entry_node(n) is not None:  # under a map scope -> many cells per outer iter, not a point
                    return None
                for e in s.in_edges(n):
                    sub = e.data.get_dst_subset(e, s) if e.data is not None else None
                    ref = FuseLoops._unify_point(ref, sub, ivar)
                    if ref is None:
                        return None
                    saw_write = True
                for e in s.out_edges(n):
                    sub = e.data.get_src_subset(e, s) if e.data is not None else None
                    ref = FuseLoops._unify_point(ref, sub, ivar)
                    if ref is None:
                        return None
                    saw_read = True
        # A contractible intermediate is both produced and consumed inside the loop; write-only is dead
        # (leave to ArrayElimination), read-only is a misclassified input.
        if not (saw_write and saw_read) or ref is None:
            return None
        return ref

    @staticmethod
    def _unify_point(ref, sub, ivar: str):
        """Fold one subset into the running ``ref``: it must be a point (start==end per dim), equal to
        ``ref`` if one exists, and reference ``ivar``. Returns the point, or ``None`` on any mismatch."""
        if sub is None:
            return None
        nd = list(sub.ndrange())
        for lo, hi, _ in nd:
            if not _symbolically_equal(lo, hi):  # not a single cell
                return None
        if not any(str(sym) == ivar for lo, _, _ in nd for sym in symbolic.pystr_to_symbolic(str(lo)).free_symbols):
            return None
        if ref is not None:
            rnd = list(ref.ndrange())
            if len(rnd) != len(nd) or not all(_symbolically_equal(a, b) for (a, _, _), (b, _, _) in zip(nd, rnd)):
                return None
        return sub

    @staticmethod
    def _rewrite_array_to_scalar(sdfg: SDFG, body_states: List[SDFGState], arr: str, desc: dt.Array) -> None:
        """Replace every access to the localized 1-D ``arr`` with a fresh ``[1]`` transient indexed at
        ``[0]``, then drop ``arr``. The ``[1]`` array (not a register scalar) persists across the two body
        states within one iteration, which is where the write and the read live."""
        local, _ = sdfg.add_array(arr + '_local', [1],
                                  desc.dtype,
                                  storage=desc.storage,
                                  transient=True,
                                  find_new_name=True)
        zero = subsets.Range([(0, 0, 1)])
        for s in body_states:
            for n in s.nodes():
                if not isinstance(n, nodes.AccessNode) or n.data != arr:
                    continue
                n.data = local
                for e in s.all_edges(n):
                    if e.data is not None and e.data.data == arr:
                        e.data.data = local
                        e.data.subset = copy.deepcopy(zero)
        try:
            sdfg.remove_data(arr, validate=False)
        except ValueError:
            pass  # still referenced by something outside our rewrite -> leave the dead array to simplify


__all__ = ['FuseLoops']
