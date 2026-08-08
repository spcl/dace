# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reconstruct a perfect 2-D loop nest so ``WavefrontSkew`` can reach it.

``WavefrontSkew``'s ``extract_two_level_nest`` gate only accepts an outer ``LoopRegion``
whose body is a single inner ``LoopRegion`` (an IMPERFECT two-statement body is refused
outright, before any dependence analysis). The npbench-style stencil kernels (polybench
``seidel_2d``) instead lower to an outer loop whose body has TWO stateful siblings: the
frontend's slice-vectorized statement (``A[i, 1:-1] += ...``) becomes a Map-bearing state,
while the sequential in-row scan (``A[i, j] += A[i, j - 1]``) stays a ``LoopRegion`` -- an
imperfect nest ``WavefrontSkew`` never even inspects for dependences.

This pass reconstructs the perfect nest by composing two EXISTING transformations, adding no
dependence-legality logic of its own:

1. ``MapToForLoop`` turns the Map-bearing sibling back into a ``LoopRegion``.
2. ``FuseLoops`` merges it with the other sibling ``LoopRegion`` into one loop.

``FuseLoops`` normally refuses to fuse a sibling that is independently DOALL (correctly --
that guard is what keeps ``LoopFusion`` from serializing parallel work it should instead
leave to ``LoopToMap``). Reconstructing this nest is the one legitimate exception: the
Map-derived loop IS the DOALL sibling, and un-parallelizing it back into the sequential
scan is the entire point -- ``WavefrontSkew`` then re-parallelizes BOTH statements together
along the anti-diagonal, which is strictly more parallelism than leaving the Map standing
alone ever recovers. ``FuseLoops.can_be_applied`` gained a narrow, caller-scoped
``allow_doall_fuse`` keyword for exactly this; every other caller keeps the default refusal.

Gated tightly, cheapest checks first: (1) the outer loop's body must be EXACTLY the
two-sibling shape above, (2) the Map's range and the loop's range must already sweep the SAME
axis -- same trip count, same stride, any origin (a slice-normalized Map starting at 0 beside
a loop starting at 1 is exactly this shape, and the only one the pass has ever actually seen
in the corpus; the origin mismatch is a rebase for whichever step converts the Map, not a
reason to refuse here) -- checked here without paying for a conversion or a deepcopy, and only
then (3) the reconstruction is tried on a throwaway deepcopy and kept only if
``WavefrontSkew``'s own gate chain (``extract_two_level_nest`` + a legal ``tau``) accepts the
result -- reusing ``WavefrontSkew._try_skew`` itself, so this pass can never drift from what
the very next pipeline stage decides. A candidate that does not demonstrably unlock a skew
leaves the SDFG untouched.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.fuse_loops import FuseLoops
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import _symbolically_equal
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew

#: ``(map_state, inner_loop)``: the two stateful siblings of a reconstruction candidate.
_Candidate = Tuple[SDFGState, LoopRegion]


def _top_level_maps(state: SDFGState) -> List[nodes.MapEntry]:
    """Every ``MapEntry`` in ``state`` with no enclosing scope (a top-level parallel Map)."""
    return [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]


def _find_candidate(outer: LoopRegion) -> Optional[_Candidate]:
    """``(map_state, inner_loop)`` if ``outer``'s body is EXACTLY one ``SDFGState`` holding a
    top-level Map plus one sibling ``LoopRegion`` -- the imperfect two-statement shape
    ``WavefrontSkew.extract_two_level_nest`` refuses. ``None`` for any other shape."""
    blocks = list(outer.nodes())
    if len(blocks) != 2:
        return None
    states = [b for b in blocks if isinstance(b, SDFGState)]
    loops = [b for b in blocks if isinstance(b, LoopRegion) and b.loop_variable]
    if len(states) != 1 or len(loops) != 1:
        return None
    map_state = states[0]
    if not _top_level_maps(map_state):
        return None
    return map_state, loops[0]


def _trip_count(lo: Any, hi: Any, step: Any) -> Any:
    """Iteration count of an inclusive axis ``[lo, hi]`` stepping by ``step``."""
    return symbolic.int_floor(hi - lo, step) + 1


def _same_axis(m_lo: Any, m_hi: Any, m_step: Any, lo: Any, hi: Any, step: Any) -> bool:
    """Whether a Map axis and a loop axis are the SAME iteration axis: same trip count and
    same stride, any origin -- a slice-normalized Map starting at 0 beside a loop starting at
    1 is the same axis if both sweep the same number of steps. The origin itself is not lost:
    it stays exactly where it always was, on the Map's own stored range and the loop's own
    init statement, for whichever later step performs the actual rebase."""
    return _symbolically_equal(m_step, step) and _symbolically_equal(_trip_count(m_lo, m_hi, m_step),
                                                                     _trip_count(lo, hi, step))


def _ranges_match(map_state: SDFGState, inner_loop: LoopRegion) -> bool:
    """Whether every top-level Map in ``map_state`` sweeps the SAME iteration axis as
    ``inner_loop`` -- same trip count, same stride, any origin. A genuinely different trip
    count (e.g. a Map over ``0:M`` beside a loop over ``1:N-1`` with ``M`` symbolically
    unrelated to ``N``) is still refused. Checked directly against the Map's stored range (no
    conversion, no deepcopy) so a hopeless pair never pays for the expensive probe below."""
    lo = loop_analysis.get_init_assignment(inner_loop)
    hi = loop_analysis.get_loop_end(inner_loop)
    step = loop_analysis.get_loop_stride(inner_loop)
    if lo is None or hi is None or step is None:
        return False
    for entry in _top_level_maps(map_state):
        if len(entry.map.params) != 1:
            return False
        m_lo, m_hi, m_step = entry.map.range[0]
        if not _same_axis(m_lo, m_hi, m_step, lo, hi, step):
            return False
    return True


def _next_top_level_map(outer: LoopRegion) -> Optional[Tuple[SDFGState, nodes.MapEntry]]:
    """The first top-level Map in any direct-child state of ``outer``, re-scanned fresh on
    every call: converting one Map can relocate a not-yet-converted sibling Map into a fresh
    boundary state (``MapToForLoop``'s inline step), so no state reference is cached."""
    for block in outer.nodes():
        if isinstance(block, SDFGState):
            maps = _top_level_maps(block)
            if maps:
                return block, maps[0]
    return None


def _convert_maps_to_loops(outer: LoopRegion) -> bool:
    """Convert every top-level Map under ``outer`` to a ``LoopRegion`` via ``MapToForLoop``,
    one at a time until none remain. ``False`` if ``MapToForLoop`` itself refuses one (e.g. a
    surviving WCR reduction output) -- the shape cannot be reconstructed."""
    while True:
        found = _next_top_level_map(outer)
        if found is None:
            return True
        state, entry = found
        instance = MapToForLoop()
        instance.keep_reductions_parallel = True  # canon preference, off in the transformation's default contract
        instance.map_entry = entry
        if not instance.can_be_applied(state, expr_index=0, sdfg=state.sdfg, permissive=False):
            return False
        instance.apply(state, state.sdfg)


def _fuse_one_pair(sdfg: SDFG, outer: LoopRegion) -> bool:
    """Fuse the first legal consecutive sibling ``LoopRegion`` pair directly under ``outer``,
    opting in to serialize an independently-DOALL sibling (``allow_doall_fuse=True`` -- the
    whole point of this reconstruction). ``True`` iff a pair was fused."""
    for first in outer.nodes():
        if not isinstance(first, LoopRegion):
            continue
        out_edges = outer.out_edges(first)
        if len(out_edges) != 1:
            continue
        second = out_edges[0].dst
        if not isinstance(second, LoopRegion) or second is first:
            continue
        instance = FuseLoops()
        instance.first = first
        instance.second = second
        if instance.can_be_applied(outer, expr_index=0, sdfg=sdfg, permissive=False, allow_doall_fuse=True):
            instance.apply(outer, sdfg)
            return True
    return False


def _fuse_siblings_to_one(sdfg: SDFG, outer: LoopRegion) -> bool:
    """Fuse every sibling ``LoopRegion`` pair directly under ``outer`` to a fixpoint.
    ``True`` iff exactly one remains (a real cross-body dependence, or a leftover range
    mismatch ``FuseLoops`` itself refuses, can stall this short of one)."""
    changed = True
    while changed:
        changed = _fuse_one_pair(sdfg, outer)
        if changed:
            # ``FuseLoops`` appends the second body as an extra state rather than merging it into
            # the first, and ``_single_compute_state`` -- the very gate the NEXT pair is judged by --
            # accepts only a single-compute-state body. Without collapsing here a three-sibling nest
            # stalls after one fusion (seidel_2d: two Map-derived siblings plus the scan).
            PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
            EmptyStateElimination().apply_pass(sdfg, {})
    loops = [b for b in outer.nodes() if isinstance(b, LoopRegion) and b.loop_variable]
    return len(loops) == 1


def _reconstruct_body(sdfg: SDFG, outer: LoopRegion) -> bool:
    """Collapse ``outer``'s two-sibling body into a single ``LoopRegion``: every top-level Map
    to a loop, then every sibling loop pair fused to a fixpoint. ``False`` if either step
    cannot finish -- the caller must then leave ``sdfg`` exactly as it found it."""
    if not _convert_maps_to_loops(outer):
        return False
    EmptyStateElimination().apply_pass(sdfg, {})
    return _fuse_siblings_to_one(sdfg, outer)


def _locate_loop(sdfg: SDFG, target: LoopRegion) -> Optional[LoopRegion]:
    """The ``LoopRegion`` in ``sdfg`` corresponding to ``target`` in the SDFG ``sdfg`` was
    deep-copied from -- matched by label, an id-stable locator that survives
    ``copy.deepcopy`` (mirrors ``EarlyExitToFindIndex._locate_corresponding``)."""
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.label == target.label:
            return region
    return None


def _probe_unlocks_skew(sdfg: SDFG, outer: LoopRegion) -> bool:
    """Try the reconstruction on a throwaway deepcopy; ``True`` iff the result is a shape
    ``WavefrontSkew`` itself would then skew. Reuses ``WavefrontSkew._try_skew`` -- the
    pass's own gate chain, not reimplemented here -- so this probe can never drift from what
    the very next pipeline stage decides. Never mutates ``sdfg``."""
    trial_sdfg = copy.deepcopy(sdfg)
    trial_outer = _locate_loop(trial_sdfg, outer)
    if trial_outer is None or not _reconstruct_body(trial_sdfg, trial_outer):
        return False
    return WavefrontSkew()._try_skew(trial_outer, trial_sdfg)


@transformation.explicit_cf_compatible
class ReconstructWavefrontNest(ppl.Pass):
    """Reconstruct an imperfect 2-D stencil nest into the perfect single-loop body
    ``WavefrontSkew`` requires, but only where doing so demonstrably unlocks a skew.

    Composes two existing transformations (``MapToForLoop``, ``FuseLoops``) and adds no
    dependence-legality logic of its own; see the module docstring for the full rationale.
    A candidate is committed to the real SDFG only after an identical trial run on a deepcopy
    proves ``WavefrontSkew`` would then skew it -- a pass that does not apply must not mutate.
    """
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Reconstruct every qualifying candidate in ``sdfg`` and its nested SDFGs.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of reconstructions performed, or ``None`` if none.
        """
        reconstructed = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                for cfr in list(sd.all_control_flow_regions()):
                    if not (isinstance(cfr, LoopRegion) and cfr.loop_variable):
                        continue
                    if self._try_reconstruct(sd, cfr):
                        reconstructed += 1
                        changed = True
                        break
        return reconstructed or None

    @staticmethod
    def _try_reconstruct(sdfg: SDFG, outer: LoopRegion) -> bool:
        """Reconstruct ``outer``'s body in ``sdfg`` if the candidate shape is present, its
        range matches, and a throwaway deepcopy proves it unlocks a skew. ``True`` iff
        committed for real."""
        candidate = _find_candidate(outer)
        if candidate is None:
            return False
        map_state, inner_loop = candidate
        if not _ranges_match(map_state, inner_loop):
            return False
        if not _probe_unlocks_skew(sdfg, outer):
            return False
        # The probe just ran this exact deterministic algorithm on a deepcopy and it
        # succeeded; a refusal here would mean the two runs diverged -- an upstream bug,
        # not a normal outcome -- surfaced loudly rather than left as a half-mutated graph.
        if not _reconstruct_body(sdfg, outer):
            raise RuntimeError('ReconstructWavefrontNest: real commit diverged from its own probe')
        return True


__all__ = ['ReconstructWavefrontNest']
