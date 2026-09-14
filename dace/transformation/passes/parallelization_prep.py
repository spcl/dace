# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone parallelization-preparation passes.

These rewrite loops so that ``LoopToMap`` can parallelize more of them. They are
plain :class:`~dace.transformation.pass_pipeline.Pass` objects so a pipeline (and
anyone else) can just compose them:

- :class:`ShortLoopUnroll` -- fully unroll constant-trip loops with at most
  ``unroll_limit`` iterations, turning small recurrence / reduction loops into
  inline straight-line code instead of atomically-parallelized maps.

Transformation classes are imported lazily inside the methods: importing them at
module load would cycle (this package is imported by the transformations those
imports pull in).
"""
from typing import Any, Dict, Optional

from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Default trip-count threshold below which a constant-trip loop is unrolled. Matches the
#: ``optimizer.canonicalization.unroll_limit`` default on the extended branch.
DEFAULT_UNROLL_LIMIT = 8


def _loops(sdfg: SDFG):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _constant_trip_count(loop: LoopRegion, sdfg: SDFG) -> Optional[int]:
    """The exact iteration count of ``loop`` if it is constant, else ``None``.

    Ascending strides only: the ``stride_val <= 0`` bail deliberately declines DESCENDING loops, which
    this pass therefore never unrolls (they fall through to LoopToMap). ``LoopUnroll`` itself does
    handle a negative stride; widening this gate to match is a behaviour change for the pipelines that
    embed the pass, not part of that fix. With the bail in place the ``+ 1`` below is only ever reached
    for a positive stride, where it is the correct inclusive-end adjustment."""
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or not loop.loop_variable:
        return None
    if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
        return None
    try:
        stride_val = int(symbolic.evaluate(stride, sdfg.constants))
        diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
    except (TypeError, ValueError):
        return None
    if stride_val <= 0 or diff <= 0:
        return None
    return len(range(0, diff, stride_val))


def _loop_depth(loop: LoopRegion) -> int:
    """Nesting depth: number of enclosing control-flow regions up to the root SDFG. Used to order
    unrolling deepest-first (bottom-up)."""
    depth = 0
    graph = loop.parent_graph
    while graph is not None and not isinstance(graph, SDFG):
        depth += 1
        graph = graph.parent_graph
    return depth


def _local_state_fusion(sdfg: SDFG, region) -> int:
    """Fuse adjacent states within ``region``'s subtree only (StateFusionExtended), leaving the rest
    of the SDFG untouched. Interstate matching ignores ``apply_transformations``' ``states=`` filter,
    so drive the fusion on each adjacent pair directly. Returns the number fused."""
    from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
    fused = 0
    changed = True
    while changed:
        changed = False
        cfrs = [region] + list(region.all_control_flow_regions(recursive=True))
        for cfr in cfrs:
            for edge in list(cfr.edges()):
                u, v = edge.src, edge.dst
                if (isinstance(u, SDFGState) and isinstance(v, SDFGState)
                        and StateFusionExtended.can_be_applied_to(sdfg, first_state=u, second_state=v)):
                    StateFusionExtended.apply_to(sdfg,
                                                 first_state=u,
                                                 second_state=v,
                                                 verify=False,
                                                 annotate=False,
                                                 save=False)
                    fused += 1
                    changed = True
                    break
            if changed:
                break
    return fused


@properties.make_properties
class ShortLoopUnroll(ppl.Pass):
    """Fully unroll every constant-trip loop with at most ``unroll_limit`` iterations.

    Unrolls **bottom-up** (deepest loops first) and fuses the freshly unrolled states back down right
    after each unroll -- scoped to just the touched region, not the whole SDFG. So an enclosing loop
    deepcopies an already-compacted, loop-free body instead of a fan-out of one-state-per-iterate
    sub-loops: far less deepcopy volume and no intermediate blow-up (measured 6.5x faster on CloudSC vs
    unroll-then-global-fuse), so it is unconditional."""

    CATEGORY: str = 'Optimization Preparation'

    unroll_limit = properties.Property(
        dtype=int,
        default=DEFAULT_UNROLL_LIMIT,
        desc='Fully unroll constant-trip loops with at most this many iterations (0 disables).')

    def __init__(self, unroll_limit: int = DEFAULT_UNROLL_LIMIT):
        self.unroll_limit = unroll_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Unroll short constant-trip loops.

        Re-collects after each unroll since unrolling rewrites the control-flow
        structure (and may expose newly-constant inner loops).

        :returns: The number of loops unrolled; ``0`` when no unroll completed but one raised
                  part-way through ``apply`` (see below), leaving a possibly half-rewritten
                  loop; ``None`` only when the SDFG was left untouched.

        ``apply_pass`` returning ``None`` means "did not modify the SDFG", and callers act
        on it: the pipeline skips its per-stage ``validate()`` and leaves ``self._modified``
        alone (stale analyses, early ``FixedPointPipeline`` exit). A partial rewrite reported
        as ``None`` would therefore go unvalidated, so it reports ``0`` -- "modified, but
        nothing of my own kind completed".
        """
        if self.unroll_limit <= 0:
            return None
        from dace.transformation.interstate.loop_unroll import LoopUnroll
        unrolled = 0
        # Unrolls that raised part-way through ``LoopUnroll.apply``. Counted separately from
        # ``unrolled`` so they feed the return value (the graph may be half-rewritten) without
        # triggering the completed-unroll propagation below.
        partial = 0
        changed = True
        while changed:
            changed = False
            # Bottom-up: unroll the deepest loops first, so an enclosing loop is only unrolled once its
            # inner loops are already unrolled + locally fused into a compact body.
            for loop in sorted(_loops(sdfg), key=_loop_depth, reverse=True):
                trip = _constant_trip_count(loop, sdfg)
                if trip is None or trip > self.unroll_limit:
                    continue
                parent = loop.parent_graph
                # Applicability is decided FIRST, on its own, so a refusal is distinguishable
                # from a failure raised part-way through ``apply``. A refusal leaves the graph
                # untouched and must not be reported as a modification; a mid-``apply`` failure
                # can leave the loop half-rewritten and must be. ``apply_to`` below therefore
                # runs with ``verify=False`` -- ``can_be_applied`` still runs exactly once, so
                # this is the same check sequence as before, just with the outcome visible here.
                try:
                    applicable = LoopUnroll.can_be_applied_to(sdfg=loop.sdfg, loop=loop)
                except Exception:
                    applicable = False
                if not applicable:
                    continue  # not unrollable in this context; leave it for LoopToMap
                try:
                    # ``annotate=False``: skip the per-apply full-SDFG memlet/state
                    # propagation. The transformation framework otherwise re-runs it
                    # after EVERY unroll -- O(unrolls x sdfg_size) redundant work
                    # (the dominant ~83% cost of unrolling a d-deep trip-t tile
                    # nest, whose SDFG grows to ~t^d blocks). The trip-count / loop
                    # re-collection below reads only loop bounds, not memlets, so
                    # the interim annotations are never observed; one propagation
                    # after the whole fixpoint (below) refreshes them.
                    LoopUnroll().apply_to(sdfg=loop.sdfg, loop=loop, annotate=False, verify=False)
                except Exception:
                    # Raised from inside ``apply``: the rewrite may be half-done, so this
                    # counts as a modification even though no loop was fully unrolled.
                    partial += 1
                    continue
                unrolled += 1
                changed = True
                if parent is not None:
                    # Compact the just-unrolled region before an enclosing loop deepcopies it.
                    _local_state_fusion(sdfg, parent)
                break
        if unrolled:
            # Propagate once, at the end of the pass (not per-apply).
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
            return unrolled
        return 0 if partial else None
