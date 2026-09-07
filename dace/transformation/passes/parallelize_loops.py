# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift every parallelizable loop of an SDFG to a Map, outermost-first."""

from typing import Any, Dict, List, Optional, Tuple

from dace import properties
from dace.sdfg import SDFG
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.loop_to_map import (LiftContext, LiftInvariants, LoopToMap, build_lift_context,
                                                        build_lift_invariants)


def loop_order_key(loop: LoopRegion) -> Tuple[int, int]:
    """Sort key placing OUTERMOST loops first: nested-SDFG level, then control-flow nesting depth."""
    sdfg_level = 0
    sd = loop.sdfg
    while sd is not None and sd.parent_sdfg is not None:
        sdfg_level += 1
        sd = sd.parent_sdfg

    depth = 0
    graph = loop.parent_graph
    while graph is not None and not isinstance(graph, SDFG):
        depth += 1
        graph = graph.parent_graph
    return (sdfg_level, depth)


def candidate_loops(sdfg: SDFG) -> List[LoopRegion]:
    """Every loop in ``sdfg`` and its nested SDFGs that carries an iteration variable."""
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable]


@properties.make_properties
@transformation.explicit_cf_compatible
class ParallelizeLoops(ppl.Pass):
    """Lift every loop ``LoopToMap`` accepts into a Map, sweeping OUTERMOST-first.

    Three things this does that ``PatternMatchAndApplyRepeated([LoopToMap()])`` cannot:

    - **Order.** The matcher walks matches in graph order. Sweeping by nesting depth, shallowest
      first, measured 515.2s against the matcher's 927.7s on CloudSC for the SAME 314 maps. The
      opposite direction is not offered: deepest-first lifts an inner loop into a NestedSDFG whose
      outer memlet propagates wider, and the enclosing loop then fails ``LoopToMap``'s ``a*i+b``
      write check -- bottom-up reaches 288 maps, and losing 26 levels of parallelism is not a knob.
    - **Shared analysis.** Every ``can_be_applied`` probe re-derives the same per-SDFG facts: the
      symbol/array type map, which states hold an access node for a container, and a topological
      order of every block in the SDFG. :class:`LiftContext` builds them once and hands them to
      every probe of that SDFG; they are dropped after each applied lift.
    Refusals are deliberately NOT cached. A lift can make a loop liftable that is nowhere near it,
    so a refusal held across any apply costs parallelism -- measured on a three-nest kernel, caching
    them cost a map the plain matcher finds, because the matcher re-enumerates every candidate after
    every apply. What is reused is the ANALYSIS, not the verdict.

    Soundness rests on one invariant: every ``apply`` is immediately preceded by a full
    ``can_be_applied`` against the CURRENT graph. Nothing is applied on the strength of a verdict
    taken before another lift.

    Applies with no per-lift memlet propagation (like the matcher, which calls ``apply`` directly)
    and propagates the whole SDFG once at the end.
    """

    CATEGORY: str = 'Optimization Preparation'

    permissive = properties.Property(dtype=bool,
                                     default=False,
                                     desc='Probe loops in permissive mode, skipping the conservative refusals.')
    propagate = properties.Property(dtype=bool,
                                    default=True,
                                    desc='Propagate memlets over the whole SDFG once, after the last lift. '
                                    'Set False only if the caller propagates itself.')

    def __init__(self, permissive: bool = False, propagate: bool = True):
        self.permissive = permissive
        self.propagate = propagate

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Lift every liftable loop, outermost-first.

        :returns: The number of loops lifted, or ``None`` if none were.
        """
        applied = 0
        # Two lifetimes. The invariants are what a lift cannot change -- the symbol/array types, the
        # StructureView flag, and every block's free symbols -- so they are built once per SDFG and
        # never rebuilt. The contexts hold what a lift does change (the access-node index, the block
        # order, the cfg ids) and are dropped after every lift.
        invariants: Dict[SDFG, LiftInvariants] = {}
        contexts: Dict[SDFG, LiftContext] = {}
        # loop -> its read/write sets. A fact about the loop's own body, so only a lift INSIDE a
        # loop invalidates it -- see the ancestor walk after each apply.
        loop_read_write: Dict[Any, Any] = {}

        # Two fixpoints, outermost-first and then in graph order. Top-down wins on the big graphs
        # (CloudSC: 314 maps in 515.2s where graph order takes 927.7s for the same 314) but it is
        # not universally at least as good -- on polybench ``adi`` it reaches 4 maps where graph
        # order reaches 5, because lifting an outer loop can put a sibling behind a NestedSDFG
        # whose propagated memlet then fails the ``a*i+b`` write check. Sweeping graph order once
        # more afterwards costs one probe round and recovers those.
        for order in (loop_order_key, None):
            applied += self.lift_fixpoint(sdfg, pipeline_results, order, contexts, invariants, loop_read_write)

        if applied and self.propagate:
            propagate_memlets_sdfg(sdfg)
        return applied or None

    def lift_fixpoint(self, sdfg: SDFG, pipeline_results: Dict[str, Any], order, contexts: Dict[SDFG, LiftContext],
                      invariants: Dict[SDFG, LiftInvariants], loop_read_write: Dict[Any, Any]) -> int:
        """Lift until no loop in ``sdfg`` is accepted any more, visiting loops in ``order``.

        :param order: sort key over the candidate loops, or ``None`` to keep graph order.
        :param contexts: per-SDFG :class:`LiftContext` cache; cleared after every lift.
        :param invariants: per-SDFG facts a lift cannot change; built on first use, never rebuilt.
        :param loop_read_write: per-loop read/write sets; dropped for the ancestors of each lift.
        :returns: the number of loops lifted.
        """
        applied = 0
        # One instance, reused: ``setup_match`` overwrites every field a probe reads, and building a
        # ``make_properties`` object per candidate is pure overhead on a graph with hundreds of them.
        xform = LoopToMap()
        while True:
            lifted_one = False
            candidates = candidate_loops(sdfg)
            for loop in (candidates if order is None else sorted(candidates, key=order)):
                sd = loop.sdfg
                graph = loop.parent_graph
                inv = invariants.get(sd)
                if inv is None:
                    inv = invariants[sd] = build_lift_invariants(sd)
                ctx = contexts.get(sd)
                if ctx is None:
                    ctx = contexts[sd] = build_lift_context(sd, inv, loop_read_write)

                xform.lift_context = ctx
                # ``override=True`` with the loop OBJECT, the way ``fuse_states`` sets up its own
                # matches: ``PatternNode.__get__`` returns a non-int subgraph value as-is, so the
                # match resolves by identity instead of through ``cfg_list[cfg_id].node(node_id)``.
                # That drops two linear scans per candidate -- ``graph.node_id(loop)`` and the
                # ``cfg_list.index()`` inside ``cfg_id`` -- and removes any chance of a stale index
                # resolving to the wrong block.
                cfg_id = ctx.cfg_ids.get(graph)
                if cfg_id is None:
                    cfg_id = graph.cfg_id  # context predates this region; fall back to the scan
                xform.setup_match(sd, cfg_id, -1, {LoopToMap.loop: loop}, 0, override=True)
                xform._pipeline_results = pipeline_results

                if not xform.can_be_applied(graph, 0, sd, permissive=self.permissive):
                    continue

                xform.apply(graph, sd)
                applied += 1
                lifted_one = True
                # The graph changed, so every context is stale. The invariants are not: they are
                # exactly the analysis a lift cannot invalidate. The per-loop read/write sets sit in
                # between -- only a loop whose body now contains this lift has different ones.
                contexts.clear()
                loop_read_write.pop(loop, None)
                region = graph
                while region is not None and not isinstance(region, SDFG):
                    loop_read_write.pop(region, None)
                    region = region.parent_graph
                break  # restart the sweep so the next probe sees a current loop list and context

            if not lifted_one:
                return applied
